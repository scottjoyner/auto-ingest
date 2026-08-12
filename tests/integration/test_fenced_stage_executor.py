from __future__ import annotations

import os
from pathlib import Path

import pytest
from neo4j import GraphDatabase

from auto_ingest import ingest_claim
from auto_ingest.stage_executor import (
    LeaseLost,
    STATE_ARTIFACT_COMMITTED,
    STATE_COMMITTED,
    execute_stage,
    read_journal,
    recover_artifact,
)

pytestmark = [pytest.mark.integration, pytest.mark.destructive]


@pytest.fixture()
def driver():
    uri = os.environ.get("NEO4J_TEST_URI")
    password = os.environ.get("NEO4J_TEST_PASSWORD")
    if not uri or not password:
        pytest.skip("Neo4j integration environment is not configured")
    drv = GraphDatabase.driver(
        uri,
        auth=(os.environ.get("NEO4J_TEST_USER", "neo4j"), password),
    )
    drv.verify_connectivity()
    yield drv
    drv.close()


@pytest.fixture(autouse=True)
def cleanup(driver):
    with driver.session(database="neo4j") as s:
        s.run("MATCH (n) WHERE n.fixture_scope='fenced-stage-ci' DETACH DELETE n").consume()
        s.run("MATCH (j:IngestJob) WHERE j.key STARTS WITH 'ci-fenced:' DETACH DELETE j").consume()
    yield
    with driver.session(database="neo4j") as s:
        s.run("MATCH (n) WHERE n.fixture_scope='fenced-stage-ci' DETACH DELETE n").consume()
        s.run("MATCH (j:IngestJob) WHERE j.key STARTS WITH 'ci-fenced:' DETACH DELETE j").consume()


def test_stale_worker_cannot_commit_after_lease_takeover(driver):
    key = "ci-fenced:takeover"
    ingest_claim.create_job(driver, key)
    token_a = ingest_claim.claim_fenced(driver, key, "worker-a", ttl_sec=1, now_ms=1_000)
    assert token_a == 1
    token_b = ingest_claim.claim_fenced(driver, key, "worker-b", ttl_sec=1, now_ms=3_000)
    assert token_b == 2
    assert ingest_claim.update_stage_fenced(
        driver, key, "copied", owner="worker-a", fence_token=token_a
    ) is None
    fresh = ingest_claim.update_stage_fenced(
        driver, key, "copied", owner="worker-b", fence_token=token_b
    )
    assert fresh and fresh["stages"]["copied"] is True
    assert not ingest_claim.release_fenced(driver, key, "worker-a", token_a)
    assert ingest_claim.release_fenced(driver, key, "worker-b", token_b)


def test_stage_executor_commits_artifact_graph_and_state(driver, tmp_path: Path):
    key = "ci-fenced:happy"
    ingest_claim.create_job(driver, key)

    def graph_commit(drv, commit):
        with drv.session(database="neo4j") as s:
            s.run(
                """
                MERGE (a:Artifact {artifact_id:$artifact_id})
                SET a.fixture_scope='fenced-stage-ci',
                    a.sha256=$sha, a.path=$path, a.fence_token=$token
                """,
                artifact_id=commit.artifact_id,
                sha=commit.artifact_sha256,
                path=commit.artifact_path,
                token=commit.fence_token,
            ).consume()

    result = execute_stage(
        driver,
        job_key=key,
        owner="worker-a",
        stage="copied",
        stage_version="1",
        source_hash="source-hash",
        artifact_root=tmp_path,
        artifact_bytes=b"canonical bytes",
        artifact_suffix="bin",
        config={"mode": "ci"},
        graph_commit=graph_commit,
    )
    assert Path(result.artifact_path).read_bytes() == b"canonical bytes"
    assert recover_artifact(tmp_path, result.artifact_id) == STATE_COMMITTED
    status = ingest_claim.stage_status(driver, key)
    assert status["stages"]["copied"] is True
    with driver.session(database="neo4j") as s:
        count = s.run(
            "MATCH (a:Artifact {artifact_id:$id, fixture_scope:'fenced-stage-ci'}) RETURN count(a) AS n",
            id=result.artifact_id,
        ).single()["n"]
    assert count == 1


def test_crash_after_artifact_is_recoverable_and_retry_reuses_bytes(driver, tmp_path: Path):
    key = "ci-fenced:crash"
    ingest_claim.create_job(driver, key)
    captured = {}

    def fail(point):
        if point == "after_artifact":
            raise RuntimeError("simulated crash")

    with pytest.raises(RuntimeError):
        execute_stage(
            driver,
            job_key=key,
            owner="worker-a",
            stage="copied",
            stage_version="1",
            source_hash="source-hash-crash",
            artifact_root=tmp_path,
            artifact_bytes=b"payload",
            artifact_suffix="bin",
            fault=fail,
        )

    journals = list((tmp_path / ".commit-journal").rglob("*.json"))
    assert len(journals) == 1
    record = read_journal(tmp_path, journals[0].stem)
    assert record["state"] == STATE_ARTIFACT_COMMITTED
    assert recover_artifact(tmp_path, journals[0].stem) == STATE_ARTIFACT_COMMITTED

    # Expire/take over the original crashed lease, then rerun. Deterministic
    # identity means the already-durable bytes are reused instead of duplicated.
    status = ingest_claim.stage_status(driver, key)
    with driver.session(database="neo4j") as s:
        s.run(
            "MATCH (j:IngestJob {key:$key}) SET j.claimed_at=0",
            key=key,
        ).consume()

    result = execute_stage(
        driver,
        job_key=key,
        owner="worker-b",
        stage="copied",
        stage_version="1",
        source_hash="source-hash-crash",
        artifact_root=tmp_path,
        artifact_bytes=b"payload",
        artifact_suffix="bin",
    )
    assert result.reused is True
    assert recover_artifact(tmp_path, result.artifact_id) == STATE_COMMITTED


def test_executor_rejects_stale_generation_before_graph_commit(driver, tmp_path: Path):
    key = "ci-fenced:stale-during-run"
    ingest_claim.create_job(driver, key)

    def steal(point):
        if point == "after_artifact":
            with driver.session(database="neo4j") as s:
                s.run("MATCH (j:IngestJob {key:$key}) SET j.claimed_at=0", key=key).consume()
            token = ingest_claim.claim_fenced(driver, key, "worker-b", ttl_sec=1, now_ms=9_000)
            assert token is not None

    with pytest.raises(LeaseLost):
        execute_stage(
            driver,
            job_key=key,
            owner="worker-a",
            stage="copied",
            stage_version="1",
            source_hash="source-stale",
            artifact_root=tmp_path,
            artifact_bytes=b"payload",
            fault=steal,
        )
