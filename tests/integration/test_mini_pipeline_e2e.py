from __future__ import annotations

import os
from pathlib import Path

import pytest
from neo4j import GraphDatabase

from auto_ingest.artifacts import build_identity, sha256_bytes
from auto_ingest.commit_protocol import atomic_commit_bytes
from auto_ingest.ingest_claim import (
    claim_fenced,
    create_job,
    release_fenced,
    stage_status,
    update_stage_fenced,
)
from auto_ingest.runtime_schema import ensure_schema

pytestmark = [pytest.mark.integration, pytest.mark.destructive]

JOB_KEY = "ci-mini-e2e:fixture"


@pytest.fixture()
def driver():
    uri = os.environ.get("NEO4J_TEST_URI")
    secret = os.environ.get("NEO4J_TEST_PASSWORD")
    if not uri or not secret:
        pytest.skip("Neo4j integration environment is not configured")
    drv = GraphDatabase.driver(
        uri,
        auth=(os.environ.get("NEO4J_TEST_USER", "neo4j"), secret),
    )
    drv.verify_connectivity()
    yield drv
    drv.close()


@pytest.fixture(autouse=True)
def cleanup(driver):
    with driver.session(database="neo4j") as session:
        session.run(
            "MATCH (j:IngestJob {key:$key}) OPTIONAL MATCH (j)-[:PRODUCED]->(a:IngestArtifact) "
            "DETACH DELETE j, a",
            key=JOB_KEY,
        ).consume()
    yield
    with driver.session(database="neo4j") as session:
        session.run(
            "MATCH (j:IngestJob {key:$key}) OPTIONAL MATCH (j)-[:PRODUCED]->(a:IngestArtifact) "
            "DETACH DELETE j, a",
            key=JOB_KEY,
        ).consume()


def _register_artifact(driver, *, token: int, artifact_id: str, path: Path, digest: str, source_hash: str):
    with driver.session(database="neo4j") as session:
        rec = session.run(
            """
            MATCH (j:IngestJob {key:$key})
            WHERE j.owner=$owner AND coalesce(j.fence_token,0)=$token
            MERGE (a:IngestArtifact {artifact_id:$artifact_id})
            ON CREATE SET a.path=$path, a.sha256=$digest, a.source_hash=$source_hash,
                          a.stage='transcribed', a.stage_version='1'
            MERGE (j)-[:PRODUCED]->(a)
            RETURN a.artifact_id AS artifact_id
            """,
            key=JOB_KEY,
            owner="mini-worker",
            token=token,
            artifact_id=artifact_id,
            path=str(path),
            digest=digest,
            source_hash=source_hash,
        ).single()
    return rec["artifact_id"] if rec else None


def test_mini_pipeline_converges_exactly_once_across_retry(driver, tmp_path: Path):
    ensure_schema(driver)
    source = b"tiny deterministic transcript fixture\n"
    source_hash = sha256_bytes(source)
    identity = build_identity(
        source_hash=source_hash,
        stage="transcribed",
        stage_version="1",
        config={"language": "en", "fixture": True},
        model={"name": "synthetic", "version": "1"},
    )
    artifact_path = tmp_path / f"{identity.artifact_id}.json"

    created = create_job(driver, JOB_KEY)
    assert created["status"] == "pending"
    token = claim_fenced(driver, JOB_KEY, "mini-worker", ttl_sec=30, now_ms=10_000)
    assert token == 1

    assert update_stage_fenced(driver, JOB_KEY, "copied", owner="mini-worker", fence_token=token)
    commit = atomic_commit_bytes(artifact_path, source)
    assert commit.reused is False
    assert update_stage_fenced(driver, JOB_KEY, "transcribed", owner="mini-worker", fence_token=token)
    assert _register_artifact(
        driver,
        token=token,
        artifact_id=identity.artifact_id,
        path=artifact_path,
        digest=commit.sha256,
        source_hash=source_hash,
    ) == identity.artifact_id

    for stage in ("diarized", "embedded", "linked", "graph_written"):
        assert update_stage_fenced(
            driver,
            JOB_KEY,
            stage,
            owner="mini-worker",
            fence_token=token,
        )
    status = stage_status(driver, JOB_KEY)
    assert status and status["status"] == "done"
    assert all(status["stages"].values())
    assert release_fenced(driver, JOB_KEY, "mini-worker", token) is True

    # Retry the same semantic input. Filesystem publication is reused and graph
    # identity/provenance MERGEs converge rather than duplicating output.
    retry_token = claim_fenced(driver, JOB_KEY, "mini-worker", ttl_sec=30, now_ms=20_000)
    assert retry_token == 2
    retry_commit = atomic_commit_bytes(
        artifact_path,
        source,
        expected_sha256=commit.sha256,
    )
    assert retry_commit.reused is True
    assert _register_artifact(
        driver,
        token=retry_token,
        artifact_id=identity.artifact_id,
        path=artifact_path,
        digest=retry_commit.sha256,
        source_hash=source_hash,
    ) == identity.artifact_id

    with driver.session(database="neo4j") as session:
        row = session.run(
            """
            MATCH (j:IngestJob {key:$key})
            OPTIONAL MATCH (j)-[r:PRODUCED]->(a:IngestArtifact)
            RETURN count(DISTINCT j) AS jobs,
                   count(DISTINCT a) AS artifacts,
                   count(DISTINCT r) AS produced,
                   max(j.fence_token) AS fence
            """,
            key=JOB_KEY,
        ).single()
    assert dict(row) == {"jobs": 1, "artifacts": 1, "produced": 1, "fence": 2}
