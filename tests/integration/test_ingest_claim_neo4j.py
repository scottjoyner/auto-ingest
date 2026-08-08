"""Real Neo4j contract tests for durable ingest claims and stage state."""
from __future__ import annotations

import os

import pytest
from neo4j import GraphDatabase

from auto_ingest import ingest_claim

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
def clean_fixture_jobs(driver):
    with driver.session(database="neo4j") as session:
        session.run(
            "MATCH (j:IngestJob) WHERE j.key STARTS WITH 'ci-hardening:' DETACH DELETE j"
        ).consume()
    yield
    with driver.session(database="neo4j") as session:
        session.run(
            "MATCH (j:IngestJob) WHERE j.key STARTS WITH 'ci-hardening:' DETACH DELETE j"
        ).consume()


def test_real_job_manifest_uses_supported_property_types_and_is_idempotent(driver):
    key = "ci-hardening:idempotent"
    first = ingest_claim.create_job(driver, key)
    assert first["status"] == "pending"
    assert first["stages"] == {stage: False for stage in ingest_claim.STAGES}

    assert ingest_claim.claim(driver, key, "worker-a", now_ms=1_000_000)
    updated = ingest_claim.update_stage(driver, key, "copied", owner="worker-a")
    assert updated and updated["status"] == "running"

    second = ingest_claim.create_job(driver, key)
    assert second["owner"] == "worker-a"
    assert second["status"] == "running"
    assert second["stages"]["copied"] is True
    assert second["attempt_count"] == 1

    with driver.session(database="neo4j") as session:
        record = session.run(
            """
            MATCH (j:IngestJob {key:$key})
            RETURN j.completed_stages AS completed,
                   j.attempt_count AS attempts,
                   j.owner AS owner
            """,
            key=key,
        ).single()
    assert record["completed"] == ["copied"]
    assert record["attempts"] == 1
    assert record["owner"] == "worker-a"


def test_real_claim_owner_guard_and_terminal_state(driver):
    key = "ci-hardening:owner-guard"
    ingest_claim.create_job(driver, key)
    assert ingest_claim.claim(driver, key, "worker-a", now_ms=5_000_000)
    assert not ingest_claim.claim(driver, key, "worker-b", now_ms=5_000_001)
    assert ingest_claim.update_stage(driver, key, "embedded", owner="worker-b") is None
    assert ingest_claim.update_stage(driver, key, "embedded", owner="worker-a")
    terminal = ingest_claim.update_stage(driver, key, "graph_written", owner="worker-a")
    assert terminal and terminal["status"] == "done"
    assert ingest_claim.release(driver, key, "worker-a")
    assert ingest_claim.stage_status(driver, key)["status"] == "done"


def test_real_reaper_limit_bounds_mutation(driver):
    now = 100_000_000
    for idx in range(3):
        key = f"ci-hardening:expired:{idx}"
        ingest_claim.create_job(driver, key)
        assert ingest_claim.claim(
            driver,
            key,
            f"worker-{idx}",
            ttl_sec=3600,
            now_ms=now - (7200 + idx) * 1000,
        )

    assert len(ingest_claim.expired_claims(driver, now_ms=now, limit=10)) == 3
    assert ingest_claim.reap(driver, now_ms=now, limit=1) == 1
    assert len(ingest_claim.expired_claims(driver, now_ms=now, limit=10)) == 2
