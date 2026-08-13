from __future__ import annotations

import os

import pytest
from neo4j import GraphDatabase

from auto_ingest import ingest_claim
from auto_ingest.watchdog import recover_stale, stale_jobs

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
        s.run("MATCH (j:IngestJob) WHERE j.key STARTS WITH 'ci-watch:' DETACH DELETE j").consume()
    yield
    with driver.session(database="neo4j") as s:
        s.run("MATCH (j:IngestJob) WHERE j.key STARTS WITH 'ci-watch:' DETACH DELETE j").consume()


def _job(driver, key: str, state: str, owner: str, last_seen: int, recoveries: int = 0):
    with driver.session(database="neo4j") as s:
        s.run(
            """
            CREATE (:IngestJob {
                key:$key, lifecycle_state:$state, owner:$owner,
                claimed_at:$last_seen, heartbeat_at:$last_seen,
                fence_token:7, stale_recovery_count:$recoveries
            })
            """,
            key=key,
            state=state,
            owner=owner,
            last_seen=last_seen,
            recoveries=recoveries,
        ).consume()


def test_watchdog_dry_listing_and_mutation_are_hard_bounded(driver):
    for i in range(5):
        _job(driver, f"ci-watch:{i}", "RUNNING", f"w{i}", 1_000 + i)
    rows = stale_jobs(driver, stale_after_sec=1, limit=2, now_ms=10_000)
    assert len(rows) == 2

    changed = recover_stale(driver, stale_after_sec=1, limit=2, now_ms=10_000)
    assert len(changed) == 2
    with driver.session(database="neo4j") as s:
        retry = s.run(
            "MATCH (j:IngestJob) WHERE j.key STARTS WITH 'ci-watch:' "
            "AND j.lifecycle_state='RETRY' RETURN count(j) AS n"
        ).single()["n"]
        running = s.run(
            "MATCH (j:IngestJob) WHERE j.key STARTS WITH 'ci-watch:' "
            "AND j.lifecycle_state='RUNNING' RETURN count(j) AS n"
        ).single()["n"]
    assert retry == 2
    assert running == 3


def test_watchdog_never_touches_terminal_or_fresh_jobs(driver):
    _job(driver, "ci-watch:done", "DONE", "old", 1_000)
    _job(driver, "ci-watch:quarantine", "QUARANTINED", "old", 1_000)
    _job(driver, "ci-watch:fresh", "RUNNING", "fresh", 9_500)
    assert recover_stale(driver, stale_after_sec=1, limit=10, now_ms=10_000) == []


def test_repeated_stale_recovery_quarantines_and_new_claim_gets_new_fence(driver):
    key = "ci-watch:repeat"
    _job(driver, key, "RUNNING", "dead-worker", 1_000, recoveries=1)
    changed = recover_stale(
        driver,
        stale_after_sec=1,
        limit=10,
        max_stale_recoveries=2,
        now_ms=10_000,
    )
    assert changed[0]["state"] == "QUARANTINED"
    with driver.session(database="neo4j") as s:
        row = s.run(
            "MATCH (j:IngestJob {key:$key}) RETURN j.owner AS owner, "
            "j.fence_token AS fence, j.last_abandoned_fence AS abandoned",
            key=key,
        ).single()
    assert row["owner"] == ""
    assert row["fence"] == 7
    assert row["abandoned"] == 7

    # Explicit operator retry is required after quarantine; simulate that narrow
    # state transition and prove the next claimant advances the fence generation.
    with driver.session(database="neo4j") as s:
        s.run(
            "MATCH (j:IngestJob {key:$key}) SET j.lifecycle_state='READY'",
            key=key,
        ).consume()
    token = ingest_claim.claim_fenced(driver, key, "replacement", ttl_sec=10, now_ms=11_000)
    assert token == 8
