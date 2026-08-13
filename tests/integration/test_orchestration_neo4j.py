from __future__ import annotations

import os

import pytest
from neo4j import GraphDatabase

from auto_ingest import ingest_claim
from auto_ingest.orchestration import (
    DONE,
    QUARANTINED,
    RETRY,
    Task,
    heartbeat,
    lifecycle,
    run_profile,
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
    with driver.session(database="neo4j") as session:
        session.run(
            "MATCH (j:IngestJob) WHERE j.key STARTS WITH 'ci-orch:' DETACH DELETE j"
        ).consume()
    yield
    with driver.session(database="neo4j") as session:
        session.run(
            "MATCH (j:IngestJob) WHERE j.key STARTS WITH 'ci-orch:' DETACH DELETE j"
        ).consume()


class _Proc:
    def __init__(self, rc):
        self.rc = rc

    def poll(self):
        return self.rc

    def terminate(self):
        return None

    def kill(self):
        return None

    def wait(self, timeout=None):
        return self.rc


def test_failed_pipeline_resumes_after_last_committed_task(driver):
    key = "ci-orch:resume"
    tasks = (Task("copy", ("copy",)), Task("transcribe", ("transcribe",)))
    calls = []
    outcomes = {"copy": [0], "transcribe": [7, 0]}

    def factory(command, **_kwargs):
        name = command[0]
        calls.append(name)
        return _Proc(outcomes[name].pop(0))

    rc1 = run_profile(
        driver,
        "test",
        job_key=key,
        owner="worker-a",
        tasks=tasks,
        ttl_sec=10,
        heartbeat_sec=1,
        popen_factory=factory,
    )
    assert rc1 == 1
    state1 = lifecycle(driver, key)
    assert state1["state"] == RETRY
    assert state1["completed_tasks"] == ["copy"]
    assert state1["failed_task"] == "transcribe"
    assert state1["error_fingerprint"]

    rc2 = run_profile(
        driver,
        "test",
        job_key=key,
        owner="worker-b",
        tasks=tasks,
        ttl_sec=10,
        heartbeat_sec=1,
        popen_factory=factory,
    )
    assert rc2 == 0
    assert calls == ["copy", "transcribe", "transcribe"]
    state2 = lifecycle(driver, key)
    assert state2["state"] == DONE
    assert state2["completed_tasks"] == ["copy", "transcribe"]
    assert state2["attempts"] == 2
    assert state2["fence_token"] == 2


def test_repeated_identical_failure_quarantines_job(driver):
    key = "ci-orch:quarantine"
    tasks = (Task("explode", ("explode",)),)

    def factory(_command, **_kwargs):
        return _Proc(9)

    for owner in ("worker-a", "worker-b"):
        assert run_profile(
            driver,
            "test",
            job_key=key,
            owner=owner,
            tasks=tasks,
            ttl_sec=10,
            heartbeat_sec=1,
            max_attempts=2,
            popen_factory=factory,
        ) == 1

    state = lifecycle(driver, key)
    assert state["state"] == QUARANTINED
    assert state["attempts"] == 2
    assert state["error_type"] == "RuntimeError"
    assert "exited with code 9" in state["error_message"]

    # A quarantined job cannot burn more compute until explicitly remediated.
    assert run_profile(
        driver,
        "test",
        job_key=key,
        owner="worker-c",
        tasks=tasks,
        ttl_sec=10,
        heartbeat_sec=1,
        max_attempts=2,
        popen_factory=factory,
    ) == 3
    assert lifecycle(driver, key)["attempts"] == 2


def test_stale_worker_cannot_renew_heartbeat_after_takeover(driver):
    key = "ci-orch:heartbeat"
    from auto_ingest.orchestration import ensure_job

    ensure_job(driver, key, "test")
    token_a = ingest_claim.claim_fenced(driver, key, "worker-a", ttl_sec=1, now_ms=1_000)
    assert token_a == 1
    assert heartbeat(driver, key, owner="worker-a", fence_token=token_a)

    with driver.session(database="neo4j") as session:
        session.run(
            "MATCH (j:IngestJob {key:$key}) SET j.claimed_at=0",
            key=key,
        ).consume()
    token_b = ingest_claim.claim_fenced(driver, key, "worker-b", ttl_sec=1, now_ms=3_000)
    assert token_b == 2
    assert not heartbeat(driver, key, owner="worker-a", fence_token=token_a)
    assert heartbeat(driver, key, owner="worker-b", fence_token=token_b)
