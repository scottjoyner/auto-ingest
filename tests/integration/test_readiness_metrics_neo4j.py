from __future__ import annotations

import os
import time

import pytest
from neo4j import GraphDatabase

from auto_ingest.metrics import collect_metrics
from auto_ingest.readiness import readiness
from auto_ingest.runtime_schema import ensure_schema

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
        s.run("MATCH (n) WHERE n.fixture_scope='readiness-ci' DETACH DELETE n").consume()
    yield
    with driver.session(database="neo4j") as s:
        s.run("MATCH (n) WHERE n.fixture_scope='readiness-ci' DETACH DELETE n").consume()


def test_readiness_requires_identity_schema(driver):
    ensure_schema(driver)
    report = readiness(driver)
    assert report["ready"] is True
    assert report["database"] is True
    assert report["schema"]["ok"] is True


def test_metrics_surface_queue_pressure_and_stale_jobs(driver):
    ensure_schema(driver)
    old = int(time.time() * 1000) - 60_000
    with driver.session(database="neo4j") as s:
        s.run(
            """
            CREATE (:IngestJob {
                key:'ci-ready-running', lifecycle_state:'RUNNING', owner:'worker-a',
                claimed_at:$old, heartbeat_at:$old, fixture_scope:'readiness-ci'
            }),
            (:IngestJob {
                key:'ci-ready-quarantine', lifecycle_state:'QUARANTINED', owner:'',
                claimed_at:0, fixture_scope:'readiness-ci'
            }),
            (:IngestArtifact {artifact_id:'ci-ready-artifact', fixture_scope:'readiness-ci'})
            """,
            old=old,
        ).consume()

    metrics = collect_metrics(driver, stale_after_sec=10)
    assert metrics["jobs_by_state"]["RUNNING"] >= 1
    assert metrics["jobs_by_state"]["QUARANTINED"] >= 1
    assert metrics["active_leases"] >= 1
    assert metrics["stale_jobs"] >= 1
    assert metrics["artifacts"] >= 1
