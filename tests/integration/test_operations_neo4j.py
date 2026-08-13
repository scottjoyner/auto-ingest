from __future__ import annotations

import os

import pytest
from neo4j import GraphDatabase

from auto_ingest.operations import inspect_job, quarantine_job, retry_job

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
        s.run("MATCH (n) WHERE n.fixture_scope='operations-ci' DETACH DELETE n").consume()
        s.run("MATCH (j:IngestJob) WHERE j.key STARTS WITH 'ci-ops:' DETACH DELETE j").consume()
    yield
    with driver.session(database="neo4j") as s:
        s.run("MATCH (n) WHERE n.fixture_scope='operations-ci' DETACH DELETE n").consume()
        s.run("MATCH (j:IngestJob) WHERE j.key STARTS WITH 'ci-ops:' DETACH DELETE j").consume()


def test_quarantine_and_explicit_retry_are_single_job_bounded(driver):
    key = "ci-ops:one"
    other = "ci-ops:other"
    with driver.session(database="neo4j") as s:
        s.run(
            """
            CREATE (:IngestJob {key:$key, lifecycle_state:'RUNNING', owner:'w', claimed_at:123}),
                   (:IngestJob {key:$other, lifecycle_state:'RUNNING', owner:'w2', claimed_at:456})
            """,
            key=key,
            other=other,
        ).consume()

    assert quarantine_job(driver, key, "operator isolation")
    info = inspect_job(driver, key)
    assert info["job"]["state"] == "QUARANTINED"
    assert info["job"]["owner"] == ""

    untouched = inspect_job(driver, other)
    assert untouched["job"]["state"] == "RUNNING"
    assert untouched["job"]["owner"] == "w2"

    assert retry_job(driver, key)
    retried = inspect_job(driver, key)
    assert retried["job"]["state"] == "READY"
    assert retried["job"]["owner"] == ""


def test_retry_refuses_healthy_running_job(driver):
    key = "ci-ops:running"
    with driver.session(database="neo4j") as s:
        s.run(
            "CREATE (:IngestJob {key:$key, lifecycle_state:'RUNNING', owner:'worker'})",
            key=key,
        ).consume()
    assert retry_job(driver, key) is False


def test_inspect_returns_bounded_artifact_provenance(driver):
    key = "ci-ops:inspect"
    with driver.session(database="neo4j") as s:
        s.run(
            """
            CREATE (j:IngestJob {key:$key, lifecycle_state:'DONE', owner:''})
            CREATE (a:IngestArtifact {
                artifact_id:'ci-artifact', stage:'copied', stage_version:'1',
                path:'/tmp/a', sha256:'abc', fence_token:1,
                fixture_scope:'operations-ci'
            })
            CREATE (j)-[:PRODUCED]->(a)
            """,
            key=key,
        ).consume()
    info = inspect_job(driver, key)
    assert info["job"]["state"] == "DONE"
    assert info["artifacts"] == [
        {
            "artifact_id": "ci-artifact",
            "stage": "copied",
            "stage_version": "1",
            "path": "/tmp/a",
            "sha256": "abc",
            "fence_token": 1,
        }
    ]
