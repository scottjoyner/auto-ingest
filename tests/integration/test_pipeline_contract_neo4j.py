from __future__ import annotations

import os

import pytest
from neo4j import GraphDatabase

from auto_ingest.orchestration import Task, ensure_job
from auto_ingest.pipeline_contract import PipelinePlanDrift, bind_plan, plan_hash

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
        s.run("MATCH (j:IngestJob) WHERE j.key STARTS WITH 'ci-plan:' DETACH DELETE j").consume()
    yield
    with driver.session(database="neo4j") as s:
        s.run("MATCH (j:IngestJob) WHERE j.key STARTS WITH 'ci-plan:' DETACH DELETE j").consume()


def test_plan_can_rebind_before_progress_but_not_after(driver):
    key = "ci-plan:drift"
    original = (Task("copy", ("copy-v1",), 30), Task("graph", ("graph-v1",), 30))
    changed = (Task("copy", ("copy-v2",), 30), Task("graph", ("graph-v1",), 30))
    ensure_job(driver, key, "ci")

    first = bind_plan(driver, key, original)
    assert first == plan_hash(original)
    second = bind_plan(driver, key, changed)
    assert second == plan_hash(changed)

    with driver.session(database="neo4j") as s:
        s.run(
            "MATCH (j:IngestJob {key:$key}) SET j.completed_tasks=['copy']",
            key=key,
        ).consume()

    with pytest.raises(PipelinePlanDrift):
        bind_plan(driver, key, original)


def test_same_plan_is_idempotent_after_progress(driver):
    key = "ci-plan:same"
    tasks = (Task("copy", ("copy-v1",), 30),)
    ensure_job(driver, key, "ci")
    expected = bind_plan(driver, key, tasks)
    with driver.session(database="neo4j") as s:
        s.run(
            "MATCH (j:IngestJob {key:$key}) SET j.completed_tasks=['copy']",
            key=key,
        ).consume()
    assert bind_plan(driver, key, tasks) == expected
