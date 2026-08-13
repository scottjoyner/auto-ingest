from __future__ import annotations

import os

import pytest
from neo4j import GraphDatabase

from auto_ingest.runtime_schema import SchemaContractError, audit_schema, ensure_schema

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
        s.run("DROP CONSTRAINT ingestjob_key_unique IF EXISTS").consume()
        s.run("DROP CONSTRAINT ingestartifact_id_unique IF EXISTS").consume()
        s.run("MATCH (n) WHERE n.fixture_scope='runtime-schema-ci' DETACH DELETE n").consume()
    yield
    with driver.session(database="neo4j") as s:
        s.run("DROP CONSTRAINT ingestjob_key_unique IF EXISTS").consume()
        s.run("DROP CONSTRAINT ingestartifact_id_unique IF EXISTS").consume()
        s.run("MATCH (n) WHERE n.fixture_scope='runtime-schema-ci' DETACH DELETE n").consume()


def test_schema_registry_creates_and_audits_unique_constraints(driver):
    report = ensure_schema(driver)
    assert report["ok"] is True
    assert report["missing_constraints"] == []
    assert report["duplicates"]["ingestjob_key_unique"] == []
    assert report["duplicates"]["ingestartifact_id_unique"] == []

    with driver.session(database="neo4j") as s:
        s.run(
            "CREATE (:IngestJob {key:'ci-unique', fixture_scope:'runtime-schema-ci'})"
        ).consume()
        with pytest.raises(Exception):
            s.run(
                "CREATE (:IngestJob {key:'ci-unique', fixture_scope:'runtime-schema-ci'})"
            ).consume()


def test_schema_registry_refuses_dirty_identity_data(driver):
    with driver.session(database="neo4j") as s:
        s.run(
            """
            CREATE (:IngestArtifact {artifact_id:'dup', fixture_scope:'runtime-schema-ci'}),
                   (:IngestArtifact {artifact_id:'dup', fixture_scope:'runtime-schema-ci'})
            """
        ).consume()

    with pytest.raises(SchemaContractError):
        ensure_schema(driver)
    report = audit_schema(driver)
    assert report["ok"] is False
    assert report["duplicates"]["ingestartifact_id_unique"][0]["value"] == "dup"
    assert report["duplicates"]["ingestartifact_id_unique"][0]["count"] == 2
