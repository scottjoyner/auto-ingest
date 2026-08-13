"""Real Neo4j contract tests for the PhoneLog spatial migration."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest
from neo4j import GraphDatabase

pytestmark = [pytest.mark.integration, pytest.mark.destructive]

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "migrate_phonelog_spatial.py"


def _load_migration_module():
    spec = importlib.util.spec_from_file_location("migrate_phonelog_spatial", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def driver():
    uri = os.environ.get("NEO4J_TEST_URI")
    user = os.environ.get("NEO4J_TEST_USER", "neo4j")
    password = os.environ.get("NEO4J_TEST_PASSWORD")
    if not uri or not password:
        pytest.skip("NEO4J_TEST_URI/NEO4J_TEST_PASSWORD are required")
    drv = GraphDatabase.driver(uri, auth=(user, password))
    drv.verify_connectivity()
    yield drv
    drv.close()


@pytest.fixture(autouse=True)
def fixture_graph(driver):
    with driver.session(database="neo4j") as session:
        session.run("MATCH (n:HardeningFixture) DETACH DELETE n").consume()
        session.run(
            """
            CREATE (:PhoneLog:HardeningFixture {
                fixture_id: 'flat', latitude: 35.2271, longitude: -80.8431
            })
            CREATE (:PhoneLog:HardeningFixture {
                fixture_id: 'coordinates', coordinates: [-80.8500, 35.2300]
            })
            CREATE (:PhoneLog:HardeningFixture {
                fixture_id: 'legacy-geometry-string',
                geometry: "{'type': 'Point', 'coordinates': [-80.86, 35.24]}"
            })
            CREATE (:PhoneLog:HardeningFixture {fixture_id: 'no-coordinates'})
            """
        ).consume()
    yield
    with driver.session(database="neo4j") as session:
        session.run("MATCH (n:HardeningFixture) DETACH DELETE n").consume()


def test_phonelog_spatial_migration_is_correct_and_idempotent(driver):
    migration = _load_migration_module()
    migration.NEO4J_DB = "neo4j"

    assert migration.count_missing_loc(driver) >= 4
    assert migration.count_eligible(driver) == 2
    assert migration.migrate_batch(driver, batch_size=100, dry_run=False) == 2

    with driver.session(database="neo4j") as session:
        rows = session.run(
            """
            MATCH (pl:PhoneLog:HardeningFixture)
            RETURN pl.fixture_id AS id,
                   pl.loc.latitude AS latitude,
                   pl.loc.longitude AS longitude,
                   pl.loc IS NOT NULL AS has_loc
            ORDER BY id
            """
        ).data()

    by_id = {row["id"]: row for row in rows}
    assert by_id["flat"]["has_loc"] is True
    assert by_id["flat"]["latitude"] == pytest.approx(35.2271)
    assert by_id["flat"]["longitude"] == pytest.approx(-80.8431)
    assert by_id["coordinates"]["has_loc"] is True
    assert by_id["coordinates"]["latitude"] == pytest.approx(35.2300)
    assert by_id["coordinates"]["longitude"] == pytest.approx(-80.8500)
    assert by_id["legacy-geometry-string"]["has_loc"] is False
    assert by_id["no-coordinates"]["has_loc"] is False

    assert migration.migrate_batch(driver, batch_size=100, dry_run=False) == 0
    assert migration.count_eligible(driver) == 0


def test_phonelog_spatial_dry_run_does_not_write(driver):
    migration = _load_migration_module()
    migration.NEO4J_DB = "neo4j"

    assert migration.migrate_batch(driver, batch_size=100, dry_run=True) == 2
    with driver.session(database="neo4j") as session:
        changed = session.run(
            "MATCH (pl:PhoneLog:HardeningFixture) WHERE pl.loc IS NOT NULL RETURN count(pl) AS n"
        ).single()["n"]
    assert changed == 0


def test_batch_safety_cap_is_enforced():
    migration = _load_migration_module()
    with pytest.raises(ValueError):
        migration.validate_batch_size(0)
    with pytest.raises(ValueError):
        migration.validate_batch_size(migration.MAX_BATCH_SIZE + 1)
