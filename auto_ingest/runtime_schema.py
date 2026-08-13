"""Runtime Neo4j schema contract for distributed ingest correctness.

The registry makes identity assumptions executable instead of implicit. Unique
constraints are only created after duplicate preflight checks pass; production
startup can audit without mutating schema.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UniqueContract:
    name: str
    label: str
    property: str


UNIQUE_CONTRACTS = (
    UniqueContract("ingestjob_key_unique", "IngestJob", "key"),
    UniqueContract("ingestartifact_id_unique", "IngestArtifact", "artifact_id"),
)


class SchemaContractError(RuntimeError):
    pass


def find_duplicates(driver, contract: UniqueContract, *, limit: int = 20) -> list[dict]:
    if limit < 1:
        raise ValueError("limit must be positive")
    query = f"""
    MATCH (n:`{contract.label}`)
    WHERE n.`{contract.property}` IS NOT NULL
    WITH n.`{contract.property}` AS value, count(*) AS count
    WHERE count > 1
    RETURN value, count
    ORDER BY count DESC
    LIMIT $limit
    """
    with driver.session() as session:
        return session.run(query, limit=limit).data()


def audit_schema(driver) -> dict:
    duplicates = {
        c.name: find_duplicates(driver, c)
        for c in UNIQUE_CONTRACTS
    }
    with driver.session() as session:
        rows = session.run(
            "SHOW CONSTRAINTS YIELD name, type, labelsOrTypes, properties "
            "RETURN name, type, labelsOrTypes, properties"
        ).data()
    present = {row["name"] for row in rows}
    return {
        "ok": all(not rows for rows in duplicates.values())
        and all(c.name in present for c in UNIQUE_CONTRACTS),
        "duplicates": duplicates,
        "missing_constraints": [c.name for c in UNIQUE_CONTRACTS if c.name not in present],
    }


def ensure_schema(driver) -> dict:
    """Create required uniqueness constraints only when data is clean."""
    dirty = {}
    for contract in UNIQUE_CONTRACTS:
        dupes = find_duplicates(driver, contract)
        if dupes:
            dirty[contract.name] = dupes
    if dirty:
        raise SchemaContractError(
            "refusing to create uniqueness constraints while duplicate identities exist: "
            + repr(dirty)
        )

    with driver.session() as session:
        for contract in UNIQUE_CONTRACTS:
            query = (
                f"CREATE CONSTRAINT `{contract.name}` IF NOT EXISTS "
                f"FOR (n:`{contract.label}`) REQUIRE n.`{contract.property}` IS UNIQUE"
            )
            session.run(query).consume()
    return audit_schema(driver)
