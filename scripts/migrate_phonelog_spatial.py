#!/usr/bin/env python3
"""Backfill ``PhoneLog.loc`` spatial points safely and idempotently.

The migration is intentionally bounded and resumable. It only mutates nodes
with valid normalized coordinates and can be re-run after interruption.
"""
from __future__ import annotations

import argparse
import logging
import sys

from auto_ingest.ops.migration_safety import (
    SafetyViolation,
    preflight_summary,
    validate_batch_size as validate_bounded_batch,
)
from auto_ingest_config import get_neo4j_env

NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DB = get_neo4j_env()

DEFAULT_BATCH_SIZE = 5_000
MAX_BATCH_SIZE = 100_000

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# PhoneLog.geometry is historically a string representation, not a Neo4j map.
# PhoneLog normalization materializes either flat latitude/longitude or the
# primitive coordinates array [longitude, latitude].
COORDINATE_PROJECTION = """
    WITH pl,
        coalesce(pl.latitude, pl.coordinates[1]) AS lat,
        coalesce(pl.longitude, pl.coordinates[0]) AS lon
    WHERE lat IS NOT NULL
      AND lon IS NOT NULL
      AND lat >= -90 AND lat <= 90
      AND lon >= -180 AND lon <= 180
"""


def validate_batch_size(batch_size: int) -> int:
    """Compatibility wrapper around the shared production safety contract."""
    return validate_bounded_batch(batch_size, max_batch_size=MAX_BATCH_SIZE)


def count_missing_loc(driver) -> int:
    with driver.session(database=NEO4J_DB) as session:
        return session.run(
            "MATCH (pl:PhoneLog) WHERE pl.loc IS NULL RETURN count(pl) AS missing"
        ).single()["missing"]


def count_eligible(driver) -> int:
    with driver.session(database=NEO4J_DB) as session:
        result = session.run(
            """
            MATCH (pl:PhoneLog)
            WHERE pl.loc IS NULL
              AND (pl.latitude IS NOT NULL OR pl.coordinates IS NOT NULL)
            """
            + COORDINATE_PROJECTION
            + "RETURN count(pl) AS eligible"
        )
        return result.single()["eligible"]


def migrate_batch(driver, batch_size: int, dry_run: bool = False) -> int:
    batch_size = validate_batch_size(batch_size)
    prefix = """
        MATCH (pl:PhoneLog)
        WHERE pl.loc IS NULL
          AND (pl.latitude IS NOT NULL OR pl.coordinates IS NOT NULL)
        WITH pl LIMIT $batch_size
    """
    with driver.session(database=NEO4J_DB) as session:
        if dry_run:
            result = session.run(
                prefix + COORDINATE_PROJECTION + "RETURN count(pl) AS would_migrate",
                batch_size=batch_size,
            )
            return result.single()["would_migrate"]
        result = session.run(
            prefix
            + COORDINATE_PROJECTION
            + """
              SET pl.loc = point({latitude: lat, longitude: lon, crs: 'wgs-84'})
              RETURN count(pl) AS migrated
            """,
            batch_size=batch_size,
        )
        return result.single()["migrated"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate PhoneLog spatial loc property")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-batches", type=int, default=0)
    args = parser.parse_args()
    try:
        validate_batch_size(args.batch_size)
    except SafetyViolation as exc:
        parser.error(str(exc))
    if args.max_batches < 0:
        parser.error("--max-batches cannot be negative")

    driver = None
    try:
        from neo4j import GraphDatabase

        logger.info("Connecting to %s/%s", NEO4J_URI, NEO4J_DB)
        driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
            connection_timeout=15,
        )
        driver.verify_connectivity()
        missing_before = count_missing_loc(driver)
        eligible_before = count_eligible(driver)
        plan = preflight_summary(
            operation="phonelog_spatial",
            total_candidates=missing_before,
            eligible_candidates=eligible_before,
            batch_size=args.batch_size,
            max_batch_size=MAX_BATCH_SIZE,
            dry_run=args.dry_run,
        )
        logger.info("Preflight: %s", plan)

        if args.dry_run:
            first = migrate_batch(driver, args.batch_size, dry_run=True)
            logger.info("DRY RUN first batch would migrate %,d", first)
            return 0

        total = 0
        batches = 0
        remaining = eligible_before
        while remaining:
            if args.max_batches and batches >= args.max_batches:
                logger.warning(
                    "Stopped at explicit max-batches=%d with %,d eligible nodes remaining",
                    args.max_batches,
                    remaining,
                )
                break
            migrated = migrate_batch(driver, args.batch_size)
            if not migrated:
                break
            total += migrated
            batches += 1
            remaining = count_eligible(driver)
            logger.info(
                "Batch %d migrated %,d (total %,d; remaining %,d)",
                batches,
                migrated,
                total,
                remaining,
            )

        logger.info("Migration complete; migrated %,d", total)
        logger.info("Remaining eligible: %,d", count_eligible(driver))
        logger.info("Remaining missing loc: %,d", count_missing_loc(driver))
        return 0
    except Exception:
        logger.exception("PhoneLog spatial migration failed")
        return 1
    finally:
        if driver is not None:
            driver.close()


if __name__ == "__main__":
    sys.exit(main())
