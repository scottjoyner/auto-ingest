#!/usr/bin/env python3
"""Backfill ``PhoneLog.loc`` spatial points safely and idempotently.

The migration is intentionally bounded and resumable. It only mutates nodes
with valid source coordinates and can be re-run after interruption.
"""
from __future__ import annotations

import argparse
import logging
import sys

from auto_ingest_config import get_neo4j_env

NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DB = get_neo4j_env()

DEFAULT_BATCH_SIZE = 5_000
MAX_BATCH_SIZE = 100_000

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


COORDINATE_PROJECTION = """
    WITH pl,
        coalesce(
            pl.latitude,
            CASE
                WHEN pl.geometry IS NOT NULL
                 AND pl.geometry.coordinates IS NOT NULL
                THEN pl.geometry.coordinates[1]
                ELSE NULL
            END
        ) AS lat,
        coalesce(
            pl.longitude,
            CASE
                WHEN pl.geometry IS NOT NULL
                 AND pl.geometry.coordinates IS NOT NULL
                THEN pl.geometry.coordinates[0]
                ELSE NULL
            END
        ) AS lon
    WHERE lat IS NOT NULL
      AND lon IS NOT NULL
      AND lat >= -90 AND lat <= 90
      AND lon >= -180 AND lon <= 180
"""


def validate_batch_size(batch_size: int) -> int:
    """Reject unbounded or accidental giant transactions before connecting."""
    if batch_size < 1:
        raise ValueError("batch size must be at least 1")
    if batch_size > MAX_BATCH_SIZE:
        raise ValueError(
            f"batch size {batch_size:,} exceeds safety cap {MAX_BATCH_SIZE:,}"
        )
    return batch_size


def count_missing_loc(driver) -> int:
    """Count all PhoneLog nodes still missing ``loc``."""
    with driver.session(database=NEO4J_DB) as session:
        return session.run(
            """
            MATCH (pl:PhoneLog)
            WHERE pl.loc IS NULL
            RETURN count(pl) AS missing
            """
        ).single()["missing"]


def count_eligible(driver) -> int:
    """Count missing nodes that contain valid coordinates and can be migrated."""
    with driver.session(database=NEO4J_DB) as session:
        result = session.run(
            """
            MATCH (pl:PhoneLog)
            WHERE pl.loc IS NULL
              AND (pl.latitude IS NOT NULL OR pl.geometry IS NOT NULL)
            """
            + COORDINATE_PROJECTION
            + "RETURN count(pl) AS eligible"
        )
        return result.single()["eligible"]


def migrate_batch(driver, batch_size: int, dry_run: bool = False) -> int:
    """Migrate one bounded batch and return the number selected/written."""
    batch_size = validate_batch_size(batch_size)

    prefix = """
        MATCH (pl:PhoneLog)
        WHERE pl.loc IS NULL
          AND (pl.latitude IS NOT NULL OR pl.geometry IS NOT NULL)
        WITH pl LIMIT $batch_size
    """

    with driver.session(database=NEO4J_DB) as session:
        if dry_run:
            result = session.run(
                prefix
                + COORDINATE_PROJECTION
                + "RETURN count(pl) AS would_migrate",
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
    parser = argparse.ArgumentParser(
        description="Migrate PhoneLog nodes to add spatial loc property"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Report eligible writes without mutating"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Nodes per transaction (default: {DEFAULT_BATCH_SIZE}; max: {MAX_BATCH_SIZE})",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Maximum batches to process (0 = until no eligible nodes remain)",
    )
    args = parser.parse_args()

    try:
        validate_batch_size(args.batch_size)
    except ValueError as exc:
        parser.error(str(exc))

    if args.max_batches < 0:
        parser.error("--max-batches cannot be negative")

    logger.info("Connecting to %s/%s", NEO4J_URI, NEO4J_DB)
    driver = None

    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
            connection_timeout=15,
        )
        driver.verify_connectivity()

        missing_before = count_missing_loc(driver)
        eligible_before = count_eligible(driver)
        logger.info("PhoneLog missing loc: %,d", missing_before)
        logger.info("PhoneLog eligible for migration: %,d", eligible_before)

        if eligible_before == 0:
            if missing_before:
                logger.warning(
                    "No valid coordinates remain; %,d PhoneLog nodes are still missing loc",
                    missing_before,
                )
            else:
                logger.info("All PhoneLog nodes already have loc")
            return 0

        if args.dry_run:
            first_batch = migrate_batch(driver, args.batch_size, dry_run=True)
            batches = (eligible_before + args.batch_size - 1) // args.batch_size
            logger.info("DRY RUN: first batch would migrate %,d", first_batch)
            logger.info("DRY RUN: approximately %,d batches required", batches)
            return 0

        total_migrated = 0
        batch_num = 0
        while True:
            if args.max_batches and batch_num >= args.max_batches:
                logger.info("Reached max-batches=%d", args.max_batches)
                break

            migrated = migrate_batch(driver, args.batch_size)
            if migrated == 0:
                break

            batch_num += 1
            total_migrated += migrated
            logger.info(
                "Batch %d migrated %,d nodes (total %,d)",
                batch_num,
                migrated,
                total_migrated,
            )

        missing_after = count_missing_loc(driver)
        eligible_after = count_eligible(driver)
        logger.info("Migration complete: migrated %,d", total_migrated)
        logger.info("Remaining missing loc: %,d", missing_after)
        logger.info("Remaining eligible: %,d", eligible_after)

        if eligible_after:
            logger.warning(
                "Migration stopped with %,d eligible nodes remaining; rerun to resume",
                eligible_after,
            )
        return 0

    except Exception:
        logger.exception("PhoneLog spatial migration failed")
        return 1
    finally:
        if driver is not None:
            driver.close()


if __name__ == "__main__":
    sys.exit(main())
