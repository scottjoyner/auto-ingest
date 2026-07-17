#!/usr/bin/env python3
"""
PhoneLog spatial point migration script.

Backfills the `loc` POINT property on PhoneLog nodes that are missing it,
using existing latitude/longitude or geometry.coordinates fields.

Usage:
    python scripts/migrate_phonelog_spatial.py [--dry-run] [--batch-size N]

This script:
1. Finds PhoneLog nodes without `loc` property
2. Extracts lat/lon from available sources (priority: loc -> latitude/longitude -> geometry.coordinates)
3. Creates proper Neo4j POINT with CRS 'wgs-84'
4. Runs in batches to avoid OOM on 21M+ node graph

Safety:
- Dry-run mode shows what would be done without writing
- Batch processing prevents transaction timeouts
- Idempotent (safe to re-run)
"""
import os
import sys
import argparse
import logging
from pathlib import Path

try:
    from auto_ingest_config import get_neo4j_password
    NEO4J_PASSWORD = get_neo4j_password()
except Exception:
    NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD") or os.environ.get("NEO4J_PASSWORD_DEFAULT") or "knowledge_graph_2026"

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_DB = os.environ.get("NEO4J_DB", "neo4j")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def count_missing_loc(driver):
    """Count how many PhoneLog nodes are missing the loc property."""
    with driver.session(database=NEO4J_DB) as session:
        result = session.run("""
            MATCH (pl:PhoneLog)
            WHERE pl.loc IS NULL
            RETURN count(pl) AS missing
        """)
        return result.single()["missing"]


def migrate_batch(driver, batch_size: int, dry_run: bool = False):
    """
    Migrate one batch of PhoneLog nodes to add loc spatial point.
    
    Returns number of nodes migrated in this batch.
    """
    with driver.session(database=NEO4J_DB) as session:
        # Find nodes missing loc but having lat/lon data
        query = """
            MATCH (pl:PhoneLog)
            WHERE pl.loc IS NULL
              AND (pl.latitude IS NOT NULL OR pl.geometry IS NOT NULL)
            WITH pl LIMIT $batch_size
            // Extract coordinates from available sources
            WITH pl,
                coalesce(pl.latitude, 
                    CASE WHEN pl.geometry IS NOT NULL AND pl.geometry.coordinates IS NOT NULL 
                         THEN pl.geometry.coordinates[1] 
                         ELSE NULL END
                ) AS lat,
                coalesce(pl.longitude,
                    CASE WHEN pl.geometry IS NOT NULL AND pl.geometry.coordinates IS NOT NULL
                         THEN pl.geometry.coordinates[0]
                         ELSE NULL END
                ) AS lon
            WHERE lat IS NOT NULL AND lon IS NOT NULL
            // Create spatial point
            SET pl.loc = point({latitude: lat, longitude: lon, crs: 'wgs-84'})
            RETURN count(pl) AS migrated
        """
        
        if dry_run:
            # Count what would be migrated without actually doing it
            count_query = """
                MATCH (pl:PhoneLog)
                WHERE pl.loc IS NULL
                  AND (pl.latitude IS NOT NULL OR pl.geometry IS NOT NULL)
                WITH pl LIMIT $batch_size
                WITH pl,
                    coalesce(pl.latitude, 
                        CASE WHEN pl.geometry IS NOT NULL AND pl.geometry.coordinates IS NOT NULL 
                             THEN pl.geometry.coordinates[1] 
                             ELSE NULL END
                    ) AS lat,
                    coalesce(pl.longitude,
                        CASE WHEN pl.geometry IS NOT NULL AND pl.geometry.coordinates IS NOT NULL
                             THEN pl.geometry.coordinates[0]
                             ELSE NULL END
                    ) AS lon
                WHERE lat IS NOT NULL AND lon IS NOT NULL
                RETURN count(pl) AS would_migrate
            """
            result = session.run(count_query, batch_size=batch_size)
            return result.single()["would_migrate"]
        else:
            result = session.run(query, batch_size=batch_size)
            return result.single()["migrated"]


def main():
    parser = argparse.ArgumentParser(description="Migrate PhoneLog nodes to add spatial loc property")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without writing")
    parser.add_argument("--batch-size", type=int, default=5000, help="Nodes per batch (default: 5000)")
    parser.add_argument("--max-batches", type=int, default=0, help="Maximum batches to process (0 = unlimited)")
    args = parser.parse_args()
    
    logger.info(f"Connecting to {NEO4J_URI}/{NEO4J_DB}...")
    driver = None
    
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        
        # Verify connection
        with driver.session() as session:
            session.run("RETURN 1")
        logger.info("Connected successfully")
        
        # Check current state
        missing_before = count_missing_loc(driver)
        logger.info(f"PhoneLog nodes missing `loc`: {missing_before:,}")
        
        if missing_before == 0:
            logger.info("All PhoneLog nodes already have `loc` property. Nothing to do.")
            return 0
        
        if args.dry_run:
            logger.info("DRY RUN MODE - no changes will be made")
            # Estimate how many batches needed
            estimated_batches = (missing_before + args.batch_size - 1) // args.batch_size
            logger.info(f"Estimated batches needed: {estimated_batches}")
            logger.info(f"Batch size: {args.batch_size:,}")
            
            # Show first batch estimate
            migrated = migrate_batch(driver, args.batch_size, dry_run=True)
            logger.info(f"First batch would migrate: {migrated:,} nodes")
            logger.info("Dry run complete. Run without --dry-run to apply changes.")
            return 0
        
        # Actual migration
        logger.info(f"Migrating in batches of {args.batch_size:,}...")
        total_migrated = 0
        batch_num = 0
        
        while True:
            batch_num += 1
            migrated = migrate_batch(driver, args.batch_size, dry_run=False)
            
            if migrated == 0:
                logger.info(f"Batch {batch_num}: No more nodes to migrate")
                break
            
            total_migrated += migrated
            logger.info(f"Batch {batch_num}: Migrated {migrated:,} nodes (total: {total_migrated:,})")
            
            if args.max_batches > 0 and batch_num >= args.max_batches:
                logger.info(f"Reached max batches limit ({args.max_batches}). Stopping.")
                break
        
        # Verify results
        missing_after = count_missing_loc(driver)
        logger.info(f"\nMigration complete!")
        logger.info(f"  Before: {missing_before:,} nodes missing `loc`")
        logger.info(f"  After:  {missing_after:,} nodes missing `loc`")
        logger.info(f"  Migrated: {total_migrated:,} nodes")
        
        if missing_after > 0:
            logger.warning(f"  Still {missing_after:,} nodes without loc (may lack lat/lon data)")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during migration: {e}", exc_info=True)
        return 1
    finally:
        if driver:
            driver.close()


if __name__ == "__main__":
    sys.exit(main())
