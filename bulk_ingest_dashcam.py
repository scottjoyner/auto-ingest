#!/usr/bin/env python3
"""
Bulk Dashcam Ingest Script
Processes all dashcam data from fileserver into Neo4j with progress tracking.
"""

import os
import sys
import subprocess
import logging
import argparse
from datetime import datetime
from pathlib import Path

# Configure logging
LOG_DIR = Path("./logs")
LOG_DIR.mkdir(exist_ok=True)
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOG_DIR / f"bulk_ingest_{STAMP}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE)
    ]
)
logger = logging.getLogger(__name__)

# Configuration
DASHCAM_ROOT = os.environ.get("DASHCAM_ROOT", "/media/scott/SSD_4TB/fileserver/dashcam")
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://host.docker.internal:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASS = os.environ.get("NEO4J_PASS", "knowledge_graph_2026")

# Years to process (can be overridden via env var)
YEARS_TO_PROCESS = os.environ.get("YEARS_TO_PROCESS", "2023 2024 2025 2026").split()


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Bulk ingest dashcam data into Neo4j")
    parser.add_argument("--years", nargs="*", help="Years to process (default: all)")
    parser.add_argument("--year-months", nargs="*", help="Specific year/month pairs (e.g., 2026/05 2026/06)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be processed without running")
    parser.add_argument("--limit", type=int, default=0, help="Limit to N days (for testing)")
    return parser.parse_args()


def get_day_path(year: str, month: str, day: str) -> Path:
    """Get the path to a specific day's dashcam data."""
    return Path(DASHCAM_ROOT) / year / month / day


def count_clips(day_path: Path) -> int:
    """Count the number of clip metadata files in a day directory."""
    try:
        # Look for _metadata.csv files (GPS/location data)
        meta_files = list(day_path.glob("*_metadata.csv"))
        if meta_files:
            return len(meta_files)
        
        # Fallback: count unique clip base names from video/YOLO files
        # This handles cases where only YOLO CSVs exist without metadata
        yolo_files = list(day_path.glob("*_YOLO*.csv"))
        if yolo_files:
            # Extract unique clip keys (remove _F/_R suffix and _YOLO* part)
            clip_keys = set()
            for f in yolo_files:
                name = f.stem  # e.g., "2026_0501_060041_F_YOLOv8n"
                # Remove view suffix (_F, _R) and YOLO part
                key = ''.join(name.split('_YOLO'))
                key = key.rsplit('_', 1)[0] if '_' in key else key
                clip_keys.add(key)
            return len(clip_keys)
        
        return 0
    except PermissionError:
        logger.warning(f"Permission denied accessing {day_path}")
        return 0


def process_day(year: str, month: str, day: str, dry_run: bool = False) -> bool:
    """Process a single day of dashcam data."""
    day_path = get_day_path(year, month, day)
    
    if not day_path.exists():
        logger.info(f"[SKIP] Directory not found: {day_path}")
        return True
    
    clip_count = count_clips(day_path)
    if clip_count == 0:
        logger.info(f"[SKIP] No clips found in: {day_path}")
        return True
    
    logger.info(f"[RUN] Processing {year}/{month}/{day} ({clip_count} clips)")
    
    if dry_run:
        logger.info(f"  [DRY-RUN] Would run: docker compose run ... --bases /dashcam/{year}/{month}/{day}")
        return True
    
    # Build the docker command
    cmd = [
        "docker", "compose", "run", "--rm", "--no-deps",
        "-v", f"{DASHCAM_ROOT}:/dashcam",
        "ingest-service",
        "bash", "-c",
        f"python /app/dashcam_yolo_embeddings.py "
        f"--bases /dashcam/{year}/{month}/{day} "
        f"--neo4j-uri {NEO4J_URI} "
        f"--neo4j-user {NEO4J_USER} "
        f"--neo4j-pass {NEO4J_PASS} "
        f"--resume"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode != 0:
            logger.error(f"[ERROR] Failed to process {year}/{month}/{day} (exit code: {result.returncode})")
            return False
        
        logger.info(f"[DONE] Completed {year}/{month}/{day}")
        return True
    except Exception as e:
        logger.error(f"[EXCEPTION] Error processing {year}/{month}/{day}: {e}")
        return False


def main():
    """Main bulk ingest loop."""
    args = parse_args()
    
    # Override years if specified
    years_to_process = args.years if args.years else YEARS_TO_PROCESS
    year_months = args.year_months if args.year_months else []
    
    logger.info("="*60)
    logger.info("BULK DASHCAM INGEST STARTING")
    logger.info(f"Years to process: {years_to_process}")
    logger.info(f"Specific year/months: {year_months}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info(f"Limit: {args.limit} days")
    logger.info(f"Dashcam root: {DASHCAM_ROOT}")
    logger.info(f"Log file: {LOG_FILE}")
    logger.info("="*60)
    
    total_days = 0
    processed_days = 0
    failed_days = 0
    skipped_days = 0
    days_processed = 0
    
    # Handle specific year/month pairs if provided
    if year_months:
        logger.info("Processing specific year/month pairs")
        for ym in year_months:
            parts = ym.split('/')
            if len(parts) != 2:
                logger.warning(f"Invalid year/month format: {ym}")
                continue
            year, month = parts
            day_path = Path(DASHCAM_ROOT) / year / month
            if not day_path.exists():
                logger.info(f"[SKIP] Year/month not found: {day_path}")
                continue
            
            days = sorted([d.name for d in day_path.iterdir() if d.is_dir()])
            for day in days:
                total_days += 1
                if args.limit and days_processed >= args.limit:
                    break
                
                success = process_day(year, month, day, args.dry_run)
                if success:
                    processed_days += 1
                else:
                    failed_days += 1
                days_processed += 1
            
            if args.limit and days_processed >= args.limit:
                break
        return failed_days
    
    # Standard year-by-year processing
    for year in years_to_process:
        year_path = Path(DASHCAM_ROOT) / year
        if not year_path.exists():
            logger.info(f"[SKIP] Year directory not found: {year_path}")
            continue
        
        months = sorted([d.name for d in year_path.iterdir() if d.is_dir()])
        logger.info(f"Processing year {year}: {len(months)} months")
        
        for month in months:
            month_path = year_path / month
            days = sorted([d.name for d in month_path.iterdir() if d.is_dir()])
            logger.info(f"  Processing {year}/{month}: {len(days)} days")
            
            for day in days:
                total_days += 1
                if args.limit and days_processed >= args.limit:
                    break
                
                success = process_day(year, month, day, args.dry_run)
                if success:
                    processed_days += 1
                else:
                    failed_days += 1
                days_processed += 1
            
            if args.limit and days_processed >= args.limit:
                break
        
        if args.limit and days_processed >= args.limit:
            break
    
    # Summary
    logger.info("")
    logger.info("="*60)
    logger.info("BULK INGEST SUMMARY")
    logger.info(f"Total days found: {total_days}")
    logger.info(f"Successfully processed: {processed_days}")
    logger.info(f"Failed: {failed_days}")
    logger.info(f"Skipped (empty): {skipped_days}")
    logger.info("="*60)
    
    return failed_days


if __name__ == "__main__":
    sys.exit(main())
