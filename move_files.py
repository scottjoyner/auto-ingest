#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Organize files named `YYYY_MMDD_HHMMSS_{tail}` into `YYYY/MM/DD/<original_name>`.

Examples:
  python organize_by_timestamp.py -i /audio -o /archive
  python organize_by_timestamp.py -i /audio -o /archive --dry-run --recursive
  python organize_by_timestamp.py -i /audio -o /archive --strategy rename --max-workers 8

File name format assumed at START of basename:
  YYYY_MMDD_HHMMSS_{tail}[.ext]
Where:
  - YYYY: year (4 digits)
  - MMDD: month (01-12) + day (01-31)
  - HHMMSS: 24h time
  - {tail}: anything (may contain underscores, dots, etc.)

Non-matching or invalid-date filenames are skipped (with a warning).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import logging
import os
from pathlib import Path
import re
import shutil
import sys
from datetime import datetime
from typing import Optional, Tuple

NAME_PATTERN = re.compile(
    r'^(?P<year>\d{4})_(?P<md>\d{4})_(?P<hms>\d{6})_(?P<tail>.+)$'
)

def parse_name(basename: str) -> Optional[Tuple[int, int, int]]:
    """
    Parse the leading timestamp from a basename and validate Y-M-D H:M:S.
    Returns (year, month, day) if valid; None otherwise.
    """
    m = NAME_PATTERN.match(basename)
    if not m:
        return None
    year = int(m.group("year"))
    md = m.group("md")
    hms = m.group("hms")
    month = int(md[:2])
    day = int(md[2:])
    hour = int(hms[:2])
    minute = int(hms[2:4])
    second = int(hms[4:6])
    try:
        # Validate actual calendar/time
        datetime(year, month, day, hour, minute, second)
    except ValueError:
        return None
    return year, month, day

def compute_hash(path: Path, algo: str = "md5", chunk_size: int = 1024 * 1024) -> str:
    h = getattr(hashlib, algo)()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def choose_collision_action(
    src: Path,
    dst: Path,
    strategy: str,
    hash_algo: Optional[str],
) -> Optional[Path]:
    """
    Decide final destination for a move considering collisions.
    Returns the finalized destination Path, or None if we should skip.
    """
    if not dst.exists():
        return dst

    if strategy == "skip":
        # If same file (optionally hash), skip; else skip with warning.
        same = False
        try:
            if src.stat().st_size == dst.stat().st_size:
                if hash_algo:
                    same = compute_hash(src, hash_algo) == compute_hash(dst, hash_algo)
                else:
                    same = True  # heuristic: same size -> likely same content
        except OSError:
            pass
        if same:
            logging.info("Skip (duplicate): %s -> %s", src, dst)
        else:
            logging.warning("Skip (exists): %s -> %s", src, dst)
        return None

    if strategy == "overwrite":
        return dst

    if strategy == "fail":
        raise FileExistsError(f"Destination exists: {dst}")

    if strategy == "rename":
        # Append suffix _1, _2, ... before extension
        stem = dst.stem
        suffix = dst.suffix
        parent = dst.parent
        n = 1
        while True:
            candidate = parent / f"{stem}_{n}{suffix}"
            if not candidate.exists():
                return candidate
            n += 1

    raise ValueError(f"Unknown strategy: {strategy}")

def build_destination(root_out: Path, basename: str) -> Optional[Path]:
    parsed = parse_name(basename)
    if not parsed:
        return None
    year, month, day = parsed
    return root_out / f"{year:04d}" / f"{month:02d}" / f"{day:02d}" / basename

def move_one(
    src: Path,
    root_out: Path,
    dry_run: bool,
    strategy: str,
    hash_algo: Optional[str],
) -> str:
    basename = src.name
    dst = build_destination(root_out, basename)
    if dst is None:
        logging.warning("Name does not match pattern, skipping: %s", src)
        return "skipped"

    final_dst = choose_collision_action(src, dst, strategy=strategy, hash_algo=hash_algo)
    if final_dst is None:
        return "skipped"

    if dry_run:
        logging.info("[DRY-RUN] Would move: %s -> %s", src, final_dst)
        return "dry"

    final_dst.parent.mkdir(parents=True, exist_ok=True)

    # Prefer atomic rename when possible; shutil.move handles cross-device moves
    try:
        shutil.move(str(src), str(final_dst))
        logging.info("Moved: %s -> %s", src, final_dst)
        return "moved"
    except Exception as e:
        logging.error("Failed moving %s -> %s: %s", src, final_dst, e)
        return "error"

def iter_files(root_in: Path, recursive: bool, include_hidden: bool, follow_symlinks: bool):
    if recursive:
        it = root_in.rglob("*")
    else:
        it = root_in.iterdir()

    for p in it:
        try:
            if p.is_dir():
                continue
            if p.is_symlink() and not follow_symlinks:
                continue
            if not include_hidden and p.name.startswith("."):
                continue
            yield p
        except OSError:
            # Permissions races, broken symlinks, etc.
            continue

def main():
    ap = argparse.ArgumentParser(
        description="Move files named 'YYYY_MMDD_HHMMSS_{tail}' into YYYY/MM/DD/ subfolders."
    )
    ap.add_argument("-i", "--input", required=True, type=Path, help="Input directory to scan")
    ap.add_argument("-o", "--output", required=True, type=Path, help="Output directory root")
    ap.add_argument("-r", "--recursive", action="store_true", help="Recurse into subdirectories")
    ap.add_argument("--follow-symlinks", action="store_true", help="Follow symlinked files")
    ap.add_argument("--include-hidden", action="store_true", help="Include dotfiles")
    ap.add_argument("--dry-run", "-n", action="store_true", help="Show actions without moving")
    ap.add_argument(
        "--strategy",
        choices=["skip", "rename", "overwrite", "fail"],
        default="skip",
        help="What to do if destination exists (default: skip)",
    )
    ap.add_argument(
        "--hash",
        choices=["md5", "sha1", "sha256"],
        default=None,
        help="If set, use content hash for duplicate detection when strategy=skip",
    )
    ap.add_argument(
        "--max-workers",
        type=int,
        default=min(32, (os.cpu_count() or 4) * 2),
        help="Parallel worker count (default: CPUs*2 up to 32)",
    )
    ap.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )

    args = ap.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not args.input.exists() or not args.input.is_dir():
        logging.error("Input is not a directory: %s", args.input)
        sys.exit(2)

    args.output.mkdir(parents=True, exist_ok=True)

    files = list(iter_files(args.input, args.recursive, args.include_hidden, args.follow_symlinks))
    if not files:
        logging.warning("No files found in input (check flags).")
        return

    logging.info("Discovered %d file(s) to consider.", len(files))

    counts = {"moved": 0, "skipped": 0, "dry": 0, "error": 0}
    work = [(p, args.output, args.dry_run, args.strategy, args.hash) for p in files]

    # Threaded I/O: safe for shutil + filesystem ops
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = [ex.submit(move_one, *w) for w in work]
        for fut in concurrent.futures.as_completed(futures):
            result = fut.result()
            counts[result] = counts.get(result, 0) + 1

    logging.info(
        "Done. moved=%d skipped=%d dry=%d errors=%d",
        counts["moved"],
        counts["skipped"],
        counts["dry"],
        counts["error"],
    )

if __name__ == "__main__":
    main()
