#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Organize media files into `YYYY/MM/DD/<original_name>` folders.

The timestamp parser is intentionally permissive so files from different
recorders/cameras can be mixed in one import pass. Supported filename examples:

  2025_0911_223045_F.MP4
  20250911223045.WAV
  R2025_0911_223045.mp3
  REC_2025-09-11_22-30-45.m4a

By default only common audio/video files are moved. Files with no recognizable
filename timestamp can optionally be placed by filesystem modified time.
"""
# python move_files.py -i /mnt/8TB_2025/fileserver/audio -o /mnt/8TB_2025/fileserver/audio --dry-run --recursive
from __future__ import annotations

import argparse
import concurrent.futures
from dataclasses import dataclass
import hashlib
import logging
import os
from pathlib import Path
import re
import shutil
import sys
from datetime import datetime
from typing import Iterable, Optional


DEFAULT_MEDIA_EXTENSIONS = {
    ".3gp", ".aac", ".aiff", ".amr", ".avi", ".flac", ".m4a", ".m4v",
    ".mkv", ".mov", ".mp3", ".mp4", ".mpeg", ".mpg", ".oga", ".ogg",
    ".opus", ".wav", ".webm", ".wma",
}

# Look for timestamps anywhere in the filename stem. Several voice recorders
# prefix the timestamp with a channel/source letter (for example R2025_...).
TIMESTAMP_PATTERNS = (
    # YYYY_MMDD_HHMMSS, RYYYY_MMDD_HHMMSS, YYYYMMDDHHMMSS
    re.compile(
        r"(?<!\d)(?:[A-Za-z]{1,8})?"
        r"(?P<year>(?:19|20)\d{2})[_-]?"
        r"(?P<month>\d{2})(?P<day>\d{2})[_-]?"
        r"(?P<hour>\d{2})(?P<minute>\d{2})(?P<second>\d{2})(?!\d)"
    ),
    # YYYY-MM-DD_HH-MM-SS, YYYY_MM_DD HH.MM.SS, etc.
    re.compile(
        r"(?<!\d)(?:[A-Za-z]{1,8}[_-]?)?"
        r"(?P<year>(?:19|20)\d{2})[_-]"
        r"(?P<month>\d{2})[_-]"
        r"(?P<day>\d{2})[ T_-]+"
        r"(?P<hour>\d{2})[:._-]?"
        r"(?P<minute>\d{2})[:._-]?"
        r"(?P<second>\d{2})(?!\d)"
    ),
    # YYYY-MM-DD or YYYY_MM_DD when the recorder omits time from the name.
    re.compile(
        r"(?<!\d)(?:[A-Za-z]{1,8}[_-]?)?"
        r"(?P<year>(?:19|20)\d{2})[_-]"
        r"(?P<month>\d{2})[_-]"
        r"(?P<day>\d{2})(?!\d)"
    ),
)

@dataclass(frozen=True)
class ParsedName:
    year: int
    month: int
    day: int
    hour: int
    minute: int
    second: int
    ts_compact: str        # e.g. 20250911223045
    ts_underscored: str    # e.g. 2025_0911_223045
    source: str            # filename or mtime


def _parsed_from_parts(parts: dict[str, str], source: str) -> Optional[ParsedName]:
    year = int(parts["year"])
    month = int(parts["month"])
    day = int(parts["day"])
    hour = int(parts.get("hour") or 0)
    minute = int(parts.get("minute") or 0)
    second = int(parts.get("second") or 0)

    try:
        dt = datetime(year, month, day, hour, minute, second)
    except ValueError:
        return None

    ts_compact = dt.strftime("%Y%m%d%H%M%S")
    ts_underscored = dt.strftime("%Y_%m%d_%H%M%S")
    return ParsedName(
        year=dt.year,
        month=dt.month,
        day=dt.day,
        hour=dt.hour,
        minute=dt.minute,
        second=dt.second,
        ts_compact=ts_compact,
        ts_underscored=ts_underscored,
        source=source,
    )


def parse_name(basename: str) -> Optional[ParsedName]:
    """Parse a date/time from a recorder filename without assuming one layout."""
    stem = Path(basename).stem
    for pattern in TIMESTAMP_PATTERNS:
        for match in pattern.finditer(stem):
            parsed = _parsed_from_parts(match.groupdict(), source="filename")
            if parsed:
                return parsed
    return None


def parse_mtime(path: Path) -> Optional[ParsedName]:
    try:
        dt = datetime.fromtimestamp(path.stat().st_mtime)
    except OSError:
        return None
    return ParsedName(
        year=dt.year,
        month=dt.month,
        day=dt.day,
        hour=dt.hour,
        minute=dt.minute,
        second=dt.second,
        ts_compact=dt.strftime("%Y%m%d%H%M%S"),
        ts_underscored=dt.strftime("%Y_%m%d_%H%M%S"),
        source="mtime",
    )


def normalize_extensions(raw: str) -> Optional[set[str]]:
    if raw.lower() == "all":
        return None
    exts: set[str] = set()
    for item in raw.split(","):
        item = item.strip().lower()
        if not item:
            continue
        exts.add(item if item.startswith(".") else f".{item}")
    return exts


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
    NOTE: dst must be a FILE path (not a directory).
    """
    if dst.exists() and dst.is_dir():
        dst = dst / src.name

    try:
        if src.resolve() == dst.resolve():
            logging.info("Skip (same path): %s", src)
            return None
    except Exception:
        pass

    if not dst.exists():
        return dst

    if strategy == "skip":
        same = False
        try:
            if src.stat().st_size == dst.stat().st_size:
                same = True
                if hash_algo:
                    same = compute_hash(src, hash_algo) == compute_hash(dst, hash_algo)
        except OSError:
            same = False

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
        parent = dst.parent
        stem = dst.stem
        suffix = dst.suffix
        n = 1
        while True:
            candidate = parent / f"{stem}_{n}{suffix}"
            if not candidate.exists():
                return candidate
            n += 1

    raise ValueError(f"Unknown strategy: {strategy}")

def build_destination(root_out: Path, basename: str, parsed: ParsedName) -> Path:
    """
    Return the FULL destination file path:
      <root_out>/YYYY/MM/DD/<basename>
    """
    subdir = Path(f"{parsed.year:04d}") / f"{parsed.month:02d}" / f"{parsed.day:02d}"
    return root_out / subdir / basename

def move_one(
    src: Path,
    root_out: Path,
    dry_run: bool,
    strategy: str,
    hash_algo: Optional[str],
    date_source: str,
) -> str:
    basename = src.name
    parsed = parse_name(basename)
    if parsed is None and date_source == "mtime-fallback":
        parsed = parse_mtime(src)

    if parsed is None:
        logging.warning("No usable date in filename, skipping: %s", src)
        return "skipped"

    dst = build_destination(root_out, basename, parsed)

    final_dst = choose_collision_action(src, dst, strategy=strategy, hash_algo=hash_algo)
    if final_dst is None:
        return "skipped"

    if dry_run:
        logging.info("[DRY-RUN] Would move (%s date): %s -> %s", parsed.source, src, final_dst)
        return "dry"

    final_dst.parent.mkdir(parents=True, exist_ok=True)

    try:
        shutil.move(str(src), str(final_dst))
        logging.info("Moved (%s date): %s -> %s", parsed.source, src, final_dst)
        return "moved"
    except Exception as e:
        logging.error("Failed moving %s -> %s: %s", src, final_dst, e)
        return "error"

def iter_files(
    root_in: Path,
    recursive: bool,
    include_hidden: bool,
    follow_symlinks: bool,
    allowed_extensions: Optional[set[str]],
) -> Iterable[Path]:
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
            if allowed_extensions is not None and p.suffix.lower() not in allowed_extensions:
                continue
            yield p
        except OSError:
            continue

def main():
    ap = argparse.ArgumentParser(
        description="Move media files into YYYY/MM/DD/ subfolders using flexible recorder timestamps."
    )
    ap.add_argument("-i", "--input", required=True, type=Path, help="Input directory to scan")
    ap.add_argument("-o", "--output", required=True, type=Path, help="Output directory root")
    ap.add_argument("-r", "--recursive", action="store_true", help="Recurse into subdirectories")
    ap.add_argument("--follow-symlinks", action="store_true", help="Follow symlinked files")
    ap.add_argument("--include-hidden", action="store_true", help="Include dotfiles")
    ap.add_argument("--dry-run", "-n", action="store_true", help="Show actions without moving")
    ap.add_argument(
        "--extensions",
        default=",".join(sorted(DEFAULT_MEDIA_EXTENSIONS)),
        help="Comma-separated extensions to process, or 'all' (default: common audio/video)",
    )
    ap.add_argument(
        "--date-source",
        choices=["filename", "mtime-fallback"],
        default="mtime-fallback",
        help="Use only filename dates or fall back to file modified time (default: mtime-fallback)",
    )
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

    allowed_extensions = normalize_extensions(args.extensions)
    args.output.mkdir(parents=True, exist_ok=True)

    files = list(iter_files(args.input, args.recursive, args.include_hidden, args.follow_symlinks, allowed_extensions))
    if not files:
        logging.warning("No matching files found in input (check flags/extensions).")
        return

    logging.info("Discovered %d file(s) to consider.", len(files))

    counts = {"moved": 0, "skipped": 0, "dry": 0, "error": 0}
    work = [(p, args.output, args.dry_run, args.strategy, args.hash, args.date_source) for p in files]

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
