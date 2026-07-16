#!/usr/bin/env python3
"""Build timelapse versions of merged dashcam videos — SAFE streaming edition.

Original `timelapse.py` used MoviePy's ``vfx.speedx`` which DECODES THE
ENTIRE CLIP THROUGH RAM. On a multi-hour 4K merged ``_FR.MP4`` that
exhausted memory and, combined with writing a temp file next to the source on
the same physical volume, corrupted source footage on at least one run
(2023 drive loss). This rewrite uses **ffmpeg directly** (streaming, bounded
memory) and adds hard guardrails:

  * NEVER overwrites or writes a temp next to the source. Output goes to a
    separate ``--out-dir`` (default: a ``_timelapse/`` sibling dir) unless
    ``--in-place`` is given explicitly.
  * Refuses to run if free disk space < ``--min-free-gb`` (default 20 GB),
    or if the input is larger than ``--max-input-gb`` (default 30 GB) unless
    ``--force``.
  * ``--dry-run`` shows the exact ffmpeg command without executing.
  * Source files are only READ, never modified or deleted.

Input:  <key>_FR.MP4
Output: <out-dir>/<key>_FRTL.MP4  (or <key>_FRTL.MP4 next to input with --in-place)
"""
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Optional, Tuple

DEFAULT_SPEEDUP = 30.0
DEFAULT_FPS = 30.0
DEFAULT_CRF = 23
DEFAULT_PRESET = "veryfast"
DEFAULT_OUT_DIRNAME = "_timelapse"
MIN_FREE_GB_DEFAULT = 20.0
MAX_INPUT_GB_DEFAULT = 30.0


def iter_fr_videos(root: Path, recursive: bool) -> Iterable[Path]:
    pattern = "**/*_FR.MP4" if recursive else "*_FR.MP4"
    for p in sorted(root.glob(pattern)):
        if p.is_file():
            yield p


def output_path_for(src: Path, out_dir: Optional[Path], in_place: bool) -> Path:
    if in_place:
        stem = src.stem  # "<key>_FR"
        base = stem[:-3] if stem.endswith("_FR") else stem
        return src.with_name(f"{base}_FRTL.MP4")
    out_dir = out_dir or src.parent / DEFAULT_OUT_DIRNAME
    return out_dir / f"{src.stem}_FRTL.MP4"


def _free_space_gb(path: Path) -> float:
    try:
        return shutil.disk_usage(str(path)).free / (1024 ** 3)
    except OSError:
        return float("inf")


def build_timelapse(
    src: Path,
    dst: Path,
    *,
    speedup: float,
    out_fps: float,
    crf: int,
    preset: str,
    audio: bool,
    dry_run: bool,
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Streaming ffmpeg: setpts speeds up WITHOUT loading the whole clip.
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-filter:v", f"setpts=PTS/{speedup}",
        "-r", str(out_fps),
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
    ]
    if audio:
        cmd += ["-c:a", "aac", "-b:a", "128k"]
    else:
        cmd += ["-an"]
    cmd.append(str(dst))

    if dry_run:
        logging.info("[DRY-RUN] %s -> %s", src.name, dst)
        logging.info("    %s", " ".join(cmd))
        return
    logging.info("Timelapse %s (x%s, %sfps, crf%s) -> %s",
                  src.name, speedup, out_fps, crf, dst)
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    logging.info("Wrote %s (%.1f MB)", dst, dst.stat().st_size / 1e6)


def process_root(
    root: Path,
    *,
    recursive: bool,
    overwrite: bool,
    out_dir: Optional[Path],
    in_place: bool,
    speedup: float,
    out_fps: float,
    crf: int,
    preset: str,
    audio: bool,
    dry_run: bool,
    min_free_gb: float,
    max_input_gb: float,
    force: bool,
) -> Tuple[int, int]:
    planned = 0
    completed = 0
    for src in iter_fr_videos(root, recursive=recursive):
        dst = output_path_for(src, out_dir, in_place)
        if dst.exists() and not overwrite:
            logging.debug("Skip (exists): %s", dst)
            continue
        size_gb = src.stat().st_size / (1024 ** 3)
        if size_gb > max_input_gb and not force:
            logging.warning("Skip (%.1f GB > max-input %.0f GB, --force to override): %s",
                           size_gb, max_input_gb, src.name)
            continue
        free = _free_space_gb(dst.parent)
        if free < min_free_gb and not force:
            logging.error("ABORT %s: only %.1f GB free < min-free %.0f GB on %s",
                          src.name, free, min_free_gb, dst.parent)
            continue
        planned += 1
        try:
            build_timelapse(
                src, dst, speedup=speedup, out_fps=out_fps, crf=crf,
                preset=preset, audio=audio, dry_run=dry_run,
            )
            completed += 1
        except subprocess.CalledProcessError as e:
            logging.exception("Failed: %s -> %s: %s", src, dst, e)
    return completed, planned


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SAFE streaming timelapse from *_FR.MP4 (ffmpeg, bounded memory).")
    p.add_argument("path", nargs="?", default=".", help="Directory to scan (default: current dir).")
    p.add_argument("--recursive", action="store_true", help="Recurse into subdirectories.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing *_FRTL.MP4 outputs.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output directory (default: <root>/_timelapse/). NEVER the source volume root.")
    p.add_argument("--in-place", action="store_true",
                   help="Write output next to the source (NOT recommended; off by default).")
    p.add_argument("--dry-run", action="store_true", help="Show ffmpeg commands without running.")
    p.add_argument("--speedup", type=float, default=DEFAULT_SPEEDUP, help="Speed-up factor (default 30x).")
    p.add_argument("--fps", type=float, default=DEFAULT_FPS, help="Output fps (default 30).")
    p.add_argument("--crf", type=int, default=DEFAULT_CRF, help="x264 CRF 0-51 (default 23).")
    p.add_argument("--preset", type=str, default=DEFAULT_PRESET,
                   help="x264 preset (default veryfast).")
    p.add_argument("--keep-audio", dest="audio", action="store_true", help="Keep audio (default: strip).")
    p.add_argument("--min-free-gb", type=float, default=MIN_FREE_GB_DEFAULT,
                   help="Refuse to write if free space < this (default 20 GB).")
    p.add_argument("--max-input-gb", type=float, default=MAX_INPUT_GB_DEFAULT,
                   help="Skip inputs larger than this unless --force (default 30 GB).")
    p.add_argument("--force", action="store_true", help="Override free-space / input-size guards.")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(message)s")
    root = Path(args.path).resolve()
    if not root.exists() or not root.is_dir():
        logging.error("Path not found or not a directory: %s", root)
        return 2
    if args.speedup <= 0 or args.fps <= 0 or not (0 <= args.crf <= 51):
        logging.error("Invalid speedup/fps/crf")
        return 2

    done, planned = process_root(
        root, recursive=args.recursive, overwrite=args.overwrite,
        out_dir=args.out_dir, in_place=args.in_place,
        speedup=args.speedup, out_fps=args.fps, crf=args.crf,
        preset=args.preset, audio=args.audio, dry_run=args.dry_run,
        min_free_gb=args.min_free_gb, max_input_gb=args.max_input_gb,
        force=args.force,
    )
    logging.info("Completed %d/%d timelapse(s).", done, planned)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
