#!/usr/bin/env python3
"""
Build timelapse versions of merged dashcam videos.

Input file pattern:
  <key>_FR.MP4   (merged front+rear)

Output file:
  <key>_FRTL.MP4 (timelapse of the merged video)

Default behavior:
  - 30x speed-up (i.e., output duration ~= input/30)
  - Output FPS = 30
  - No recursion unless --recursive is set
  - Skips outputs that already exist unless --overwrite is used
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Iterable, List, Tuple

from moviepy.editor import VideoFileClip, vfx

# ----------------------------
# Discovery
# ----------------------------

def iter_fr_videos(root: Path, recursive: bool) -> Iterable[Path]:
    """Yield paths to files matching *_FR.MP4 under root."""
    pattern = "**/*_FR.MP4" if recursive else "*_FR.MP4"
    for p in root.glob(pattern):
        if p.is_file():
            yield p

def output_path_for(input_fr: Path) -> Path:
    """Return path for the timelapse output *_FRTL.MP4 next to input."""
    stem = input_fr.stem  # "<key>_FR"
    if not stem.endswith("_FR"):
        return input_fr.with_name(f"{stem}_FRTL.MP4")
    base = stem[:-3]  # strip "_FR"
    return input_fr.with_name(f"{base}_FRTL.MP4")

# ----------------------------
# Timelapse builder
# ----------------------------

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
    """
    Create timelapse from src -> dst:
    - speedx factor 'speedup'
    - output fps 'out_fps'
    - H.264 with CRF/preset
    - audio kept or stripped
    """
    if dry_run:
        logging.info(f"[DRY-RUN] {src.name} -> {dst.name} (speedup={speedup}, fps={out_fps}, crf={crf}, preset={preset})")
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")

    # Open safely and always close
    with VideoFileClip(str(src)) as clip:
        # speed up (drop audio if requested)
        if not audio and clip.audio is not None:
            clip = clip.without_audio()

        sped = clip.fx(vfx.speedx, factor=speedup)

        # If input has odd dimensions, let ffmpeg handle; MoviePy will pass thru
        # Write to tmp file first
        logging.info(f"Writing timelapse: {dst.name} (x{speedup}, fps={out_fps}, crf={crf}, preset={preset})")
        sped.write_videofile(
            str(tmp),
            codec="libx264",
            audio_codec="aac" if audio else None,
            audio=audio,
            fps=out_fps,
            preset=preset,
            ffmpeg_params=["-crf", str(crf)],
            threads=max(1, (os.cpu_count() or 2) - 1),
            temp_audiofile=str(dst.with_suffix(".audio.m4a.tmp")),
            remove_temp=True,
            verbose=False,
            logger=None,
        )

    tmp.replace(dst)
    logging.info(f"Wrote {dst}")

# ----------------------------
# Orchestration
# ----------------------------

def process_root(
    root: Path,
    *,
    recursive: bool,
    overwrite: bool,
    speedup: float,
    out_fps: float,
    crf: int,
    preset: str,
    audio: bool,
    dry_run: bool,
) -> Tuple[int, int]:
    """
    Process one root directory; returns (completed, planned).
    """
    planned = 0
    completed = 0

    for src in sorted(iter_fr_videos(root, recursive=recursive)):
        dst = output_path_for(src)
        if dst.exists() and not overwrite:
            logging.debug(f"Skip (exists): {dst}")
            continue
        planned += 1
        try:
            build_timelapse(
                src, dst,
                speedup=speedup,
                out_fps=out_fps,
                crf=crf,
                preset=preset,
                audio=audio,
                dry_run=dry_run,
            )
            completed += 1
        except Exception as e:
            logging.exception(f"Failed: {src} -> {dst}: {e}")

    return completed, planned

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create timelapse videos from merged *_FR.MP4 files.")
    p.add_argument("path", nargs="?", default=".", help="Directory to scan (default: current dir).")
    p.add_argument("--recursive", action="store_true", help="Recurse into subdirectories.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing *_FRTL.MP4 outputs.")
    p.add_argument("--dry-run", action="store_true", help="Show what would happen without writing files.")

    # Timelapse controls
    p.add_argument("--speedup", type=float, default=30.0, help="Speed-up factor (default: 30x).")
    p.add_argument("--fps", type=float, default=30.0, help="Output frames per second (default: 30).")

    # Encoding
    p.add_argument("--crf", type=int, default=20, help="x264 CRF (lower=better quality; default: 20).")
    p.add_argument("--preset", type=str, default="medium",
                   help="x264 preset: ultrafast/superfast/veryfast/faster/fast/medium/slow/slower/veryslow.")
    p.add_argument("--keep-audio", dest="audio", action="store_true", help="Keep audio in timelapse (default: strip).")

    p.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR).")
    return p.parse_args()

def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    root = Path(args.path).resolve()
    if not root.exists() or not root.is_dir():
        logging.error(f"Path not found or not a directory: {root}")
        return 2

    # Guard rails
    if args.speedup <= 0:
        logging.error("--speedup must be > 0")
        return 2
    if args.fps <= 0:
        logging.error("--fps must be > 0")
        return 2
    if not (0 <= args.crf <= 51):
        logging.error("--crf must be in [0,51]")
        return 2

    done, planned = process_root(
        root,
        recursive=args.recursive,
        overwrite=args.overwrite,
        speedup=args.speedup,
        out_fps=args.fps,
        crf=args.crf,
        preset=args.preset,
        audio=args.audio,
        dry_run=args.dry_run,
    )

    logging.info(f"Completed {done}/{planned} timelapse(s).")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
