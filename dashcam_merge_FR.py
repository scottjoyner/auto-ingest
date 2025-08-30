#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import sys
import math
import shutil
import argparse
import logging
from pathlib import Path
from typing import Iterable, List, Tuple, Set
from datetime import datetime

from moviepy.editor import (
    VideoFileClip,
    vfx,
    clips_array,
    CompositeVideoClip,
)

DATE_DIR_RE = re.compile(r"^\d{4}/\d{2}/\d{2}$")

def is_valid_date_structure(base: Path, d: Path) -> bool:
    try:
        rel = d.relative_to(base).as_posix()
    except ValueError:
        return False
    if not DATE_DIR_RE.match(rel):
        return False
    y, m, day = rel.split("/")
    try:
        datetime(year=int(y), month=int(m), day=int(day))
        return True
    except ValueError:
        return False

def iter_date_dirs(base: Path) -> Iterable[Path]:
    for y in sorted(p for p in base.iterdir() if p.is_dir() and p.name.isdigit() and len(p.name) == 4):
        for m in sorted(p for p in y.iterdir() if p.is_dir() and p.name.isdigit() and len(p.name) == 2):
            for d in sorted(p for p in m.iterdir() if p.is_dir() and p.name.isdigit() and len(p.name) == 2):
                if is_valid_date_structure(base, d):
                    yield d

def pair_keys(directory: Path, overwrite: bool) -> List[str]:
    f_files = {p.stem[:-2] for p in directory.glob("*_F.MP4") if p.is_file() and p.stem.endswith("_F")}
    r_files = {p.stem[:-2] for p in directory.glob("*_R.MP4") if p.is_file() and p.stem.endswith("_R")}
    keys = sorted(f_files & r_files)
    if not overwrite:
        keys = [k for k in keys if not (directory / f"{k}_FR.MP4").exists()]
    return keys

def crop_by_percent(clip: VideoFileClip, top: float, bottom: float) -> VideoFileClip:
    H = clip.h
    y1 = int(H * top); y2 = int(H * (1.0 - bottom))
    return clip.crop(y1=y1, y2=y2)

def extract_overlay_from_front(front: VideoFileClip, x: int, y_from_bottom: int, width: int, height: int) -> VideoFileClip:
    y1 = max(0, front.h - (y_from_bottom + height))
    x1 = max(0, x)
    w = min(width, front.w - x1)
    h = min(height, front.h - y1)
    return front.crop(x1=x1, y1=y1, width=w, height=h)

def compose_front_rear(
    front_path: Path,
    rear_path: Path,
    out_path: Path,
    *,
    front_top_crop: float,
    front_bottom_crop: float,
    rear_top_crop: float,
    rear_bottom_crop: float,
    mirror_rear: bool,
    overlay_x: int,
    overlay_y_from_bottom: int,
    overlay_w: int,
    overlay_h: int,
    x264_crf: int,
    x264_preset: str,
    audio_bitrate: str | None,
    fps: float | None,
    dry_run: bool,
) -> None:
    if dry_run:
        logging.info(f"[DRY-RUN] Would create: {out_path.name} from {front_path.name} + {rear_path.name}")
        return

    tmp_path = out_path.with_suffix(".mp4.tmp")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with VideoFileClip(str(front_path)) as clip_f, VideoFileClip(str(rear_path)) as clip_r:
        clip_r = clip_r.without_audio()
        if mirror_rear:
            clip_r = clip_r.fx(vfx.mirror_x)

        clip_f_c = crop_by_percent(clip_f, front_top_crop, front_bottom_crop)
        clip_r_c = crop_by_percent(clip_r, rear_top_crop, rear_bottom_crop)
        if clip_r_c.w != clip_f_c.w:
            clip_r_c = clip_r_c.resize(width=clip_f_c.w)

        background = clips_array([[clip_f_c], [clip_r_c]])
        overlay = extract_overlay_from_front(
            clip_f, x=overlay_x, y_from_bottom=overlay_y_from_bottom, width=overlay_w, height=overlay_h
        )

        pos_x = (background.w - overlay.w) / 2
        pos_y = ((background.h - overlay.h) / 2) + 100
        final = CompositeVideoClip([background, overlay.set_position((pos_x, pos_y))])

        target_fps = fps or clip_f.fps or 30
        logging.info(f"Writing {out_path.name} (fps={target_fps}, crf={x264_crf}, preset={x264_preset})")
        final.write_videofile(
            str(tmp_path),
            codec="libx264",
            audio_codec="aac",
            audio_bitrate=audio_bitrate,
            fps=target_fps,
            preset=x264_preset,
            ffmpeg_params=["-crf", str(x264_crf)],
            threads=max(1, (os.cpu_count() or 2) - 1),
            temp_audiofile=str(out_path.with_suffix(".audio.m4a.tmp")),
            remove_temp=True,
            verbose=False,
            logger=None,
        )

    tmp_path.replace(out_path)
    logging.info(f"Wrote {out_path}")

def process_date_dir(
    date_dir: Path,
    *,
    overwrite: bool,
    dry_run: bool,
    front_top_crop: float,
    front_bottom_crop: float,
    rear_top_crop: float,
    rear_bottom_crop: float,
    mirror_rear: bool,
    overlay_x: int,
    overlay_y_from_bottom: int,
    overlay_w: int,
    overlay_h: int,
    x264_crf: int,
    x264_preset: str,
    audio_bitrate: str | None,
    fps: float | None,
) -> Tuple[int, int]:
    keys = pair_keys(date_dir, overwrite=overwrite)
    total = len(keys)
    if total == 0:
        logging.debug(f"No merge candidates in {date_dir}")
        return (0, 0)

    done = 0
    logging.info(f"{total} video(s) to process in {date_dir}")
    for i, key in enumerate(keys, 1):
        front = date_dir / f"{key}_F.MP4"
        rear  = date_dir / f"{key}_R.MP4"
        out   = date_dir / f"{key}_FR.MP4"
        logging.info(f"[{i}/{total}] {key}")
        try:
            compose_front_rear(
                front, rear, out,
                front_top_crop=front_top_crop,
                front_bottom_crop=front_bottom_crop,
                rear_top_crop=rear_top_crop,
                rear_bottom_crop=rear_bottom_crop,
                mirror_rear=mirror_rear,
                overlay_x=overlay_x,
                overlay_y_from_bottom=overlay_y_from_bottom,
                overlay_w=overlay_w,
                overlay_h=overlay_h,
                x264_crf=x264_crf,
                x264_preset=x264_preset,
                audio_bitrate=audio_bitrate,
                fps=fps,
                dry_run=dry_run,
            )
            done += 1
        except Exception as e:
            logging.exception(f"Failed to merge {key}: {e}")
    return (done, total)

def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge dashcam front+rear pairs into stacked _FR.MP4 outputs.")
    # MULTI-BASE SUPPORT
    p.add_argument("--base", dest="bases", action="append", default=[],
                   help="Base directory containing YYYY/MM/DD subfolders. Repeat flag for multiple roots.")
    p.add_argument("--bases-file", type=Path, default=None,
                   help="Optional text file with one base path per line. Lines starting with # are ignored.")
    # MULTI SINGLE-DIR SUPPORT
    p.add_argument("--single-dir", dest="single_dirs", action="append", default=[],
                   help="Specific YYYY/MM/DD directory to process. Repeat flag to target multiple days (can be under different bases).")
    p.add_argument("--overwrite", action="store_true", help="Rebuild outputs even if *_FR.MP4 already exists.")
    p.add_argument("--dry-run", action="store_true", help="List actions without producing output.")

    # Cropping (fractions of height)
    p.add_argument("--front-top-crop", type=float, default=0.30, help="Fraction of front height to crop from top (default: 0.30).")
    p.add_argument("--front-bottom-crop", type=float, default=0.05, help="Fraction of front height to crop from bottom (default: 0.05).")
    p.add_argument("--rear-top-crop", type=float, default=0.50, help="Fraction of rear height to crop from top (default: 0.50).")
    p.add_argument("--rear-bottom-crop", type=float, default=0.05, help="Fraction of rear height to crop from bottom (default: 0.05).")
    p.add_argument("--no-mirror-rear", dest="mirror_rear", action="store_false", default=True,
                   help="Do not mirror the rear view horizontally (default is mirrored).")

    # Overlay rectangle (cropped from front)
    p.add_argument("--overlay-x", type=int, default=20, help="Overlay crop X (from left) on the front clip (default: 20).")
    p.add_argument("--overlay-y", type=int, default=20, help="Overlay crop Y from bottom on the front clip (default: 20).")
    p.add_argument("--overlay-w", type=int, default=165, help="Overlay crop width (default: 165).")
    p.add_argument("--overlay-h", type=int, default=50, help="Overlay crop height (default: 50).")

    # Encoding
    p.add_argument("--crf", type=int, default=20, help="x264 Constant Rate Factor (lower=better; default: 20).")
    p.add_argument("--preset", type=str, default="medium",
                   help="x264 preset: ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow.")
    p.add_argument("--audio-bitrate", type=str, default=None, help="AAC audio bitrate (e.g., 128k). Defaults to ffmpeg's choice.")
    p.add_argument("--fps", type=float, default=None, help="Force output FPS (default: inherit front FPS).")

    p.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR).")
    return p.parse_args(argv)

def _read_bases_file(path: Path) -> List[str]:
    bases = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            bases.append(line)
    return bases

def gather_date_dirs(bases: List[Path], single_dirs: List[Path]) -> List[Tuple[Path, Path]]:
    """
    Return list of (base, datedir) pairs to process.
    - If single_dirs provided: include those (validated under a matching base if possible).
    - Otherwise: scan all bases for YYYY/MM/DD folders.
    De-duplicates identical datedir paths.
    """
    result: Set[Path] = set()

    # Explicit single dirs first
    for sd in single_dirs:
        if not sd.is_dir():
            logging.warning(f"--single-dir not found or not a directory, skipping: {sd}")
            continue
        # If it sits under one of the bases, validate structure; else accept if it *looks* like YYYY/MM/DD
        matched_base = None
        for b in bases:
            try:
                sd.relative_to(b)
                matched_base = b; break
            except ValueError:
                continue
        if matched_base and not is_valid_date_structure(matched_base, sd):
            logging.warning(f"--single-dir is not a valid date dir under base {matched_base}: {sd}")
            continue
        # Light regex check if not under a provided base
        rel = "/".join(sd.parts[-3:])
        if not DATE_DIR_RE.match(rel):
            logging.warning(f"--single-dir does not look like YYYY/MM/DD: {sd}")
            continue
        result.add(sd.resolve())

    # If no single-dirs, scan all bases
    if not result:
        for b in bases:
            if not b.is_dir():
                logging.warning(f"--base not found or not a directory, skipping: {b}")
                continue
            for d in iter_date_dirs(b):
                result.add(d.resolve())

    # Return with the base that owns each datedir if any (best-effort)
    pairs: List[Tuple[Path, Path]] = []
    for d in sorted(result):
        owner = None
        for b in bases:
            try:
                d.relative_to(b)
                owner = b; break
            except ValueError:
                continue
        pairs.append((owner if owner else Path("/"), d))
    return pairs

def main(argv: List[str]) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    bases: List[str] = list(args.bases)
    if args.bases_file:
        if args.bases_file.is_file():
            bases += _read_bases_file(args.bases_file)
        else:
            logging.error(f"--bases-file not found: {args.bases_file}")
            return 2
    if not bases and not args.single_dirs:
        # default to your common base if nothing was supplied
        bases = ["/mnt/8TB_2025/fileserver/dashcam/"]

    base_paths = [Path(b).resolve() for b in bases]
    single_dirs = [Path(s).resolve() for s in args.single_dirs]

    # Validate crops
    for name in ["front_top_crop", "front_bottom_crop", "rear_top_crop", "rear_bottom_crop"]:
        val = getattr(args, name)
        if not (0.0 <= val <= 0.99):
            logging.error(f"--{name.replace('_','-')} must be in [0, 0.99], got {val}")
            return 2

    worklist = gather_date_dirs(base_paths, single_dirs)
    if not worklist:
        logging.warning("No date directories found to process.")
        return 0

    total_done = 0
    total_plan = 0
    for base, d in worklist:
        logging.info(f"Scanning: {d} (base: {base})")
        done, planned = process_date_dir(
            d,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            front_top_crop=args.front_top_crop,
            front_bottom_crop=args.front_bottom_crop,
            rear_top_crop=args.rear_top_crop,
            rear_bottom_crop=args.rear_bottom_crop,
            mirror_rear=args.mirror_rear,
            overlay_x=args.overlay_x,
            overlay_y_from_bottom=args.overlay_y,
            overlay_w=args.overlay_w,
            overlay_h=args.overlay_h,
            x264_crf=args.crf,
            x264_preset=args.preset,
            audio_bitrate=args.audio_bitrate,
            fps=args.fps,
        )
        total_done += done
        total_plan += planned

    logging.info(f"Completed {total_done}/{total_plan} merge(s).")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
