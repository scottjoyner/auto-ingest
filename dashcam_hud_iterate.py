#!/usr/bin/env python3
"""
Dashcam HUD → OCR → CSV (+ optional Neo4j upsert)

Iterates YYYY/MM/DD folders under a base directory, finds dashcam files like
`<KEY>_F.MP4`, and creates `<KEY>_metadata.csv` if missing (or --force).

OCR:
- Speed (MPH)
- Latitude (xx.xxxxxx via two slices)
- Longitude (xx.xxxxxx via two slices)
- Frame index (per-second sampling)

Neo4j (optional):
- Upserts (:Video {key}) and (:Frame {key,frame}) with properties
- Adds [:HAS_FRAME] relationships

Example:
    python dashcam_hud_iterate.py \
      --base /mnt/8TB_2025/fileserver/dashcam \
      --neo4j-uri bolt://localhost:7687 --neo4j-user neo4j --neo4j-pass password
"""

from __future__ import annotations
import os
import re
import sys
import math
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple

from datetime import datetime
import pandas as pd
import numpy as np

# MoviePy (video I/O)
from moviepy.editor import VideoFileClip

# EasyOCR (OCR)
import easyocr

# Neo4j (optional)
try:
    from neo4j import GraphDatabase, Driver
    NEO4J_AVAILABLE = True
except Exception:
    NEO4J_AVAILABLE = False


# ----------------------------
# Configuration & Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger("dashcam")

# Singleton OCR reader (loaded once)
_READER: Optional[easyocr.Reader] = None


def get_reader(lang: str = "en") -> easyocr.Reader:
    global _READER
    if _READER is None:
        log.info("Loading EasyOCR reader (once)...")
        _READER = easyocr.Reader([lang])
    return _READER


# ----------------------------
# Cropping Profile
# ----------------------------
@dataclass(frozen=True)
class CropBox:
    """Crop in reference coordinates (ref_w x ref_h). Will scale to actual video size."""
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass(frozen=True)
class HudCropProfile:
    """
    Define HUD crop rectangles in *reference* pixels (default: 2560x1440).
    These came from your original code; adjust here if the HUD changes.
    """
    ref_w: int = 2560
    ref_h: int = 1440

    # MPH block
    speed: CropBox = CropBox(20, 1370, 102, 1420)

    # Longitude split (two boxes to reduce OCR errors)
    lon_deg1: CropBox = CropBox(280, 1375, 340, 1410)
    lon_deg2: CropBox = CropBox(355, 1375, 472, 1410)

    # Latitude split (two boxes)
    lat_deg1: CropBox = CropBox(540, 1375, 600, 1410)
    lat_deg2: CropBox = CropBox(615, 1375, 735, 1410)


def scale_box(box: CropBox, w: int, h: int, ref_w: int, ref_h: int) -> Tuple[int, int, int, int]:
    """Scale a reference crop box to actual video size."""
    sx = w / ref_w
    sy = h / ref_h
    x1 = int(round(box.x1 * sx))
    y1 = int(round(box.y1 * sy))
    x2 = int(round(box.x2 * sx))
    y2 = int(round(box.y2 * sy))
    # clamp
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return x1, y1, x2, y2


# ----------------------------
# Filesystem Scanning
# ----------------------------
DATE_PATTERN = re.compile(r"^\d{4}/\d{2}/\d{2}$")  # YYYY/MM/DD


def is_valid_date_path(base: Path, p: Path) -> bool:
    try:
        rel = p.relative_to(base).as_posix().strip("/")
        if not DATE_PATTERN.match(rel):
            return False
        datetime.strptime(rel, "%Y/%m/%d")
        return True
    except Exception:
        return False


def iter_date_dirs(base: Path) -> Iterator[Path]:
    for root, dirs, files in os.walk(base):
        path = Path(root)
        if is_valid_date_path(base, path):
            yield path


def collect_keys(folder: Path) -> List[str]:
    """
    Collect `<key>` where `<key>_F.MP4` exists.
    We only create metadata for keys missing `<key>_metadata.csv` unless forced.
    """
    keys = set()
    for name in os.listdir(folder):
        if name.endswith("_F.MP4"):
            key = name[:-6]  # remove "_F.MP4"
            keys.add(key)
    return sorted(keys)


# ----------------------------
# OCR Helpers
# ----------------------------
def ocr_single(reader: easyocr.Reader, img: np.ndarray) -> str:
    """
    OCR a cropped image (numpy RGB) and return a compacted string.
    """
    # easyocr expects BGR? It accepts RGB ndarray as well; results are similar.
    out = reader.readtext(img, detail=0)
    if not out:
        return ""
    # Join tokens, normalize
    s = "".join(str(x) for x in out)
    s = s.strip()
    # Fix common OCR issues
    s = s.replace("O", "0").replace("o", "0").replace(" ", "")
    return s


def parse_speed(text: str) -> Optional[int]:
    # Keep digits only; assume MPH integer on HUD
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return None
    try:
        return int(digits)
    except Exception:
        return None


def parse_coord(part_a: str, part_b: str) -> Optional[float]:
    """
    Build a decimal string AA.BBBBBB from two OCR boxes (robust against noise).
    """
    a = "".join(ch for ch in part_a if ch in "0123456789-")
    b = "".join(ch for ch in part_b if ch in "0123456789")
    if not a or not b:
        return None
    # If a accidentally contains minus, keep it at front
    if a.count("-") > 1:
        a = a.lstrip("-")
    s = f"{a}.{b}"
    try:
        return float(s)
    except Exception:
        return None


# ----------------------------
# Per-Video Processing
# ----------------------------
@dataclass
class FrameRecord:
    key: str
    frame: int
    mph: Optional[int]
    lat: Optional[float]
    lon: Optional[float]


def extract_hud_for_video(
    video_path: Path,
    key: str,
    profile: HudCropProfile,
    sample_rate: float = 1.0,
) -> List[FrameRecord]:
    """
    Extract OCR each second (sample_rate seconds) from `<key>_F.MP4`.
    Returns list of FrameRecord.
    """
    reader = get_reader()
    records: List[FrameRecord] = []

    with VideoFileClip(str(video_path)) as clip:
        w, h = clip.w, clip.h
        duration = float(clip.duration)

        # Scale crop boxes to actual resolution
        sp = scale_box(profile.speed, w, h, profile.ref_w, profile.ref_h)
        lod1 = scale_box(profile.lon_deg1, w, h, profile.ref_w, profile.ref_h)
        lod2 = scale_box(profile.lon_deg2, w, h, profile.ref_w, profile.ref_h)
        lad1 = scale_box(profile.lat_deg1, w, h, profile.ref_w, profile.ref_h)
        lad2 = scale_box(profile.lat_deg2, w, h, profile.ref_w, profile.ref_h)

        # Sample at t=0,1,2,... (or custom step)
        step = max(1.0, sample_rate)
        t = 0.0
        frame_idx = 0
        while t < duration:
            try:
                frame = clip.get_frame(t)  # RGB ndarray (H,W,3) float[0,1]
                # convert to uint8
                frame8 = np.clip(frame * 255.0, 0, 255).astype("uint8")

                def crop(box):
                    x1, y1, x2, y2 = box
                    return frame8[y1:y2, x1:x2, :]

                speed_img = crop(sp)
                lon_a_img = crop(lod1)
                lon_b_img = crop(lod2)
                lat_a_img = crop(lad1)
                lat_b_img = crop(lad2)

                # OCR
                speed_txt = ocr_single(reader, speed_img)
                lon_a_txt = ocr_single(reader, lon_a_img)
                lon_b_txt = ocr_single(reader, lon_b_img)
                lat_a_txt = ocr_single(reader, lat_a_img)
                lat_b_txt = ocr_single(reader, lat_b_img)

                mph = parse_speed(speed_txt)
                lon = parse_coord(lon_a_txt, lon_b_txt)
                lat = parse_coord(lat_a_txt, lat_b_txt)

                records.append(
                    FrameRecord(key=key, frame=frame_idx, mph=mph, lat=lat, lon=lon)
                )
            except Exception as e:
                log.warning("Frame %s@%0.1fs failed OCR: %s", key, t, e)
                records.append(FrameRecord(key=key, frame=frame_idx, mph=None, lat=None, lon=None))

            frame_idx += 1
            t += step

    return records


# ----------------------------
# CSV Write
# ----------------------------
def write_csv(folder: Path, key: str, rows: List[FrameRecord]) -> Path:
    out = folder / f"{key}_metadata.csv"
    df = pd.DataFrame(
        [{
            "Key": r.key,
            "MPH": r.mph if r.mph is not None else "",
            "Lat": r.lat if r.lat is not None else "",
            "Long": r.lon if r.lon is not None else "",
            "Frame": r.frame,
        } for r in rows]
    )
    df.to_csv(out, index=False)
    return out


# ----------------------------
# Neo4j Upsert (Optional)
# ----------------------------
def neo4j_driver(uri: str, user: str, password: str) -> Driver:
    if not NEO4J_AVAILABLE:
        raise RuntimeError("neo4j Python driver not installed. `pip install neo4j`")
    return GraphDatabase.driver(uri, auth=(user, password))


def neo4j_prepare_constraints(tx):
    # Idempotent constraint creation (Neo4j 4+ syntax; ignore errors if already exist)
    tx.run("CREATE CONSTRAINT video_key IF NOT EXISTS FOR (v:Video) REQUIRE v.key IS UNIQUE")
    tx.run("CREATE CONSTRAINT frame_key_idx IF NOT EXISTS FOR (f:Frame) REQUIRE (f.key, f.frame) IS UNIQUE")


def neo4j_upsert_batch(driver: Driver, key: str, batch: List[FrameRecord]) -> None:
    with driver.session() as sess:
        sess.execute_write(neo4j_prepare_constraints)

        # Upsert video
        sess.run(
            """
            MERGE (v:Video {key:$key})
            ON CREATE SET v.created_at = timestamp()
            """,
            key=key,
        )

        # Upsert frames (batch)
        # Using UNWIND for efficiency
        sess.run(
            """
            MATCH (v:Video {key:$key})
            UNWIND $rows AS row
            MERGE (f:Frame {key: row.key, frame: row.frame})
              ON CREATE SET f.created_at = timestamp()
            SET f.mph = row.mph,
                f.lat = row.lat,
                f.lon = row.lon
            MERGE (v)-[:HAS_FRAME]->(f)
            """,
            key=key,
            rows=[{"key": r.key, "frame": r.frame, "mph": r.mph, "lat": r.lat, "lon": r.lon} for r in batch],
        )


# ----------------------------
# Main Iterator
# ----------------------------
def should_process(folder: Path, key: str, force: bool) -> bool:
    meta = folder / f"{key}_metadata.csv"
    video_f = folder / f"{key}_F.MP4"
    if not video_f.exists():
        return False
    if not force and meta.exists():
        return False
    return True


def process_folder(
    folder: Path,
    profile: HudCropProfile,
    step_seconds: float,
    force: bool,
    neo4j: Optional[Driver],
) -> None:
    keys = collect_keys(folder)
    if not keys:
        return

    log.info("Folder: %s | %d candidate keys", folder, len(keys))
    done = 0
    for key in keys:
        if not should_process(folder, key, force):
            continue

        video = folder / f"{key}_F.MP4"
        if not video.exists():
            continue

        log.info("Processing %s (%d/%d)", key, done + 1, len(keys))
        try:
            rows = extract_hud_for_video(video, key, profile, sample_rate=step_seconds)
            out = write_csv(folder, key, rows)
            log.info("Wrote %s (%d rows)", out.name, len(rows))

            if neo4j is not None:
                neo4j_upsert_batch(neo4j, key, rows)
                log.info("Neo4j upserted %s (%d frames)", key, len(rows))

        except Exception as e:
            log.exception("Failed on key=%s: %s", key, e)

        done += 1


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Iterate dashcam HUD OCR → CSV (+ Neo4j).")
    ap.add_argument("--base", type=Path, required=True,
                    help="Base directory (contains YYYY/MM/DD subfolders).")
    ap.add_argument("--force", action="store_true",
                    help="Recreate metadata even if CSV exists.")
    ap.add_argument("--step-seconds", type=float, default=1.0,
                    help="Sampling interval in seconds (default: 1.0).")
    ap.add_argument("--ref-w", type=int, default=2560, help="Reference HUD width.")
    ap.add_argument("--ref-h", type=int, default=1440, help="Reference HUD height.")

    # Neo4j (optional)
    ap.add_argument("--neo4j-uri", type=str, default=None, help="bolt://host:7687")
    ap.add_argument("--neo4j-user", type=str, default=None)
    ap.add_argument("--neo4j-pass", type=str, default=None)

    args = ap.parse_args()

    base: Path = args.base
    if not base.exists():
        log.error("Base path does not exist: %s", base)
        sys.exit(1)

    profile = HudCropProfile(ref_w=args.ref_w, ref_h=args.ref_h)

    neo4j_drv: Optional[Driver] = None
    if args.neo4j_uri:
        if not (args.neo4j_user and args.neo4j_pass):
            log.error("Provide --neo4j-user and --neo4j-pass with --neo4j-uri")
            sys.exit(2)
        if not NEO4J_AVAILABLE:
            log.error("neo4j Python driver not installed. `pip install neo4j`")
            sys.exit(3)
        neo4j_drv = neo4j_driver(args.neo4j_uri, args.neo4j_user, args.neo4j_pass)
        log.info("Connected to Neo4j at %s", args.neo4j_uri)

    # Iterate date dirs
    total_dirs = 0
    for folder in iter_date_dirs(base):
        total_dirs += 1
        process_folder(
            folder=folder,
            profile=profile,
            step_seconds=args.step_seconds,
            force=args.force,
            neo4j=neo4j_drv,
        )

    log.info("Done. Scanned %d date folders under %s", total_dirs, base)


if __name__ == "__main__":
    main()
