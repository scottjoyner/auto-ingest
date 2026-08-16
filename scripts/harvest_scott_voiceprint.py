#!/usr/bin/env python3
"""Harvest Scott's utterances from Neo4j, extract audio segments, and enroll as voiceprint samples."""

import json
import logging
import subprocess
import sys
from pathlib import Path

from neo4j import GraphDatabase

from auto_ingest_config import get_neo4j_env

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DB = get_neo4j_env()

MAX_CLIPS = 200
MIN_SEGMENT_SECONDS = 2.0
MAX_SEGMENT_SECONDS = 30.0

AUDIO_MOUNTS = [
    Path("/media/scott/NAS5/fileserver/audio"),
    Path("/media/scott/SSD_4TB/fileserver/audio"),
    Path("/mnt/8TB_2025/fileserver/audio"),
]


def driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def resolve_audio_path(clip_key: str) -> str | None:
    base_key = clip_key.rstrip("_F").rstrip("_R").rstrip("_")
    parts = base_key.split("_")
    if len(parts) < 3:
        return None
    year, month_day = parts[0], parts[1]
    month, day = month_day[:2], month_day[2:]
    for base in AUDIO_MOUNTS:
        for ext in [".mp3", ".wav", ".MP3", ".WAV"]:
            p = base / year / month / day / f"{base_key}{ext}"
            if p.exists():
                return str(p)
    return None


def get_scott_segments(session):
    return session.run("""
        MATCH (sp:Speaker {person_id: $uid})-[:SPOKEN_BY]-(u:Utterance)
        MATCH (u)-[:OF_SEGMENT]-(seg:Segment)
        WHERE seg.clip_key IS NOT NULL AND seg.start IS NOT NULL AND seg.end IS NOT NULL
        RETURN seg.clip_key AS clip_key,
               seg.start AS start_seconds,
               seg.end AS end_seconds
    """, uid="scott").data()


def extract_segment(audio_path: str, start: float, end: float, out_path: Path) -> bool:
    duration = end - start
    result = subprocess.run([
        "ffmpeg", "-y", "-ss", str(start), "-t", str(duration),
        "-i", audio_path, "-ac", "1", "-ar", "16000",
        "-sample_fmt", "s16", out_path
    ], capture_output=True)
    return result.returncode == 0


def enroll_clips(clip_paths: list[str], user_id: str = "scott"):
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Sophia" / "voice-agent" / "src"))
    from voice_agent.config import load_config
    from voice_agent.auth.enroll import enroll_from_files

    config = load_config(None)
    return enroll_from_files(
        config, user_id, clip_paths,
        append=True, source="harvest_segments"
    )


def main():
    drv = driver()
    with drv.session(database=NEO4J_DB) as session:
        segments = get_scott_segments(session)
    drv.close()

    logging.info(f"Found {len(segments)} raw segments for scott")
    seen = set()
    unique = []
    for seg in segments:
        key = (seg["clip_key"], seg["start_seconds"])
        if key not in seen:
            seen.add(key)
            unique.append(seg)

    logging.info(f"Deduplicated to {len(unique)} unique segments")
    clips_dir = Path("/tmp/voice_enroll_clips")
    clips_dir.mkdir(parents=True, exist_ok=True)

    enrolled_count = 0
    clip_paths = []
    for seg in unique:
        if enrolled_count >= MAX_CLIPS:
            break

        clip_key = seg["clip_key"]
        start = float(seg["start_seconds"])
        end = float(seg["end_seconds"])
        duration = end - start
        if duration < MIN_SEGMENT_SECONDS or duration > MAX_SEGMENT_SECONDS:
            continue

        audio_path = resolve_audio_path(clip_key)
        if not audio_path:
            continue

        safe_name = f"{clip_key}_{start:.0f}-{end:.0f}".replace(".", "_")
        out_path = clips_dir / f"{safe_name}.wav"
        if extract_segment(audio_path, start, end, out_path):
            clip_paths.append(str(out_path))
            enrolled_count += 1

    logging.info(f"Extracted {len(clip_paths)} valid audio clips for enrollment")
    if clip_paths:
        result = enroll_clips(clip_paths)
        print(json.dumps(result, indent=2))
    else:
        logging.warning("No clips to enroll")


if __name__ == "__main__":
    main()
