#!/usr/bin/env python3
"""Salvage clips marked failed with `decode_failed`.

Many source MP4s are re-muxable even when ffmpeg refuses to decode them via the
default path. This pass re-muxes each failed clip to a clean container (and, as a
fallback, re-encodes) and retries frame extraction + vision description. Recovered
clips are stored and their `failed` flag is cleared.

Usage:
  python3 dashcam_salvage.py --base /mnt/8TB_2025/fileserver/dashcam --limit 50
  python3 dashcam_salvage.py --base /mnt/8TB_2025/fileserver/dashcam --all-failed
"""
import os
import sys
import argparse
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dashcam_frame_vision import (
    extract_minute_first_frames, describe, store_frame,
    mark_failed, parse_key_datetime, view_of,
)
from neo4j import GraphDatabase

SALVAGE_DIR = "/mnt/8TB_2025/fileserver/dashcam/_salvage"


def find_mp4(base, key):
    for b in base if isinstance(base, (list, tuple)) else [base]:
        for root, dirs, files in os.walk(b):
            if f"{key}.MP4" in files:
                return os.path.join(root, f"{key}.MP4")
    return None


def salvage(in_path, out_path, reencode=False):
    if reencode:
        cmd = ["ffmpeg", "-y", "-err_detect", "ignore_err", "-i", in_path,
               "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac", out_path]
    else:
        cmd = ["ffmpeg", "-y", "-err_detect", "ignore_err", "-i", in_path,
               "-c", "copy", "-movflags", "+faststart", out_path]
    r = subprocess.run(cmd, timeout=300, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if r.returncode != 0 or not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        return False
    # confirm it actually decodes now
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                            "-of", "csv=p=0", out_path],
                           timeout=30, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    v = probe.stdout.decode().strip()
    try:
        return float(v) > 0
    except Exception:
        return os.path.getsize(out_path) > 10000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/mnt/8TB_2025/fileserver/dashcam")
    ap.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    ap.add_argument("--neo4j-user", default="neo4j")
    ap.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD", "knowledge_graph_2026"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--all-failed", action="store_true", help="also retry timeout_or_hang clips")
    ap.add_argument("--vision-url", default="http://100.85.64.117:1234/v1/chat/completions")
    ap.add_argument("--vision-model", default="qwen3.5-0.8b-mlx")
    ap.add_argument("--max-minutes", type=int, default=5)
    ap.add_argument("--vision-retries", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--reencode", action="store_true", help="force re-encode fallback first")
    args = ap.parse_args()

    os.makedirs(SALVAGE_DIR, exist_ok=True)
    driver = GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_password))
    sys_prompt = "You are a precise dashcam scene analyst."
    default_prompt = ("You are a dashcam scene analyst. Describe this dashcam frame concisely and "
                      "factually for later semantic search. Cover: road type and condition, weather "
                      "and lighting, vehicles, pedestrians/cyclists, signage, and notable events. "
                      "2-4 sentences.")

    BASES = [args.base, "/mnt/8TBHDD/fileserver/dashcam"]
    with driver.session() as sess:
        if args.all_failed:
            rows = sess.run("MATCH (c:DashcamClip {failed:true}) RETURN c.key AS key, c.path AS path").data()
        else:
            rows = sess.run("MATCH (c:DashcamClip {failed:true}) "
                            "WHERE c.failure_reason IS NULL OR c.failure_reason = 'decode_failed' "
                            "RETURN c.key AS key, c.path AS path").data()
    print(f"[info] {len(rows)} failed clips to attempt", flush=True)

    recovered = 0
    for i, row in enumerate(rows, 1):
        key, path = row["key"], row["path"]
        if not path or not os.path.exists(path):
            path = find_mp4(BASES, key)
        if not path or not os.path.exists(path):
            print(f"[skip] {key}: source file not found", flush=True)
            continue
        v = view_of(key)
        out_path = os.path.join(SALVAGE_DIR, f"{key}.mp4")
        ok = salvage(path, out_path, reencode=args.reencode)
        if not ok and not args.reencode:
            ok = salvage(path, out_path, reencode=True)
        if not ok:
            print(f"[fail] {key}: still unreadable after re-mux/re-encode", flush=True)
            with driver.session() as sess:
                sess.run("MATCH (c:DashcamClip {key:$key}) SET c.failure_reason='unrecoverable'", key=key)
            continue
        frames = list(extract_minute_first_frames(out_path, args.max_minutes))
        if not frames:
            print(f"[fail] {key}: no frames after salvage", flush=True)
            if os.path.exists(out_path):
                os.unlink(out_path)
            with driver.session() as sess:
                sess.run("MATCH (c:DashcamClip {key:$key}) SET c.failure_reason='unrecoverable'", key=key)
            continue
        stored = 0
        with driver.session() as sess:
            for m, t_sec, png in frames:
                desc = None
                for attempt in range(args.vision_retries):
                    try:
                        desc = describe(png, args.vision_url, args.vision_model, default_prompt,
                                        sys_prompt, args.timeout)
                        break
                    except Exception:
                        if attempt < args.vision_retries - 1:
                            continue
                if desc:
                    store_frame(sess, key, v, m, t_sec, desc, args.vision_model, path=path)
                    stored += 1
            if stored:
                sess.run("MATCH (c:DashcamClip {key:$key}) "
                         "SET c.failed=false, c.failure_reason=null, c.processed=true, "
                         "c.processed_at=datetime(), c.recovered=true", key=key)
        if os.path.exists(out_path):
            os.unlink(out_path)
        if stored:
            recovered += 1
            print(f"[recovered] {key}: {stored} frames ({i}/{len(rows)})", flush=True)
        if args.limit and recovered >= args.limit:
            print(f"[info] reached --limit {args.limit}", flush=True)
            break
    print(f"[done] recovered {recovered} clips", flush=True)
    driver.close()


if __name__ == "__main__":
    main()
