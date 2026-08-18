#!/usr/bin/env python3
"""Batch-describe dashcam clips via a vision LLM (default: MacBook Air qwen3.5-0.8b-mlx).

For every dashcam clip under the fileserver dashcam base (YYYY/MM/DD/{key}.MP4),
extract the FIRST frame of each minute, send it to the vision endpoint, and store
a DashcamFrame node (description + metadata) in the bodycam KG.

Resumable: a (key, minute) pair that already has a DashcamFrame node is skipped,
so re-runs only process new clips / new minutes.

Stored descriptions are embedded into emb_e5_large (same space as the rest of the
bodycam text) by reembed.py once "DashcamFrame" is registered in SCHEMA.
"""
import os
import sys
import argparse
import base64
import time
from neo4j import GraphDatabase
import requests

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
try:
    from auto_ingest_config import get_fileserver_path
except Exception:
    get_fileserver_path = None

import cv2

DEFAULT_PROMPT = (
    "You are a dashcam scene analyst. Describe this dashcam frame concisely and "
    "factually for later semantic search. Cover: road type and condition, weather "
    "and lighting, vehicles (type, color, position, direction), pedestrians or "
    "cyclists, signage, and any notable or unusual event. Be specific and avoid "
    "guessing details not visible. 2-4 sentences."
)


def walk_clip_dirs(base):
    out = []
    for root, dirs, files in os.walk(base):
        if any(f.endswith(".MP4") for f in files):
            out.append(root)
    return out


def list_clip_keys(directory):
    return sorted(f[:-4] for f in os.listdir(directory) if f.endswith(".MP4"))


def view_of(key):
    k = key.upper()
    if k.endswith("_FR"):
        return "FR"
    if k.endswith("_F"):
        return "F"
    if k.endswith("_R"):
        return "R"
    return "?"


def extract_minute_first_frames(mp4_path, max_minutes=None, max_size=640):
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    dur = (n / fps) if fps else 0
    total_min = int(dur // 60)
    if max_minutes is not None:
        total_min = min(total_min, max_minutes)
    for m in range(total_min + 1):
        cap.set(cv2.CAP_PROP_POS_MSEC, m * 60000.0)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        h, w = frame.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        ok, buf = cv2.imencode(".png", frame)
        if not ok:
            continue
        yield m, float(m * 60), buf.tobytes()
    cap.release()


def describe(png_bytes, vision_url, vision_model, prompt, sys_prompt, timeout):
    b64 = base64.b64encode(png_bytes).decode()
    data_url = f"data:image/png;base64,{b64}"
    payload = {
        "model": vision_model,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": prompt},
            ]},
        ],
        "max_tokens": 300,
        "temperature": 0.2,
    }
    r = requests.post(vision_url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def already_done(sess, key, minute):
    rec = sess.run(
        "MATCH (f:DashcamFrame {key:$key, minute:$minute}) RETURN count(f) AS c",
        key=key, minute=minute,
    ).single()
    return rec and rec["c"] > 0


def store_frame(sess, key, view, minute, t_sec, desc, model):
    sess.run(
        """
        MERGE (c:DashcamClip {key:$key}) ON CREATE SET c.key=$key
        MERGE (f:DashcamFrame {key:$key, minute:$minute})
          ON CREATE SET f.description=$desc, f.t_sec=$t, f.view=$view,
                       f.model=$model, f.created=datetime()
          ON MATCH  SET f.description=$desc, f.model=$model, f.updated=datetime()
        MERGE (c)-[:HAS_FRAME]->(f)
        """,
        key=key, view=view, minute=minute, t=t_sec, desc=desc, model=model,
    )


def main():
    ap = argparse.ArgumentParser()
    default_base = get_fileserver_path("dashcam") if get_fileserver_path else "/mnt/8TB_2025/fileserver/dashcam"
    ap.add_argument("--base", default=default_base)
    ap.add_argument("--vision-url", default="http://100.85.64.117:1234/v1/chat/completions")
    ap.add_argument("--vision-model", default="qwen3.5-0.8b-mlx")
    ap.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    ap.add_argument("--neo4j-user", default="neo4j")
    ap.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD", "knowledge_graph_2026"))
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--sys-prompt", default="You are a precise dashcam scene analyst.")
    ap.add_argument("--max-minutes", type=int, default=None, help="cap minutes per clip (None=all)")
    ap.add_argument("--max-size", type=int, default=640, help="max frame dimension (downscale for batch sustainability)")
    ap.add_argument("--limit-clips", type=int, default=None, help="process only N clips (testing)")
    ap.add_argument("--view", choices=["F", "R", "FR", "both"], default="both")
    ap.add_argument("--sleep", type=float, default=0.1, help="seconds between vision calls")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--dry-run", action="store_true", help="extract+count but skip vision/store")
    args = ap.parse_args()

    driver = GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_password))
    clip_dirs = walk_clip_dirs(args.base)
    print(f"[info] found {len(clip_dirs)} clip directories under {args.base}", flush=True)
    processed = 0
    framed = 0
    skipped = 0
    errors = 0
    with driver.session() as sess:
        for d in clip_dirs:
            keys = list_clip_keys(d)
            for key in keys:
                v = view_of(key)
                if args.view != "both" and v != args.view:
                    continue
                processed += 1
                if args.limit_clips and processed > args.limit_clips:
                    print(f"[info] reached --limit-clips {args.limit_clips}; stopping", flush=True)
                    break
                mp4 = os.path.join(d, f"{key}.MP4")
                if not os.path.exists(mp4):
                    continue
                try:
                    for minute, t_sec, png in extract_minute_first_frames(mp4, args.max_minutes, args.max_size):
                        if already_done(sess, key, minute):
                            skipped += 1
                            continue
                        if args.dry_run:
                            framed += 1
                            continue
                        desc = describe(png, args.vision_url, args.vision_model, args.prompt, args.sys_prompt, args.timeout)
                        store_frame(sess, key, v, minute, t_sec, desc, args.vision_model)
                        framed += 1
                        if args.sleep:
                            time.sleep(args.sleep)
                except Exception as e:
                    errors += 1
                    print(f"[error] {key}: {e}", flush=True)
            if args.limit_clips and processed >= args.limit_clips:
                break
    print(f"[done] clips_scanned={processed} frames_written={framed} skipped={skipped} errors={errors}", flush=True)
    driver.close()


if __name__ == "__main__":
    main()
