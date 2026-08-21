#!/usr/bin/env python3
"""Batch-describe dashcam clips via a vision LLM (default: MacBook Air Tailscale 100.85.64.117 LM Studio qwen3.5-0.8b-mlx).

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
import re
import argparse
import base64
import time
from datetime import datetime, timedelta
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
import subprocess
import tempfile

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


KEY_RE = re.compile(r"^(\d{4})_(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})")


def parse_key_datetime(key):
    """Clip keys look like 2026_0709_203536_R -> 2026-07-09 20:35:36."""
    m = KEY_RE.match(key)
    if not m:
        return None
    y, mo, da, h, mi, s = map(int, m.groups())
    try:
        return datetime(y, mo, da, h, mi, s)
    except Exception:
        return None


def view_of(key):
    k = key.upper()
    if k.endswith("_FR"):
        return "FR"
    if k.endswith("_F"):
        return "F"
    if k.endswith("_R"):
        return "R"
    return "?"


def probe_duration(mp4_path):
    try:
        out = subprocess.run(["ffprobe", "-v", "error",
                              "-show_entries", "format=duration", "-of", "csv=p=0",
                              mp4_path], timeout=20,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        v = out.stdout.decode().strip().split("\n")[0].strip()
        return float(v)
    except Exception:
        return 0.0


def extract_minute_first_frames(mp4_path, max_minutes=None, max_size=640, ff_timeout=30):
    """Decode frames with ffmpeg (robust on corrupt clips); cv2 only resizes/encodes."""
    dur = probe_duration(mp4_path)
    total_min = int(dur // 60)
    if max_minutes is not None:
        total_min = min(total_min, max_minutes)
    for m in range(total_min + 1):
        t = float(m * 60)
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            r = subprocess.run(["ffmpeg", "-y", "-ss", f"{t:.3f}", "-i", mp4_path,
                                "-frames:v", "1", "-q:v", "2", "-update", "1", path],
                               timeout=ff_timeout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if r.returncode != 0 or not os.path.getsize(path):
                continue
            img = cv2.imread(path)
        finally:
            if os.path.exists(path):
                os.unlink(path)
        if img is None:
            continue
        h, w = img.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        ok, buf = cv2.imencode(".png", img)
        if not ok:
            continue
        yield m, t, buf.tobytes()


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
        "max_tokens": 400,
        "temperature": 0.2,
        "enable_thinking": False,
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


def done_minutes(sess, key):
    recs = sess.run("MATCH (f:DashcamFrame {key:$key}) RETURN f.minute AS m", key=key)
    return set(r["m"] for r in recs if r["m"] is not None)


def store_frame(sess, key, view, minute, t_sec, desc, model, path=None):
    dt = parse_key_datetime(key)
    day_date = dt.strftime("%Y-%m-%d") if dt else None
    y = dt.year if dt else None
    mo = dt.month if dt else None
    d = dt.day if dt else None
    fdt = (dt + timedelta(minutes=minute)) if dt else None
    sess.run(
        """
        MERGE (c:DashcamClip {key:$key}) ON CREATE SET c.key=$key
        MERGE (f:DashcamFrame {key:$key, minute:$minute})
          ON CREATE SET f.description=$desc, f.t_sec=$t, f.view=$view,
                        f.model=$model, f.created=datetime()
          ON MATCH  SET f.description=$desc, f.model=$model, f.updated=datetime()
        MERGE (c)-[:HAS_FRAME]->(f)
        SET c.path=$path, c.timestamp=$dt, c.date=$date, f.timestamp=$fdt
        WITH c, $dt AS dt, $date AS date, $y AS y, $mo AS mo, $d AS d
        FOREACH (_ IN CASE WHEN dt IS NOT NULL THEN [1] ELSE [] END |
            MERGE (day:DashcamDay {date:date})
            SET day.year=y, day.month=mo, day.day=d
            MERGE (c)-[:ON_DAY]->(day))
        """,
        key=key, view=view, minute=minute, t=t_sec, desc=desc, model=model,
        path=path, dt=dt, date=day_date, y=y, mo=mo, d=d, fdt=fdt,
    )


def mark_failed(sess, key, reason="unknown", path=None):
    sess.run(
        "MERGE (c:DashcamClip {key:$key}) ON CREATE SET c.key=$key "
        "SET c.failed=true, c.failure_reason=$reason, c.failed_at=datetime()"
        + (" SET c.path=$path" if path else ""),
        key=key, reason=reason, path=path,
    )


def mark_processed(sess, key, path=None):
    sess.run(
        "MERGE (c:DashcamClip {key:$key}) ON CREATE SET c.key=$key "
        "SET c.processed=true, c.processed_at=datetime()"
        + (" SET c.path=$path" if path else ""),
        key=key, path=path,
    )


def run_single_clip(mp4, key, view, args):
    """Worker: process ONE clip in its own process (isolates ffmpeg/cv2 faults).
    Decode failures -> raise (orchestrator marks clip failed, permanent).
    Vision failures -> retried; if all fail, exit 0 (transient endpoint, retry later)."""
    driver = GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_password))
    with driver.session() as sess:
        sess.run("MERGE (c:DashcamClip {key:$key}) ON CREATE SET c.key=$key SET c.path=$path",
                 key=key, path=mp4)
        done = done_minutes(sess, key)
        frames = list(extract_minute_first_frames(mp4, args.max_minutes, args.max_size))
        if not frames:
            raise RuntimeError("no frames extracted (decode failure)")
        stored = 0
        for m, t_sec, png in frames:
            if m in done:
                continue
            desc = None
            for attempt in range(args.vision_retries):
                try:
                    desc = describe(png, args.vision_url, args.vision_model, args.prompt, args.sys_prompt, args.timeout)
                    break
                except Exception as e:
                    if attempt < args.vision_retries - 1:
                        time.sleep(5 * (attempt + 1))
                        continue
                    print(f"[vision-fail] {key} m={m}: {e}", flush=True)
            if desc is None:
                continue
            store_frame(sess, key, view, m, t_sec, desc, args.vision_model, path=mp4)
            stored += 1
            if args.sleep:
                time.sleep(args.sleep)
        if stored == 0:
            print(f"[no-store] {key}: transient vision failure, will retry later", flush=True)
    driver.close()


def run_orchestrate(args):
    import subprocess
    from concurrent.futures import ThreadPoolExecutor, as_completed
    driver = GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_password))
    clip_dirs = walk_clip_dirs(args.base)
    print(f"[info] found {len(clip_dirs)} clip directories under {args.base}", flush=True)

    jobs = []
    with driver.session() as sess:
        for d in clip_dirs:
            for key in list_clip_keys(d):
                v = view_of(key)
                if args.view != "both" and v != args.view:
                    continue
                mp4 = os.path.join(d, f"{key}.MP4")
                if not os.path.exists(mp4):
                    continue
                rec = sess.run("MATCH (c:DashcamClip {key:$key}) RETURN c.failed AS f", key=key).single()
                if rec and rec["f"]:
                    continue
                # cheap skip: clip already has >=1 frame (assume prior session completed it)
                have = sess.run("MATCH (f:DashcamFrame {key:$key}) RETURN count(f) AS c", key=key).single()["c"]
                if have > 0:
                    continue
                jobs.append((mp4, key, v))
                if args.limit_clips and len(jobs) >= args.limit_clips:
                    break
            if args.limit_clips and len(jobs) >= args.limit_clips:
                break
    print(f"[info] {len(jobs)} clips to process (already-done/failed skipped)", flush=True)

    def run_one(mp4, key, v):
        cmd = [sys.executable, __file__, "--single-clip", mp4, "--key", key, "--view", v,
               "--vision-url", args.vision_url, "--vision-model", args.vision_model,
               "--neo4j-uri", args.neo4j_uri, "--neo4j-user", args.neo4j_user,
               "--neo4j-password", args.neo4j_password,
               "--max-size", str(args.max_size), "--sleep", str(args.sleep),
               "--timeout", str(args.timeout), "--vision-retries", str(args.vision_retries),
               "--prompt", args.prompt, "--sys-prompt", args.sys_prompt]
        if args.max_minutes is not None:
            cmd += ["--max-minutes", str(args.max_minutes)]
        try:
            subprocess.run(cmd, timeout=args.clip_timeout, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return (key, mp4, "ok", "")
        except subprocess.TimeoutExpired:
            return (key, mp4, "timeout", "")
        except subprocess.CalledProcessError:
            return (key, mp4, "crash", "")

    workers = max(1, args.workers)
    if workers == 1:
        results = [run_one(*j) for j in jobs]
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(run_one, *j) for j in jobs]
            results = [f.result() for f in as_completed(futs)]

    processed = failed = 0
    with driver.session() as sess:
        for key, mp4, status, err in results:
            if status == "ok":
                mark_processed(sess, key, mp4)
                processed += 1
            elif status == "timeout":
                print(f"[timeout] {key}", flush=True)
                mark_failed(sess, key, "timeout_or_hang", mp4)
                failed += 1
            else:
                print(f"[crash] {key}", flush=True)
                mark_failed(sess, key, "decode_failed", mp4)
                failed += 1
    print(f"[done] processed={processed} failed={failed}", flush=True)
    driver.close()


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
    ap.add_argument("--vision-retries", type=int, default=4, help="retries per frame on vision endpoint failure")
    ap.add_argument("--single-clip", default=None, help="(worker) process only this MP4 path")
    ap.add_argument("--key", default=None, help="(worker) clip key")
    ap.add_argument("--clip-timeout", type=int, default=600, help="orchestrate: max seconds per clip subprocess")
    ap.add_argument("--workers", type=int, default=1, help="orchestrate: parallel clip subprocesses")
    ap.add_argument("--extract-timeout", type=int, default=45, help="hard timeout (s) for cv2 frame extraction per clip")
    ap.add_argument("--orchestrate", action="store_true", help="supervisor: isolate each clip in a subprocess")
    args = ap.parse_args()

    if args.single_clip:
        if not args.key:
            print("--key required with --single-clip", flush=True)
            sys.exit(2)
        run_single_clip(args.single_clip, args.key, args.view, args)
        return

    run_orchestrate(args)


if __name__ == "__main__":
    main()
