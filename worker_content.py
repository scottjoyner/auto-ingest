#!/usr/bin/env python3
"""
worker_content.py — iteratively turn compressed dashcam clips into TikTok shorts.

Intended to run on idle machines (driven by run_worker.sh). It scans the
compressed dashcam tree for clips that haven't been turned into a short yet,
picks the newest few, generates a vertical short via tiktok_shorts.py (auto
captions from Neo4j), and records them in a state file so each clip is processed
once. Resumable and safe to run repeatedly.

Usage:
  python3 worker_content.py --state content_state.json --limit 5
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Set

from auto_ingest_config import get_fileserver_path

COMPRESSED_ROOT = os.environ.get("COMPRESSED_ROOT") or get_fileserver_path("dashcam/compressed")
OUT_ROOT = os.environ.get("TIKTOK_OUT_ROOT", "/media/scott/SSD_4TB/tiktok_shorts")
MUSIC = os.environ.get("CONTENT_MUSIC", "")  # optional background track


def _already_done(state: dict, path: str) -> bool:
    return path in state.get("done", {})


def scan_candidates(root: str, done: Set[str]) -> List[Path]:
    cands = []
    rp = Path(root)
    if not rp.exists():
        return cands
    for p in rp.rglob("*_FR.mp4"):
        if not p.is_file():
            continue
        if str(p) in done:
            continue
        cands.append(p)
    cands.sort(key=lambda q: q.stat().st_mtime, reverse=True)
    return cands


def main() -> int:
    ap = argparse.ArgumentParser(description="Iteratively generate TikTok shorts from compressed clips.")
    ap.add_argument("--state", default="./content_state.json", help="Resume state file")
    ap.add_argument("--limit", type=int, default=5, help="Max shorts this run")
    ap.add_argument("--compressed-root", default=COMPRESSED_ROOT)
    ap.add_argument("--out-root", default=OUT_ROOT)
    ap.add_argument("--music", default=MUSIC or None)
    ap.add_argument("--hook", default=None, help="Hook text (or auto from date)")
    args = ap.parse_args()

    state: dict = {}
    if Path(args.state).exists():
        try:
            state = json.loads(Path(args.state).read_text())
        except Exception:
            state = {}
    state.setdefault("done", {})
    done: Set[str] = set(state["done"].keys())

    cands = scan_candidates(args.compressed_root, done)
    if not cands:
        print("No new compressed clips to process.")
        return 0

    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    n = 0
    for clip in cands[: args.limit]:
        # Filename is YYYY_MMDD_HHMMSS(_FR); year + 4-digit MMDD are parts[0..1].
        parts = [p for p in clip.stem.split("_") if p]
        if (len(parts) >= 2 and re.fullmatch(r"\d{4}", parts[0])
                and re.fullmatch(r"\d{4}", parts[1])):
            ymd = parts[0] + parts[1]
            nice = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}"
        else:
            nice = "undated"
        out_dir = Path(args.out_root) / nice
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{clip.stem}_tok.mp4"
        hook = args.hook or f"Dashcam diaries — {nice}"
        print(f"+ generating short: {clip.name} -> {out}")
        cmd = [sys.executable, "tiktok_shorts.py", "--clip", str(clip),
               "--out", str(out), "--hook", hook]
        if args.music and Path(args.music).exists():
            cmd += ["--music", args.music]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                print("  FAILED:", r.stderr.splitlines()[-5:])
                state["done"][str(clip)] = {"ok": False, "at": datetime.now().isoformat()}
            else:
                state["done"][str(clip)] = {"ok": True, "at": datetime.now().isoformat()}
                n += 1
        except Exception as ex:
            print(f"  error: {ex}")
            state["done"][str(clip)] = {"ok": False, "at": datetime.now().isoformat()}
        Path(args.state).write_text(json.dumps(state, indent=2))

    print(f"Generated {n} short(s). State: {args.state}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
