#!/usr/bin/env python3
"""
watch_embed.py — incremental embed watcher.

Keeps every text label's vector property (default emb_e5_large) fresh by re-running
reembed.py --resume on a loop. Because reembed only fills nodes still missing the
vector, sweeps are cheap once a label is caught up. Point it at a KG via NEO4J_URI
(and open any required SSH tunnel in the wrapper). Run on a GPU box.

Single-KG per invocation; the wrapper alternates KGs (e.g. bodycam via tunnel,
research locally). Use --once for one sweep (then exit) inside a wrapper loop.
"""
from __future__ import annotations
import os
import sys
import time
import subprocess
import argparse
from pathlib import Path

# Default label set covering both KGs (the wrapper selects the relevant subset).
DEFAULT_LABELS = [
    "Segment", "Utterance", "Transcription", "Summary",
    "Entity", "Concept", "Topic", "Keyword", "KgNode", "Note",
    "Speaker", "GlobalSpeaker", "Chunk",
]

REEMBED = Path(__file__).resolve().parent / "reembed.py"
VENV_PY = os.getenv("WATCH_EMBED_PY", sys.executable)


def main() -> int:
    ap = argparse.ArgumentParser(description="Incremental embed watcher (reembed.py --resume loop).")
    ap.add_argument("--interval", type=int, default=900, help="Seconds between sweeps (ignored with --once).")
    ap.add_argument("--model", default="intfloat/multilingual-e5-large")
    ap.add_argument("--prop", default="emb_e5_large")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--labels", nargs="*", default=DEFAULT_LABELS)
    ap.add_argument("--once", action="store_true", help="Run one sweep then exit (for wrapper loops).")
    args = ap.parse_args()

    if not REEMBED.exists():
        print(f"[watch_embed] ERROR: reembed.py not found at {REEMBED}", file=sys.stderr)
        return 2

    while True:
        t0 = time.time()
        for lbl in args.labels:
            cmd = [
                VENV_PY, str(REEMBED), lbl,
                "--model", args.model, "--prop", args.prop,
                "--batch-size", str(args.batch_size), "--resume",
            ]
            print(f"[watch_embed] {time.strftime('%H:%M:%S')} sweep {lbl}", flush=True)
            subprocess.run(cmd, check=False)
        dt = int(time.time() - t0)
        if args.once:
            print(f"[watch_embed] one sweep done in {dt}s; exiting (--once)", flush=True)
            return 0
        print(f"[watch_embed] sweep done in {dt}s; sleeping {args.interval}s", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
