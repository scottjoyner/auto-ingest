#!/usr/bin/env python3
"""Wait for all three platform secrets, then sign in to each sequentially.

Usage:
    .venv/bin/python scripts/watch_all_auth.py

Polls until the YouTube, TikTok, and Instagram client-secret files all exist,
then runs the sign-in flow for each (browser-based, local-server callback).
Run in a terminal you can interact with. Secrets expected at:
    ~/.config/auto-ingest/{yt,tiktok,ig}_client_secret.json
(or set via YT_CLIENT_SECRET_JSON / TIKTOK_CLIENT_SECRET_JSON / IG_CLIENT_SECRET_JSON).
"""
from __future__ import annotations
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CFG = Path.home() / ".config" / "auto-ingest"
SECRETS = {
    "youtube": Path(os.environ.get("YT_CLIENT_SECRET_JSON", str(CFG / "yt_client_secret.json"))),
    "tiktok": Path(os.environ.get("TIKTOK_CLIENT_SECRET_JSON", str(CFG / "tiktok_client_secret.json"))),
    "instagram": Path(os.environ.get("IG_CLIENT_SECRET_JSON", str(CFG / "ig_client_secret.json"))),
}


def _present() -> list:
    return [k for k, p in SECRETS.items() if p.exists()]


def main() -> int:
    print("Watching for platform client secrets:")
    for k, p in SECRETS.items():
        print(f"  {k:10} -> {p}  [{'OK' if p.exists() else 'waiting'}]")
    while len(_present()) < len(SECRETS):
        time.sleep(10)
        got = _present()
        if got:
            print(f"  found: {', '.join(got)}  (still waiting on the rest)")
    print("\nAll secrets present. Starting sign-in flows...\n")

    from auto_ingest.shorts.cli import main as cli_main
    for plat in ("youtube", "tiktok", "instagram"):
        print(f"\n===== {plat} sign-in =====")
        rc = cli_main(["publish", "auth", plat])
        if rc not in (0, None):
            print(f"  WARNING: {plat} auth returned {rc}")
    print("\nAll platform authentications complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
