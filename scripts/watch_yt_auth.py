#!/usr/bin/env python3
"""Wait for a YouTube OAuth client_secret.json, then run the sign-in flow.

Usage:
    .venv/bin/python scripts/watch_yt_auth.py

Polls until $YT_CLIENT_SECRET_JSON (or ~/.config/auto-ingest/yt_client_secret.json)
exists, then runs `publish auth youtube` (local-server browser sign-in) and
exits. Run this in a terminal you can interact with (it opens a browser).
"""
from __future__ import annotations
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SECRET = Path(os.environ.get(
    "YT_CLIENT_SECRET_JSON",
    str(Path.home() / ".config" / "auto-ingest" / "yt_client_secret.json")))


def main() -> int:
    print(f"Waiting for YouTube client secret at: {SECRET}")
    while not SECRET.exists():
        time.sleep(10)
    print("Secret found. Launching YouTube sign-in flow...")
    from auto_ingest.shorts.cli import main as cli_main
    return cli_main(["publish", "auth", "youtube"])


if __name__ == "__main__":
    raise SystemExit(main())
