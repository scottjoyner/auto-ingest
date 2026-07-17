#!/usr/bin/env python3
"""Shim: delegates to auto_ingest.ingest.transcripts (post W-42 restructure)."""
from auto_ingest.ingest.transcripts import *  # noqa: F401,F403

if __name__ == "__main__":
    from auto_ingest.ingest.transcripts import main
    main()
