"""Fenced compatibility entrypoint for transcript ingestion.

Historical callers may still invoke ``run_ingest_all.sh``. That wrapper now
lands here, where the transcript stage is supervised by the same persisted
lease/heartbeat/retry/quarantine controller as scheduled production work.
"""
from __future__ import annotations

import os
import socket
import sys
from typing import Sequence

from auto_ingest.orchestration import Task, run_profile


def _driver():
    from neo4j import GraphDatabase

    from auto_ingest_config import get_neo4j_config

    cfg = get_neo4j_config()
    return GraphDatabase.driver(cfg["uri"], auth=(cfg["user"], cfg["password"]))


def main(argv: Sequence[str] | None = None) -> int:
    if argv:
        raise ValueError("transcript orchestration is configured through environment variables")
    driver = _driver()
    try:
        return run_profile(
            driver,
            "transcripts",
            job_key=os.environ.get("INGEST_JOB_KEY") or None,
            owner=os.environ.get("INGEST_OWNER")
            or f"transcripts:{socket.gethostname()}:{os.getpid()}",
            ttl_sec=int(os.environ.get("INGEST_LEASE_TTL_SEC", "300")),
            heartbeat_sec=int(os.environ.get("INGEST_HEARTBEAT_SEC", "30")),
            max_attempts=int(os.environ.get("INGEST_MAX_ATTEMPTS", "3")),
            tasks=(
                Task(
                    "transcript-ingest",
                    (sys.executable, "-m", "auto_ingest.transcript_entrypoint"),
                    int(os.environ.get("INGEST_TIMEOUT_SEC", "21600")),
                ),
            ),
        )
    finally:
        driver.close()


if __name__ == "__main__":
    raise SystemExit(main())
