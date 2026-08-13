"""Canonical stage-level full ingest pipeline.

Each heavy stage is a persisted orchestration task. Successful stages are not
replayed after a later failure; retries resume at the first incomplete task.
The persisted plan fingerprint prevents mixed-generation resumes when commands
or stage settings change after partial progress.
"""
from __future__ import annotations

import os
import socket
import sys
import time
from pathlib import Path
from typing import Sequence

from auto_ingest.orchestration import Task, ensure_job, run_profile
from auto_ingest.pipeline_contract import bind_plan

REPO = Path(__file__).resolve().parent.parent


def state_root() -> Path:
    root = Path(os.environ.get("STATE_ROOT", str(REPO / ".state")))
    root.mkdir(parents=True, exist_ok=True)
    return root


def build_tasks() -> tuple[Task, ...]:
    state = state_root()
    tasks: list[Task] = [
        Task("copy-audio", ("bash", "audio_copy.sh"), 7200),
        Task("copy-dashcam", ("bash", "dashcam_copy.sh"), 14400),
        Task("copy-bodycam", ("bash", "bodycam_copy.sh"), 7200),
        Task("diarize", (sys.executable, "-m", "speakers"), 21600),
        Task(
            "transcript-ingest",
            (sys.executable, "-m", "auto_ingest.transcript_entrypoint"),
            21600,
        ),
        Task(
            "speaker-reconcile",
            (
                sys.executable,
                "-m",
                "speakers_reconcile",
                "--batch",
                os.environ.get("RECONCILE_BATCH", "50"),
                "--only-missing",
                "--allow-discovery",
            ),
            14400,
        ),
        Task(
            "music-segments",
            (sys.executable, "-m", "01_precompute_music_segments", "--push-neo4j"),
            14400,
        ),
        Task(
            "lyrics-classification",
            (
                sys.executable,
                "-m",
                "02_classify_lyrics",
                "--segments-source",
                "neo4j",
                "--limit",
                os.environ.get("LYRICS_LIMIT", "50000"),
            ),
            14400,
        ),
        Task(
            "speaker-link",
            (
                "bin/auto-ingest",
                "link-speakers",
                "--state-file",
                str(state / "full_linker_state.json"),
            ),
            21600,
        ),
        Task(
            "yolo-embeddings",
            (sys.executable, "-m", "auto_ingest.yolo_entrypoint"),
            21600,
        ),
    ]
    return tuple(tasks)


def default_job_key(now: float | None = None) -> str:
    ts = int(time.time() if now is None else now)
    window = int(os.environ.get("FULL_PIPELINE_WINDOW_SEC", "1800"))
    if window < 60:
        raise ValueError("FULL_PIPELINE_WINDOW_SEC must be at least 60")
    return f"pipeline:full:{ts // window}"


def _driver():
    from neo4j import GraphDatabase

    from auto_ingest_config import get_neo4j_config

    cfg = get_neo4j_config()
    return GraphDatabase.driver(cfg["uri"], auth=(cfg["user"], cfg["password"]))


def main(argv: Sequence[str] | None = None) -> int:
    if argv:
        raise ValueError("full pipeline configuration is environment-driven")
    driver = _driver()
    try:
        key = os.environ.get("FULL_INGEST_JOB_KEY") or default_job_key()
        tasks = build_tasks()
        ensure_job(driver, key, "full-stages")
        bind_plan(driver, key, tasks)
        return run_profile(
            driver,
            "full-stages",
            job_key=key,
            owner=os.environ.get("FULL_INGEST_OWNER")
            or f"full:{socket.gethostname()}:{os.getpid()}",
            ttl_sec=int(os.environ.get("FULL_INGEST_LEASE_TTL_SEC", "300")),
            heartbeat_sec=int(os.environ.get("FULL_INGEST_HEARTBEAT_SEC", "30")),
            max_attempts=int(os.environ.get("FULL_INGEST_MAX_ATTEMPTS", "3")),
            tasks=tasks,
        )
    finally:
        driver.close()


if __name__ == "__main__":
    raise SystemExit(main())
