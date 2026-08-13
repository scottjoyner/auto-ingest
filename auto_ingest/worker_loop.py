"""Resource-aware background worker using the canonical fenced orchestrator."""
from __future__ import annotations

import argparse
import os
import socket
import sys
import time
from pathlib import Path
from typing import Sequence

from auto_ingest.orchestration import Task, run_profile
from auto_ingest.resources import ResourcePolicy, admission, snapshot

REPO = Path(__file__).resolve().parent.parent


def build_worker_tasks() -> tuple[Task, ...]:
    link_chunk = os.environ.get("LINK_CHUNK", "200")
    drop_root = os.environ.get("DROP_ROOT", "/nas/drop")
    tasks: list[Task] = [
        # Safe local/NAS fallback queue. This only accepts typed *.job.json
        # profiles and explicitly rejects legacy executable *.job shell files.
        Task(
            "fallback-queue",
            (
                sys.executable,
                "-m",
                "auto_ingest.file_queue",
                "work",
                "--once",
                "--root",
                drop_root,
            ),
            3600,
        ),
        Task(
            "speaker-link",
            (
                str(REPO / "bin" / "auto-ingest"),
                "link-speakers",
                "--faiss",
                "--state-file",
                str(REPO / "linker_state.json"),
                "--max-speakers",
                link_chunk,
            ),
            7200,
        ),
        Task("dashcam-compress", ("bash", str(REPO / "run_compress_dashcam.sh")), 14400),
    ]
    if os.environ.get("CONTENT", "1") != "0":
        tasks.append(
            Task(
                "content",
                (
                    sys.executable,
                    str(REPO / "worker_content.py"),
                    "--state",
                    str(REPO / "content_state.json"),
                    "--limit",
                    os.environ.get("CONTENT_LIMIT", "5"),
                ),
                7200,
            )
        )

    # The media ingester itself resolves Nextcloud URL/user/token from config.yaml
    # when no explicit URL is passed, keeping credentials out of argv/process lists.
    try:
        from auto_ingest_config import get_nextcloud_webdav

        url, _user, _password = get_nextcloud_webdav()
    except Exception:
        url = None
    if url:
        tasks.append(
            Task(
                "nextcloud-media",
                (
                    str(REPO / "bin" / "auto-ingest"),
                    "ingest",
                    "--source",
                    "nextcloud",
                    "--kind",
                    "all",
                    "--slideshow",
                    "--limit",
                    os.environ.get("NC_LIMIT", "40"),
                    "--state",
                    str(REPO / "nc_ingest_state.json"),
                ),
                14400,
            )
        )
    return tuple(tasks)


def _driver():
    from neo4j import GraphDatabase

    from auto_ingest_config import get_neo4j_config

    cfg = get_neo4j_config()
    return GraphDatabase.driver(cfg["uri"], auth=(cfg["user"], cfg["password"]))


def run_cycle(
    driver,
    *,
    policy: ResourcePolicy,
    resource_path: str | Path,
    now: float | None = None,
) -> int:
    snap = snapshot(resource_path)
    allowed, reasons = admission(snap, policy)
    if not allowed:
        print("[worker] admission denied: " + "; ".join(reasons), flush=True)
        return 4
    ts = time.time() if now is None else now
    window = max(60, int(os.environ.get("WORKER_WINDOW_SEC", "300")))
    key = f"worker-cycle:{socket.gethostname()}:{int(ts) // window}"
    return run_profile(
        driver,
        "worker",
        job_key=key,
        owner=f"worker:{socket.gethostname()}:{os.getpid()}",
        tasks=build_worker_tasks(),
        ttl_sec=int(os.environ.get("WORKER_LEASE_TTL_SEC", "300")),
        heartbeat_sec=int(os.environ.get("WORKER_HEARTBEAT_SEC", "30")),
        max_attempts=int(os.environ.get("WORKER_MAX_ATTEMPTS", "3")),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m auto_ingest.worker_loop")
    parser.add_argument("--once", action="store_true", help="Run at most one admitted cycle.")
    parser.add_argument("--sleep-sec", type=int, default=int(os.environ.get("SLEEP", "120")))
    parser.add_argument(
        "--max-load-per-cpu",
        type=float,
        default=float(os.environ.get("MAX_LOAD_PER_CPU", "0.60")),
    )
    parser.add_argument(
        "--min-memory-mb",
        type=int,
        default=int(os.environ.get("MIN_MEMORY_AVAILABLE_MB", "2048")),
    )
    parser.add_argument(
        "--min-disk-gb",
        type=float,
        default=float(os.environ.get("MIN_DISK_FREE_GB", "20")),
    )
    parser.add_argument(
        "--resource-path",
        default=os.environ.get("HOT_STORAGE_ROOT", str(REPO)),
    )
    args = parser.parse_args(argv)
    if args.sleep_sec < 1:
        parser.error("--sleep-sec must be positive")

    policy = ResourcePolicy(
        max_load_per_cpu=args.max_load_per_cpu,
        min_memory_available_mb=args.min_memory_mb,
        min_disk_free_gb=args.min_disk_gb,
    )
    stop = REPO / "worker.stop"
    driver = _driver()
    try:
        while not stop.exists():
            rc = run_cycle(driver, policy=policy, resource_path=args.resource_path)
            if args.once:
                return rc
            time.sleep(args.sleep_sec)
        return 0
    finally:
        driver.close()
        if stop.exists():
            stop.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
