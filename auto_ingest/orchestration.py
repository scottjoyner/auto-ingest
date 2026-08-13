"""Canonical persisted pipeline orchestration for scheduled/worker execution.

This module is the scheduler boundary: cron, service loops, and workers should
submit work here rather than launching pipeline scripts directly. It provides:
- fenced lease ownership + heartbeats
- explicit lifecycle state persisted in Neo4j
- resumable completed task manifest
- structured primitive-only failure envelopes
- retry/quarantine policy
- bounded subprocess execution with periodic lease renewal
"""
from __future__ import annotations

import argparse
import hashlib
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from auto_ingest import ingest_claim

REPO = Path(__file__).resolve().parent.parent

DISCOVERED = "DISCOVERED"
READY = "READY"
CLAIMED = "CLAIMED"
RUNNING = "RUNNING"
VALIDATING = "VALIDATING"
DONE = "DONE"
RETRY = "RETRY"
FAILED = "FAILED"
QUARANTINED = "QUARANTINED"

TERMINAL = {DONE, QUARANTINED}


@dataclass(frozen=True)
class Task:
    name: str
    command: tuple[str, ...]
    timeout_sec: int = 7200


PROFILES: dict[str, tuple[Task, ...]] = {
    "full": (
        Task("full-ingest", (str(REPO / "bin" / "auto-ingest"), "run-all"), 21600),
    ),
    "dashcam": (
        Task("dashcam-batch", (sys.executable, str(REPO / "bulk_ingest_dashcam.py"), "--years", "2026"), 21600),
    ),
    "sync": (
        Task("legacy-sync", ("bash", str(REPO / "deploy" / "sync_from_legacy_drop.sh")), 1800),
    ),
}


def _fingerprint(exc_type: str, message: str, task: str) -> str:
    raw = f"{exc_type}\0{task}\0{message}".encode("utf-8", errors="replace")
    return hashlib.sha256(raw).hexdigest()[:24]


def ensure_job(driver, key: str, profile: str) -> None:
    now = int(time.time() * 1000)
    with driver.session() as session:
        session.run(
            """
            MERGE (j:IngestJob {key:$key})
            ON CREATE SET j.created_at=$now,
                          j.owner='', j.claimed_at=0,
                          j.fence_token=0,
                          j.attempt_count=0,
                          j.completed_stages=[],
                          j.completed_tasks=[],
                          j.lifecycle_state='DISCOVERED'
            SET j.profile=$profile,
                j.updated_at=$now,
                j.lifecycle_state=CASE
                    WHEN j.lifecycle_state IS NULL THEN 'DISCOVERED'
                    ELSE j.lifecycle_state END
            """,
            key=key,
            profile=profile,
            now=now,
        ).consume()


def lifecycle(driver, key: str) -> dict | None:
    with driver.session() as session:
        rec = session.run(
            """
            MATCH (j:IngestJob {key:$key})
            RETURN j.lifecycle_state AS state,
                   coalesce(j.completed_tasks,[]) AS completed_tasks,
                   coalesce(j.attempt_count,0) AS attempts,
                   coalesce(j.fence_token,0) AS fence_token,
                   coalesce(j.owner,'') AS owner,
                   j.current_task AS current_task,
                   j.error_type AS error_type,
                   j.error_message AS error_message,
                   j.error_fingerprint AS error_fingerprint
            LIMIT 1
            """,
            key=key,
        ).single()
    return dict(rec) if rec else None


def transition_fenced(
    driver,
    key: str,
    *,
    owner: str,
    fence_token: int,
    state: str,
    current_task: str | None = None,
) -> bool:
    with driver.session() as session:
        rec = session.run(
            """
            MATCH (j:IngestJob {key:$key})
            WHERE j.owner=$owner AND coalesce(j.fence_token,0)=$fence_token
            SET j.lifecycle_state=$state,
                j.current_task=$current_task,
                j.updated_at=timestamp()
            RETURN j.key AS key
            """,
            key=key,
            owner=owner,
            fence_token=fence_token,
            state=state,
            current_task=current_task,
        ).single()
    return rec is not None


def heartbeat(driver, key: str, *, owner: str, fence_token: int) -> bool:
    now = int(time.time() * 1000)
    with driver.session() as session:
        rec = session.run(
            """
            MATCH (j:IngestJob {key:$key})
            WHERE j.owner=$owner AND coalesce(j.fence_token,0)=$fence_token
            SET j.claimed_at=$now, j.heartbeat_at=$now, j.updated_at=$now
            RETURN j.key AS key
            """,
            key=key,
            owner=owner,
            fence_token=fence_token,
            now=now,
        ).single()
    return rec is not None


def mark_task_complete(driver, key: str, task: str, *, owner: str, fence_token: int) -> bool:
    with driver.session() as session:
        rec = session.run(
            """
            MATCH (j:IngestJob {key:$key})
            WHERE j.owner=$owner AND coalesce(j.fence_token,0)=$fence_token
            SET j.completed_tasks=CASE
                    WHEN $task IN coalesce(j.completed_tasks,[]) THEN coalesce(j.completed_tasks,[])
                    ELSE coalesce(j.completed_tasks,[]) + $task END,
                j.updated_at=timestamp()
            RETURN j.key AS key
            """,
            key=key,
            task=task,
            owner=owner,
            fence_token=fence_token,
        ).single()
    return rec is not None


def record_failure(
    driver,
    key: str,
    *,
    owner: str,
    fence_token: int,
    task: str,
    exc_type: str,
    message: str,
    max_attempts: int,
) -> str:
    fp = _fingerprint(exc_type, message, task)
    with driver.session() as session:
        rec = session.run(
            """
            MATCH (j:IngestJob {key:$key})
            WHERE j.owner=$owner AND coalesce(j.fence_token,0)=$fence_token
            WITH j, coalesce(j.attempt_count,0) AS attempts
            SET j.lifecycle_state=CASE WHEN attempts >= $max_attempts THEN 'QUARANTINED' ELSE 'RETRY' END,
                j.error_type=$exc_type,
                j.error_message=left($message,2000),
                j.error_fingerprint=$fingerprint,
                j.failed_task=$task,
                j.failed_at=timestamp(),
                j.updated_at=timestamp()
            RETURN j.lifecycle_state AS state
            """,
            key=key,
            owner=owner,
            fence_token=fence_token,
            exc_type=exc_type,
            message=message,
            fingerprint=fp,
            task=task,
            max_attempts=max_attempts,
        ).single()
    return rec.get("state") if rec else FAILED


def _run_task(
    task: Task,
    *,
    driver,
    key: str,
    owner: str,
    fence_token: int,
    env: dict[str, str],
    heartbeat_sec: int,
    popen_factory: Callable[..., subprocess.Popen] = subprocess.Popen,
) -> int:
    proc = popen_factory(task.command, cwd=str(REPO), env=env)
    started = time.monotonic()
    next_heartbeat = started
    while True:
        rc = proc.poll()
        if rc is not None:
            return int(rc)
        now = time.monotonic()
        if now >= next_heartbeat:
            if not heartbeat(driver, key, owner=owner, fence_token=fence_token):
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                raise ingest_claim.LeaseLostError("lease lost while subprocess was running")
            next_heartbeat = now + heartbeat_sec
        if now - started >= task.timeout_sec:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            raise TimeoutError(f"task {task.name} exceeded {task.timeout_sec}s")
        time.sleep(min(1.0, max(0.05, heartbeat_sec / 10)))


def run_profile(
    driver,
    profile: str,
    *,
    job_key: str | None = None,
    owner: str | None = None,
    ttl_sec: int = 300,
    heartbeat_sec: int = 30,
    max_attempts: int = 3,
    env: dict[str, str] | None = None,
    tasks: Sequence[Task] | None = None,
    popen_factory: Callable[..., subprocess.Popen] = subprocess.Popen,
) -> int:
    if profile not in PROFILES and tasks is None:
        raise ValueError(f"unknown profile {profile!r}")
    key = job_key or f"pipeline:{profile}"
    worker = owner or f"{socket.gethostname()}:{os.getpid()}"
    ensure_job(driver, key, profile)
    existing = lifecycle(driver, key) or {}
    if existing.get("state") == QUARANTINED:
        return 3

    token = ingest_claim.claim_fenced(driver, key, worker, ttl_sec=ttl_sec)
    if token is None:
        return 2
    transition_fenced(driver, key, owner=worker, fence_token=token, state=CLAIMED)
    completed = set((lifecycle(driver, key) or {}).get("completed_tasks", []))
    selected = tuple(tasks if tasks is not None else PROFILES[profile])
    run_env = dict(os.environ if env is None else env)

    try:
        transition_fenced(driver, key, owner=worker, fence_token=token, state=READY)
        for task in selected:
            if task.name in completed:
                continue
            if not transition_fenced(
                driver, key, owner=worker, fence_token=token,
                state=RUNNING, current_task=task.name,
            ):
                raise ingest_claim.LeaseLostError("lease lost before task start")
            try:
                rc = _run_task(
                    task,
                    driver=driver,
                    key=key,
                    owner=worker,
                    fence_token=token,
                    env=run_env,
                    heartbeat_sec=heartbeat_sec,
                    popen_factory=popen_factory,
                )
                if rc != 0:
                    raise RuntimeError(f"task {task.name} exited with code {rc}")
                if not transition_fenced(
                    driver, key, owner=worker, fence_token=token,
                    state=VALIDATING, current_task=task.name,
                ):
                    raise ingest_claim.LeaseLostError("lease lost before validation")
                if not mark_task_complete(
                    driver, key, task.name, owner=worker, fence_token=token
                ):
                    raise ingest_claim.LeaseLostError("lease lost before task commit")
            except Exception as exc:
                record_failure(
                    driver,
                    key,
                    owner=worker,
                    fence_token=token,
                    task=task.name,
                    exc_type=type(exc).__name__,
                    message=str(exc),
                    max_attempts=max_attempts,
                )
                return 1

        if not transition_fenced(
            driver, key, owner=worker, fence_token=token,
            state=DONE, current_task=None,
        ):
            raise ingest_claim.LeaseLostError("lease lost before DONE")
        return 0
    finally:
        ingest_claim.release_fenced(driver, key, worker, token)


def _driver_from_env():
    from neo4j import GraphDatabase
    from auto_ingest_config import get_neo4j_config

    cfg = get_neo4j_config()
    return GraphDatabase.driver(cfg["uri"], auth=(cfg["user"], cfg["password"]))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m auto_ingest.orchestration")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--profile", choices=sorted(PROFILES), required=True)
    run.add_argument("--job-key", default=None)
    run.add_argument("--owner", default=None)
    run.add_argument("--ttl-sec", type=int, default=300)
    run.add_argument("--heartbeat-sec", type=int, default=30)
    run.add_argument("--max-attempts", type=int, default=3)
    status = sub.add_parser("status")
    status.add_argument("--job-key", required=True)
    args = parser.parse_args(argv)

    driver = _driver_from_env()
    try:
        if args.command == "run":
            return run_profile(
                driver,
                args.profile,
                job_key=args.job_key,
                owner=args.owner,
                ttl_sec=args.ttl_sec,
                heartbeat_sec=args.heartbeat_sec,
                max_attempts=args.max_attempts,
            )
        print(lifecycle(driver, args.job_key))
        return 0
    finally:
        driver.close()


if __name__ == "__main__":
    raise SystemExit(main())
