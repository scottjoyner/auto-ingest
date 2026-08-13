"""Plan-bound runner for named orchestration profiles.

This is the preferred entrypoint for reusable named profiles. It binds the
current semantic task graph to the persisted job before acquiring a lease, so a
retry cannot silently mix completed work from one profile generation with a
changed command graph. The historical ``full`` alias resolves to the canonical
stage-level graph rather than the legacy monolithic run-all task.
"""
from __future__ import annotations

import argparse
import socket
from typing import Sequence

from auto_ingest.orchestration import PROFILES, default_job_key, ensure_job, run_profile
from auto_ingest.pipeline_contract import bind_plan


def profile_tasks(profile: str):
    if profile == "full":
        from auto_ingest.full_pipeline import build_tasks

        return build_tasks()
    return PROFILES[profile]


def run_bound_profile(
    driver,
    profile: str,
    *,
    job_key: str | None = None,
    owner: str | None = None,
    ttl_sec: int = 300,
    heartbeat_sec: int = 30,
    max_attempts: int = 3,
) -> int:
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}")
    key = job_key or default_job_key(profile)
    tasks = profile_tasks(profile)
    ensure_job(driver, key, profile)
    bind_plan(driver, key, tasks)
    return run_profile(
        driver,
        profile,
        job_key=key,
        owner=owner,
        ttl_sec=ttl_sec,
        heartbeat_sec=heartbeat_sec,
        max_attempts=max_attempts,
        tasks=tasks,
    )


def _driver():
    from neo4j import GraphDatabase

    from auto_ingest_config import get_neo4j_config

    cfg = get_neo4j_config()
    return GraphDatabase.driver(cfg["uri"], auth=(cfg["user"], cfg["password"]))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m auto_ingest.profile_runner")
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--job-key", default=None)
    parser.add_argument("--owner", default=None)
    parser.add_argument("--ttl-sec", type=int, default=300)
    parser.add_argument("--heartbeat-sec", type=int, default=30)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args(argv)
    driver = _driver()
    try:
        return run_bound_profile(
            driver,
            args.profile,
            job_key=args.job_key,
            owner=args.owner or f"profile:{args.profile}:{socket.gethostname()}",
            ttl_sec=args.ttl_sec,
            heartbeat_sec=args.heartbeat_sec,
            max_attempts=args.max_attempts,
        )
    finally:
        driver.close()


if __name__ == "__main__":
    raise SystemExit(main())
