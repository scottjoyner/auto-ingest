"""Bounded recovery watchdog for abandoned ingest orchestration leases.

The watchdog is deliberately conservative: it only mutates nonterminal jobs in
CLAIMED/RUNNING/VALIDATING whose heartbeat/claim timestamp is older than the
configured threshold. Candidate limiting happens before mutation. The fencing
token is never decremented; the next claimant receives a newer generation, so
a stale process cannot commit after watchdog recovery.
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Sequence

ACTIVE_STATES = ("CLAIMED", "RUNNING", "VALIDATING")


def stale_jobs(driver, *, stale_after_sec: int = 600, limit: int = 100,
               now_ms: int | None = None) -> list[dict]:
    if stale_after_sec < 1 or limit < 1:
        raise ValueError("stale_after_sec and limit must be positive")
    now = int(time.time() * 1000) if now_ms is None else int(now_ms)
    cutoff = now - stale_after_sec * 1000
    with driver.session() as session:
        rows = session.run(
            """
            MATCH (j:IngestJob)
            WHERE j.lifecycle_state IN ['CLAIMED','RUNNING','VALIDATING']
              AND coalesce(j.owner,'') <> ''
              AND coalesce(j.heartbeat_at,j.claimed_at,0) < $cutoff
            RETURN j.key AS key,
                   j.owner AS owner,
                   j.lifecycle_state AS state,
                   coalesce(j.fence_token,0) AS fence_token,
                   coalesce(j.heartbeat_at,j.claimed_at,0) AS last_seen,
                   coalesce(j.stale_recovery_count,0) AS recovery_count
            ORDER BY last_seen ASC, key ASC
            LIMIT $limit
            """,
            cutoff=cutoff,
            limit=limit,
        ).data()
    return [dict(row) for row in rows]


def recover_stale(
    driver,
    *,
    stale_after_sec: int = 600,
    limit: int = 100,
    max_stale_recoveries: int = 3,
    now_ms: int | None = None,
) -> list[dict]:
    """Recover at most ``limit`` stale jobs and return exactly what changed."""
    if stale_after_sec < 1 or limit < 1 or max_stale_recoveries < 1:
        raise ValueError("stale_after_sec, limit, and max_stale_recoveries must be positive")
    now = int(time.time() * 1000) if now_ms is None else int(now_ms)
    cutoff = now - stale_after_sec * 1000
    with driver.session() as session:
        rows = session.run(
            """
            MATCH (j:IngestJob)
            WHERE j.lifecycle_state IN ['CLAIMED','RUNNING','VALIDATING']
              AND coalesce(j.owner,'') <> ''
              AND coalesce(j.heartbeat_at,j.claimed_at,0) < $cutoff
            WITH j
            ORDER BY coalesce(j.heartbeat_at,j.claimed_at,0) ASC, j.key ASC
            LIMIT $limit
            WITH j, coalesce(j.stale_recovery_count,0) + 1 AS next_count,
                 j.owner AS abandoned_owner,
                 coalesce(j.fence_token,0) AS abandoned_fence
            SET j.stale_recovery_count=next_count,
                j.last_abandoned_owner=abandoned_owner,
                j.last_abandoned_fence=abandoned_fence,
                j.last_stale_recovery_at=$now,
                j.recovery_reason='stale_lease',
                j.owner='',
                j.claimed_at=0,
                j.heartbeat_at=0,
                j.current_task=CASE
                    WHEN next_count >= $max_recoveries THEN j.current_task
                    ELSE j.current_task END,
                j.lifecycle_state=CASE
                    WHEN next_count >= $max_recoveries THEN 'QUARANTINED'
                    ELSE 'RETRY' END,
                j.updated_at=$now
            RETURN j.key AS key,
                   j.lifecycle_state AS state,
                   next_count AS recovery_count,
                   abandoned_owner,
                   abandoned_fence
            """,
            cutoff=cutoff,
            limit=limit,
            max_recoveries=max_stale_recoveries,
            now=now,
        ).data()
    return [dict(row) for row in rows]


def _driver():
    from neo4j import GraphDatabase

    from auto_ingest_config import get_neo4j_config

    cfg = get_neo4j_config()
    return GraphDatabase.driver(cfg["uri"], auth=(cfg["user"], cfg["password"]))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m auto_ingest.watchdog")
    parser.add_argument("--stale-after-sec", type=int, default=600)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-stale-recoveries", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    driver = _driver()
    try:
        if args.dry_run:
            rows = stale_jobs(
                driver,
                stale_after_sec=args.stale_after_sec,
                limit=args.limit,
            )
        else:
            rows = recover_stale(
                driver,
                stale_after_sec=args.stale_after_sec,
                limit=args.limit,
                max_stale_recoveries=args.max_stale_recoveries,
            )
        print(json.dumps(rows, indent=2, default=str))
        return 0
    finally:
        driver.close()


if __name__ == "__main__":
    raise SystemExit(main())
