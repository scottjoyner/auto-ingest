"""Operator recovery/integrity commands for auto-ingest.

All mutations are single-job bounded and explicit. Nothing bulk-destructive is
performed implicitly. This gives quarantined/retry states a supported recovery
surface instead of requiring hand-written Cypher.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from auto_ingest.artifact_reconcile import reconcile
from auto_ingest.orchestration import lifecycle
from auto_ingest.runtime_schema import audit_schema, ensure_schema


def retry_job(driver, key: str) -> bool:
    """Explicitly return one failed/quarantined job to READY state."""
    with driver.session() as session:
        rec = session.run(
            """
            MATCH (j:IngestJob {key:$key})
            WHERE j.lifecycle_state IN ['RETRY','FAILED','QUARANTINED']
            SET j.lifecycle_state='READY',
                j.owner='',
                j.claimed_at=0,
                j.current_task=NULL,
                j.manual_retry_count=coalesce(j.manual_retry_count,0)+1,
                j.last_manual_retry_at=timestamp(),
                j.updated_at=timestamp()
            RETURN j.key AS key
            """,
            key=key,
        ).single()
    return rec is not None


def quarantine_job(driver, key: str, reason: str) -> bool:
    if not reason.strip():
        raise ValueError("quarantine reason must be non-empty")
    with driver.session() as session:
        rec = session.run(
            """
            MATCH (j:IngestJob {key:$key})
            SET j.lifecycle_state='QUARANTINED',
                j.owner='',
                j.claimed_at=0,
                j.quarantine_reason=left($reason,2000),
                j.quarantined_at=timestamp(),
                j.updated_at=timestamp()
            RETURN j.key AS key
            """,
            key=key,
            reason=reason,
        ).single()
    return rec is not None


def inspect_job(driver, key: str) -> dict | None:
    state = lifecycle(driver, key)
    if state is None:
        return None
    with driver.session() as session:
        artifacts = session.run(
            """
            MATCH (:IngestJob {key:$key})-[:PRODUCED]->(a:IngestArtifact)
            RETURN a.artifact_id AS artifact_id,
                   a.stage AS stage,
                   a.stage_version AS stage_version,
                   a.path AS path,
                   a.sha256 AS sha256,
                   a.fence_token AS fence_token
            ORDER BY a.stage, a.artifact_id
            LIMIT 500
            """,
            key=key,
        ).data()
    return {"job": state, "artifacts": artifacts}


def _driver():
    from neo4j import GraphDatabase

    from auto_ingest_config import get_neo4j_config

    cfg = get_neo4j_config()
    return GraphDatabase.driver(cfg["uri"], auth=(cfg["user"], cfg["password"]))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m auto_ingest.operations")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect_p = sub.add_parser("inspect")
    inspect_p.add_argument("--job-key", required=True)

    retry_p = sub.add_parser("retry")
    retry_p.add_argument("--job-key", required=True)

    quarantine_p = sub.add_parser("quarantine")
    quarantine_p.add_argument("--job-key", required=True)
    quarantine_p.add_argument("--reason", required=True)

    reconcile_p = sub.add_parser("reconcile")
    reconcile_p.add_argument("--artifact-root", required=True)

    sub.add_parser("schema-audit")
    sub.add_parser("schema-ensure")

    args = parser.parse_args(argv)
    driver = _driver()
    try:
        if args.command == "inspect":
            print(json.dumps(inspect_job(driver, args.job_key), indent=2, default=str))
            return 0
        if args.command == "retry":
            return 0 if retry_job(driver, args.job_key) else 2
        if args.command == "quarantine":
            return 0 if quarantine_job(driver, args.job_key, args.reason) else 2
        if args.command == "reconcile":
            findings = reconcile(Path(args.artifact_root), driver)
            print(json.dumps([f.__dict__ for f in findings], indent=2, default=str))
            return 1 if any(f.classification not in {"HEALTHY", "JOURNAL_STALE"} for f in findings) else 0
        if args.command == "schema-audit":
            report = audit_schema(driver)
            print(json.dumps(report, indent=2, default=str))
            return 0 if report["ok"] else 1
        report = ensure_schema(driver)
        print(json.dumps(report, indent=2, default=str))
        return 0 if report["ok"] else 1
    finally:
        driver.close()


if __name__ == "__main__":
    raise SystemExit(main())
