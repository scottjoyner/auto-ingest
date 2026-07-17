#!/usr/bin/env python3
"""Graph-side helpers for the claim protocol (manifest create/claim/release).

These wrappers are the bridge between the file-based drop queue
(``deploy/worker_ingest.sh``, ``deploy/create_job.sh``) and the Neo4j
``IngestJob`` manifest. Every call degrades gracefully if Neo4j is unreachable
(log + return non-zero without raising) so the primary ``mv`` file lock always
wins and a down graph never blocks an ingest run.

Usage:
    python3 scripts/claim_job.py create KEY
    python3 scripts/claim_job.py claim  KEY OWNER [--ttl-sec 3600]
    python3 scripts/claim_job.py release KEY OWNER
    python3 scripts/claim_job.py stage  KEY STAGE [--owner OWNER]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from auto_ingest import ingest_claim  # noqa: E402
from auto_ingest_config import get_neo4j_config, get_neo4j_password  # noqa: E402


def _driver(uri: str, user: str, pw: str) -> "object":
    from neo4j import GraphDatabase
    return GraphDatabase.driver(uri, auth=(user, pw))


def _connect():
    cfg = get_neo4j_config() or {}
    uri = cfg.get("uri") or "bolt://localhost:7687"
    user = cfg.get("user") or "neo4j"
    pw = cfg.get("password") or get_neo4j_password()
    return _driver(uri, user, pw)


def main() -> int:
    ap = argparse.ArgumentParser(description="IngestJob manifest helpers.")
    sub = ap.add_subparsers(dest="action", required=True)

    p = sub.add_parser("create")
    p.add_argument("key")

    p = sub.add_parser("claim")
    p.add_argument("key")
    p.add_argument("owner")
    p.add_argument("--ttl-sec", type=int, default=3600)

    p = sub.add_parser("release")
    p.add_argument("key")
    p.add_argument("owner")

    p = sub.add_parser("stage")
    p.add_argument("key")
    p.add_argument("stage")
    p.add_argument("--owner", default="")

    args = ap.parse_args()

    try:
        d = _connect()
    except Exception as e:
        print(f"claim_job: neo4j unavailable, skipping: {e}", file=sys.stderr)
        return 0

    try:
        if args.action == "create":
            ingest_claim.create_job(d, args.key)
            print(f"claim_job: created manifest {args.key}")
        elif args.action == "claim":
            won = ingest_claim.claim(d, args.key, args.owner, ttl_sec=args.ttl_sec)
            print(f"claim_job: claim {args.key} by {args.owner} -> {won}")
            return 0 if won else 2
        elif args.action == "release":
            ingest_claim.release(d, args.key, args.owner)
            print(f"claim_job: released {args.key}")
        elif args.action == "stage":
            ingest_claim.update_stage(d, args.key, args.stage, owner=args.owner or None)
            print(f"claim_job: stage {args.key} {args.stage}")
    except Exception as e:
        print(f"claim_job: graph call failed, skipping: {e}", file=sys.stderr)
        return 0
    finally:
        d.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
