#!/usr/bin/env python3
"""Reaper: clear expired IngestJob claims so they become re-claimable.

Read-only-safe by construction: it only issues an index-backed, ``LIMIT``-bounded
``SET owner=''`` against ``IngestJob`` nodes whose TTL has elapsed. Run from cron
(e.g. every 10 min). Never issues a full-graph scan/aggregation.

Usage:
    python3 scripts/reap_claims.py [--ttl-sec 3600] [--limit 200]
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


def main() -> int:
    ap = argparse.ArgumentParser(description="Reap expired ingest claims.")
    ap.add_argument("--ttl-sec", type=int, default=3600)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--db", type=str, default="neo4j")
    args = ap.parse_args()

    cfg = get_neo4j_config() or {}
    uri = cfg.get("uri") or "bolt://localhost:7687"
    user = cfg.get("user") or "neo4j"
    pw = cfg.get("password") or get_neo4j_password()

    try:
        d = _driver(uri, user, pw)
        cleared = ingest_claim.reap(d, ttl_sec=args.ttl_sec, limit=args.limit)
        d.close()
    except Exception as e:
        print(f"reap_claims: neo4j unavailable, skipping: {e}", file=sys.stderr)
        return 0
    print(f"reap_claims: cleared {cleared} expired claim(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
