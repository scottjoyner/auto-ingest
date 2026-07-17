"""Neo4j watchdog: poll the bolt URI and alert on down / OOM (O-G9).

Read-only. Opens a connection, runs a trivial ``RETURN 1`` and verifies
connectivity. On failure it classifies the error (service unavailable vs
memory-pool OOM vs other), logs it, and optionally POSTs a small JSON alert to
``WATCHDOG_ALERT_URL`` if that env var is set.

Never modifies the graph. Safe to run on a cron/loop.

Usage::

    python3 scripts/neo4j_watchdog.py            # single check, exit 0/1
    python3 scripts/neo4j_watchdog.py --loop 60  # poll every 60s
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
import urllib.request
from typing import Optional, Tuple

log = logging.getLogger("neo4j.watchdog")

OOM_MARKERS = ("MemoryPoolOutOfMemory", "OutOfMemory", "TransientError")
DOWN_MARKERS = ("ServiceUnavailable", "SessionExpired",
                "Connection refused", "defunct connection")


def _neo4j_conf() -> Tuple[str, str, str, str]:
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    db = os.environ.get("NEO4J_DB", "neo4j")
    pw = os.environ.get("NEO4J_PASSWORD", "")
    if not pw:
        try:
            from auto_ingest_config import get_neo4j_password
            pw = get_neo4j_password()
        except Exception:
            pw = ""
    return uri, user, pw, db


def classify_error(exc: BaseException) -> str:
    """Return 'oom', 'down', or 'error' for an exception."""
    s = f"{type(exc).__name__}: {exc}"
    if any(m in s for m in OOM_MARKERS):
        return "oom"
    if any(m in s for m in DOWN_MARKERS):
        return "down"
    return "error"


def send_alert(status: str, detail: str, url: Optional[str] = None) -> bool:
    """POST a small JSON alert to WATCHDOG_ALERT_URL (best-effort).

    Returns True if an alert was sent, False otherwise. Never raises.
    """
    url = url or os.environ.get("WATCHDOG_ALERT_URL")
    if not url:
        return False
    payload = json.dumps({"service": "neo4j", "status": status,
                          "detail": detail, "ts": int(time.time())}).encode()
    try:
        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=5).read()
        return True
    except Exception as e:  # pragma: no cover - network dependent
        log.warning("watchdog alert POST failed: %s", e)
        return False


def check_neo4j(uri: Optional[str] = None, user: Optional[str] = None,
                password: Optional[str] = None, db: Optional[str] = None,
                *, alert: bool = True) -> bool:
    """Return True if Neo4j answers a trivial read; False (and alert) if not.

    Read-only: runs ``RETURN 1``. Import-safe — if the neo4j driver is missing
    it logs and returns False without raising.
    """
    c_uri, c_user, c_pw, c_db = _neo4j_conf()
    uri = uri or c_uri
    user = user or c_user
    password = password if password is not None else c_pw
    db = db or c_db
    try:
        from neo4j import GraphDatabase
    except Exception as e:
        log.error("neo4j driver unavailable: %s", e)
        return False

    driver = None
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session(database=db) as s:
            val = s.run("RETURN 1 AS ok").single()["ok"]
        if val == 1:
            log.info("neo4j OK (%s)", uri)
            return True
        log.warning("neo4j unexpected response: %r", val)
        return False
    except Exception as e:
        kind = classify_error(e)
        log.error("neo4j check failed [%s]: %s", kind, e)
        if alert:
            send_alert(kind, f"{type(e).__name__}: {e}")
        return False
    finally:
        if driver is not None:
            try:
                driver.close()
            except Exception:
                pass


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Neo4j watchdog (read-only).")
    ap.add_argument("--loop", type=int, default=0,
                    help="Poll every N seconds (0 = single check).")
    ap.add_argument("--no-alert", action="store_true",
                    help="Do not POST to WATCHDOG_ALERT_URL on failure.")
    args = ap.parse_args()

    if args.loop > 0:
        while True:
            check_neo4j(alert=not args.no_alert)
            time.sleep(args.loop)
    ok = check_neo4j(alert=not args.no_alert)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
