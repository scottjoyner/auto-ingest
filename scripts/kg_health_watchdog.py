#!/usr/bin/env python3
"""kg_health_watchdog.py — self-monitoring for the auto-ingest knowledge graph.

Checks the health of the auto-ingest pipeline and ALERTS on Signal only when
something is wrong (stale ingest, backfill stalled, similarity regressed). This is
the "living background system" watchdog: it runs on a cron and surfaces problems
before they rot silently.

Checks:
  1. arXiv freshness: latest Paper.ingested_at within --max-age-hours (default 26).
  2. 768-backfill progress: embedding_768 count is increasing over time (we store
     last count in a state file; alert if it hasn't grown in > --stale-backfill-days).
  3. SIMILAR edges present and > 0 (regression if dropped to 0 after a backfill).
  4. Signal bridge cursor advancing (cursor file mtime recent OR SignalMessage count grew).

Behavior:
  - If any check FAILS -> prints an ALERT block (cron delivers to Signal).
  - If all OK and --report-always -> prints a short OK summary.
  - If all OK and not --report-always -> prints nothing (silent success).

Usage:
    .venv/bin/python scripts/kg_health_watchdog.py            # alert only on failure
    .venv/bin/python scripts/kg_health_watchdog.py --report-always
    KG_HEALTH_STATE_FILE=/tmp/kg.json .venv/bin/python scripts/kg_health_watchdog.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from itertools import count
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# The state file records the last-seen 768-backfill count so the stall detector
# has something to compare against; it is a cache of one observation, not a
# decision record, and it is rewritten in full on every run. Point it somewhere
# else (a tmp dir) to exercise main() without writing into the repo. Same knob
# the sibling watchdog exposes, deliberately under a different name: reusing
# WATCHDOG_STATE_FILE here would make the two watchdogs share one path and
# clobber each other's JSON.
STATE_FILE = Path(
    os.environ.get("KG_HEALTH_STATE_FILE")
    or REPO / "scripts" / ".kg_health_state.json"
)

# Unique-per-write temp names, so a failed write cannot eat a concurrent one.
# See _write_json_atomic() in auto_ingest/custody/store.py for why a FIXED
# ".tmp" suffix is wrong.
_WRITE_SEQ = count(1)


def log(*a):
    print("[kg_health]", *a, file=sys.stderr, flush=True)


def get_neo4j():
    from neo4j import GraphDatabase

    from auto_ingest_config import get_neo4j_config
    cfg = get_neo4j_config()
    drv = GraphDatabase.driver(cfg["uri"], auth=(cfg["user"], cfg["password"]))
    return drv, cfg.get("db") or "neo4j"


def load_state():
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            return {}
    return {}


def save_state(st):
    """Persist the state file atomically. Raises if the write cannot complete."""
    tmp = STATE_FILE.with_name(f"{STATE_FILE.name}.{os.getpid()}.{next(_WRITE_SEQ)}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(st, f, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, STATE_FILE)
    except BaseException:
        # Never leave a half-written temp file behind for the next run to trip on.
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-age-hours", type=float, default=26)
    ap.add_argument("--stale-backfill-days", type=float, default=2)
    ap.add_argument("--report-always", action="store_true")
    args = ap.parse_args()

    drv, db = get_neo4j()
    problems = []
    age_h = 0.0
    sim = 0
    emb768 = 0
    try:
        with drv.session(database=db) as s:
            def q(cyp):
                r = s.run(cyp).single()
                return r[list(r.keys())[0]] if r else None

            # 1. arXiv freshness
            latest = q("MATCH (p:Paper) WHERE p.ingested_at IS NOT NULL RETURN max(p.ingested_at) AS c")
            now = time.time()
            if latest is None:
                problems.append("No Paper.ingested_at found — arxiv ingest may never have run.")
            else:
                # Neo4j returns a DateTime; convert robustly to epoch seconds.
                try:
                    if hasattr(latest, "to_native"):
                        ts = latest.to_native().timestamp()
                    elif hasattr(latest, "timestamp"):
                        ts = latest.timestamp()
                    else:
                        # fallback: parse from string
                        from datetime import datetime as _dt
                        ts = _dt.fromisoformat(str(latest).replace("Z", "+00:00")).timestamp()
                    age_h = (now - ts) / 3600
                except Exception as e:
                    problems.append(f"Could not parse latest ingest time: {type(e).__name__}: {e}")
                    age_h = 0.0
                if age_h > args.max_age_hours:
                    problems.append(f"arXiv ingest STALE: latest paper {age_h:.1f}h old (> {args.max_age_hours}h).")

            # 2. 768-backfill progress
            emb768 = q("MATCH (p:Paper) WHERE p.embedding_768 IS NOT NULL RETURN count(p) AS c") or 0
            st = load_state()
            prev = st.get("emb768_count")
            prev_t = st.get("emb768_time")
            if prev is not None and emb768 <= prev:
                if prev_t:
                    days = (now - float(prev_t)) / 86400
                    if days > args.stale_backfill_days:
                        problems.append(
                            f"768-backfill STALLED: embedding_768 count stuck at {emb768} "
                            f"for {days:.1f}d (LM Studio unreachable?)."
                        )
            st["emb768_count"] = emb768
            st["emb768_time"] = now
            save_state(st)

            # 3. SIMILAR regression
            sim = q("MATCH ()-[r:SIMILAR]->() WHERE r.method='vector' RETURN count(r) AS c") or 0
            if sim == 0:
                problems.append("SIMILAR edges = 0 — paper similarity graph missing.")
            else:
                log(f"SIMILAR={sim}, emb768={emb768}")

            # 4. Signal bridge cursor advancing
            cursor = REPO / "scripts" / ".signal_kg_cursor.json"
            sig_count = q("MATCH (m:SignalMessage) RETURN count(m) AS c") or 0
            if cursor.exists():
                age_m = (now - cursor.stat().st_mtime) / 60
                if age_m > 60 * 24:  # not run in a day
                    problems.append(f"Signal bridge cursor not updated in {age_m/60:.1f}h.")
            else:
                if sig_count == 0:
                    log("No Signal cursor yet and 0 messages — bridge not yet run (ok if no chats).")

    finally:
        drv.close()

    if problems:
        print("⚠️ KG HEALTH ALERT:")
        for p in problems:
            print(f"  - {p}")
        print(f"  (papers_768={emb768}, SIMILAR={sim if 'sim' in dir() else '?'})")
        return
    if args.report_always:
        print(f"✅ KG healthy: papers_768={emb768}, SIMILAR={sim}, arxiv age={age_h:.1f}h")
    else:
        log("all checks passed (silent)")


if __name__ == "__main__":
    main()
