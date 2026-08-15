"""auto_ingest.outbox - durable outbound outbox for ingest writers.

Mirrors ``birdcam/graph/outbox.py`` (LLD §3.4 W-48). Ingest writers record a
pending graph operation to a local SQLite store BEFORE attempting the Neo4j
write. If the write fails (e.g. Neo4j outage mid-ingest), the op is not lost —
it remains in the outbox for later replay. A separate replayer (or a future
AssistX consumer) drains ``pending()`` and calls ``mark_done()``.

This is intentionally dependency-free (stdlib sqlite3) so it cannot break the
ML-heavy ingest import graph.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_OUTBOX_PATH = ".auto_ingest_outbox.sqlite"


class GraphOutbox:
    """Durable store of pending graph operations."""

    def __init__(self, sqlite_path: str = DEFAULT_OUTBOX_PATH) -> None:
        self.path = sqlite_path
        self.conn = sqlite3.connect(sqlite_path, check_same_thread=False)
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS graph_outbox(
                   id INTEGER PRIMARY KEY,
                   op_type TEXT,
                   correlation_id TEXT,
                   payload TEXT,
                   created_at TEXT,
                   retry_count INTEGER DEFAULT 0,
                   last_error TEXT
               )"""
        )
        self.conn.commit()

    def append(self, op_type: str, payload: Dict[str, Any],
               correlation_id: str = "", last_error: str = "") -> int:
        cur = self.conn.execute(
            "INSERT INTO graph_outbox(op_type, correlation_id, payload, created_at, last_error) "
            "VALUES(?,?,?,?,?)",
            (op_type, correlation_id, json.dumps(payload),
             datetime.now(timezone.utc).isoformat(), last_error),
        )
        self.conn.commit()
        return cur.lastrowid

    def pending(self, limit: int = 100) -> List[Tuple[int, str, str, str, int]]:
        rows = self.conn.execute(
            "SELECT id, op_type, correlation_id, payload, retry_count "
            "FROM graph_outbox ORDER BY id LIMIT ?",
            (limit,),
        ).fetchall()
        return [(r[0], r[1], r[2], r[3], r[4]) for r in rows]

    def mark_done(self, id_: int) -> None:
        self.conn.execute("DELETE FROM graph_outbox WHERE id=?", (id_,))
        self.conn.commit()

    def mark_failed(self, id_: int, err: str) -> None:
        self.conn.execute(
            "UPDATE graph_outbox SET retry_count=retry_count+1, last_error=? WHERE id=?",
            (err, id_),
        )
        self.conn.commit()

    def count_pending(self) -> int:
        row = self.conn.execute("SELECT count(*) FROM graph_outbox").fetchone()
        return int(row[0]) if row else 0

    def replay(self, verify: Optional[callable] = None,
               max_ops: int = 1000, max_retries: int = 3,
               on_status=None) -> Dict[str, Any]:
        """Drain pending outbox ops, returning a summary dict.

        Ops are staged *before* the Neo4j write, so a pending op means the write
        may or may not have landed. The op payload is only the durable intent
        (not the full row), so replay cannot rebuild a missed write by itself.

        ``verify(op_type, payload) -> bool`` decides whether the intent was
        satisfied: return True to mark the op done (drained), False to keep it
        pending (a real gap for re-ingest from source). When ``verify`` is None,
        ops are only counted (no mutation).

        Ops exceeding ``max_retries`` are marked failed and left pending.
        """
        pending = self.pending(limit=max_ops)
        summary = {"drained": 0, "kept": 0, "failed": 0, "seen": len(pending),
                   "max_retries": max_retries}
        for op_id, op_type, correlation_id, payload_json, retry_count in pending:
            if retry_count > max_retries:
                self.mark_failed(op_id, "replay: exceeded max_retries")
                summary["failed"] += 1
                continue
            try:
                payload = json.loads(payload_json)
            except Exception as e:
                self.mark_failed(op_id, f"replay: bad payload: {e}")
                summary["failed"] += 1
                continue
            if verify is None:
                summary["kept"] += 1
                continue
            try:
                ok = bool(verify(op_type, payload))
            except Exception as e:
                self.mark_failed(op_id, f"replay: verify error: {e}")
                summary["failed"] += 1
                continue
            if ok:
                self.mark_done(op_id)
                summary["drained"] += 1
            else:
                # Intent not satisfied in the graph — genuine gap. Keep it
                # pending (retry_count untouched) for a later re-ingest pass.
                summary["kept"] += 1
            if on_status:
                on_status(op_id, op_type, ok)
        return summary

    def close(self) -> None:
        self.conn.close()


_OUTBOX: Optional[GraphOutbox] = None


def get_outbox(sqlite_path: Optional[str] = None) -> Optional[GraphOutbox]:
    """Return the process-wide outbox (or None if disabled).

    The outbox is opt-in via env ``AUTO_INGEST_OUTBOX=1`` (or an explicit path)
    so existing ingest runs are untouched by default. When enabled, writers
    stage ops durably before touching Neo4j.
    """
    global _OUTBOX
    if _OUTBOX is not None:
        return _OUTBOX
    env = (sqlite_path or "").strip() or ""
    import os

    if os.environ.get("AUTO_INGEST_OUTBOX"):
        path = env or os.environ.get("AUTO_INGEST_OUTBOX_PATH", DEFAULT_OUTBOX_PATH)
        _OUTBOX = GraphOutbox(path)
        return _OUTBOX
    return None


def _verify_transcription_exists(op_type: str, payload: Dict[str, Any]) -> bool:
    """Default verifier: True when the staged Transcription node exists.

    Replay only drains ops whose write actually landed; ops whose Transcription
    is still absent stay pending so the source can be re-ingested.
    """
    from auto_ingest_config import get_neo4j_config
    from neo4j import GraphDatabase

    t_id = payload.get("t_id") or payload.get("id")
    if not t_id:
        return False
    cfg = get_neo4j_config()
    driver = GraphDatabase.driver(
        cfg["uri"], auth=(cfg["user"], cfg["password"]), database=cfg.get("database")
    )
    try:
        with driver.session() as sess:
            rec = sess.run(
                "MATCH (t:Transcription {id:$tid}) RETURN t.id", tid=t_id
            ).single()
            return rec is not None
    finally:
        driver.close()


def _drain(limit: int = 1000, max_retries: int = 3, dry_run: bool = False) -> Dict[str, Any]:
    """Drain the durable outbox into the graph, returning a summary dict."""
    ob = GraphOutbox(os.environ.get("AUTO_INGEST_OUTBOX_PATH", DEFAULT_OUTBOX_PATH))
    try:
        n = ob.count_pending()
        print(f"outbox: {n} pending op(s)")
        verify = None if dry_run else _verify_transcription_exists
        summary = ob.replay(
            verify=verify, max_ops=limit, max_retries=max_retries,
            on_status=lambda op_id, op_type, ok: print(
                f"  op {op_id} [{op_type}]: {'drained' if ok else 'kept'}"
            ),
        )
        print(f"outbox: {summary['drained']} drained, {summary['kept']} kept "
              f"(re-ingest), {summary['failed']} failed, {summary['seen']} seen")
        return summary
    finally:
        ob.close()


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        description="Drain the durable ingest outbox: mark done ops whose "
                    "writes landed; keep true gaps pending for re-ingest.")
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--dry-run", action="store_true",
                   help="Report pending ops without draining (no graph writes)")
    args = p.parse_args(argv)
    summary = _drain(limit=args.limit, max_retries=args.max_retries, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
