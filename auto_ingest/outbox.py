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
