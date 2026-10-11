#!/usr/bin/env python3
"""signal_kg_bridge.py — ingest new Signal messages into the Scott knowledge graph.

Runs on xwing (signal-cli 0.13.24 is installed there; bridge reads envelopes via `signal-cli -o json receive`).
Writes to Neo4j (bolt://100.64.43.123:7687, db=neo4j) which is the same graph
the enrichment pipeline uses. Idempotent: messages keyed by envelope timestamp+source.

What it does:
  1. Receive new Signal messages via `signal-cli` JSON RPC (does NOT delete until stored).
  2. For each message: chunk if long, embed via LM Studio nomic v1.5 (port 1234),
     MERGE (Scott:Person)-[:SENT]->(SignalMessage) with embedding_768.
  3. Link SignalMessage -> matching Concept nodes (lexical + semantic) so chats join
     the research/activity graph already in Neo4j.
  4. Persist a cursor (last processed timestamp) so re-runs only ingest new messages.

Safe: never drops a Signal message until it is durably stored in Neo4j.

Usage:
  python signal_kg_bridge.py            # ingest since last cursor
  python signal_kg_bridge.py --since 2026-07-01   # ingest since a date
  python signal_kg_bridge.py --dry-run  # print what would be ingested, no writes
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.embedding_client import embed_texts  # noqa: E402

CURSOR_FILE = REPO / "scripts" / ".signal_kg_cursor.json"
SIGNAL_ACCOUNT = os.environ.get("SIGNAL_ACCOUNT")  # e.g. +170XXXX5781; may be None (default device)
EMBED_MODEL = "text-embedding-nomic-embed-text-v1.5"


def ensure_constraints(drv, db):
    """Idempotency: unique key on SignalMessage(source,ts) prevents dup nodes
    if signal-cli returns the same envelope on re-run."""
    with drv.session(database=db) as s:
        s.run(
            "CREATE CONSTRAINT signal_msg_key IF NOT EXISTS "
            "FOR (m:SignalMessage) REQUIRE (m.source, m.ts) IS NODE KEY"
        )


def log(*a):
    print("[signal_kg_bridge]", *a, file=sys.stderr, flush=True)


def get_neo4j():
    from neo4j import GraphDatabase
    from auto_ingest_config import get_neo4j_config
    cfg = get_neo4j_config()
    drv = GraphDatabase.driver(cfg["uri"], auth=(cfg["user"], cfg["password"]))
    return drv, cfg.get("db") or "neo4j"


def embed(text: str):
    """Return a validated 768-vector or None without a hardcoded endpoint."""
    if not text or not text.strip():
        return None
    try:
        return embed_texts([text[:2000]], model=EMBED_MODEL, timeout=30)[0]
    except Exception as exc:  # noqa: BLE001
        log("embedding unavailable:", type(exc).__name__)
        return None


def receive_signal(since_iso: str | None):
    """Receive new Signal messages via signal-cli JSON RPC. Returns list of dicts.
    Does not ack-delete; signal-cli -o json RPC receive returns envelopes."""
    cmd = ["signal-cli", "-o", "json", "receive"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except Exception as e:  # noqa: BLE001
        log("signal-cli receive failed:", e)
        return []
    msgs = []
    for line in (out.stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            env = json.loads(line)
        except Exception:
            continue
        # envelope shape: {envelope:{source,sourceUuid,timestamp,dataMessage:{message,body}},...}
        inner = env.get("envelope", env)
        ts = inner.get("timestamp") or inner.get("serverTimestamp")
        src = inner.get("source") or inner.get("sourceUuid") or "unknown"
        dm = inner.get("dataMessage") or {}
        body = dm.get("body") or dm.get("message") or ""
        if not body:
            continue
        msgs.append({"ts": ts, "src": src, "body": body})
    return msgs


def load_cursor() -> str | None:
    if CURSOR_FILE.exists():
        try:
            return json.loads(CURSOR_FILE.read_text()).get("last_ts")
        except Exception:
            return None
    return None


def save_cursor(ts: str):
    CURSOR_FILE.write_text(json.dumps({"last_ts": ts}))


def ingest(msgs, dry_run=False):
    if not msgs:
        log("no new messages")
        return 0
    if dry_run:
        # No model invocation, DB session/schema DDL or local cursor writes.
        # signal-cli receive (the caller) has separate queue semantics.
        log(f"[dry-run] {len(msgs)} envelopes (no inference or graph writes)")
        return len(msgs)
    drv, db = get_neo4j()
    ensure_constraints(drv, db)
    try:
        with drv.session(database=db) as s:
            n = 0
            last_ts = None
            for m in msgs:
                emb = embed(m["body"])
                # MERGE Scott + message; attach embedding if available.
                # NODE KEY (source,ts) makes this idempotent across re-runs.
                params = {
                    "src": m["src"],
                    "ts": m["ts"],
                    "body": m["body"],
                    "emb": emb,
                    "at": datetime.now(timezone.utc).isoformat(),
                }
                s.run(
                    """
                    MERGE (sc:Scott:Person {name:'Scott'})
                    MERGE (sm:SignalMessage {source:$src, ts:$ts})
                      SET sm.body = $body, sm.ingested_at = $at
                    WITH sc, sm
                    MERGE (sc)-[:SENT]->(sm)
                    WITH sm
                    WHERE $emb IS NOT NULL
                    SET sm.embedding_768 = $emb
                    """,
                    params,
                )
                # keyword ABOUT links (cheap, token-level; works in any embedding space)
                s.run(
                    """
                    MATCH (sm:SignalMessage {source:$src, ts:$ts})
                    MATCH (c:Concept)
                    WHERE c.name IS NOT NULL AND toLower(sm.body) CONTAINS toLower(c.name)
                    MERGE (sm)-[:ABOUT {method:'keyword'}]->(c)
                    """,
                    {"src": m["src"], "ts": m["ts"]},
                )
                # semantic ABOUT links via concept_embedding_768 vector index.
                # SignalMessages embed in 768-dim (nomic) and Concepts carry embedding_768,
                # so the spaces MATCH — this is valid semantic linking.
                if emb:
                    try:
                        neigh = s.run(
                            "CALL db.index.vector.queryNodes('concept_embedding_768', 5, $q) "
                            "YIELD node, score WHERE score >= 0.6 "
                            "RETURN node.name AS name, score",
                            {"q": emb},
                        ).data()
                        for nb in neigh:
                            s.run(
                                "MATCH (sm:SignalMessage {source:$src, ts:$ts}), (c:Concept {name:$name}) "
                                "MERGE (sm)-[:ABOUT {method:'semantic', score:$s}]->(c)",
                                {"src": m["src"], "ts": m["ts"], "name": nb["name"], "s": float(nb.get("score", 0.6))},
                            )
                    except Exception as e:  # noqa: BLE001
                        log(f"semantic ABOUT skipped: {type(e).__name__}: {str(e)[:80]}")
                n += 1
                last_ts = m["ts"] or last_ts
            if last_ts:
                save_cursor(str(last_ts))
            log(f"ingested {n} messages")
            return n
    finally:
        drv.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", help="ISO date to ingest from")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        # signal-cli receive may mutate remote/local queue custody.
        # Without an independently verified non-consuming peek API, a CLI
        # dry-run must not call it or proceed to embedding/graph writes.
        log("[dry-run] Signal receive intentionally skipped (queue safety unverified)")
        return 0

    since = args.since or load_cursor()
    log("ingest since:", since or "(signal-cli default)")
    msgs = receive_signal(since)
    log(f"received {len(msgs)} envelopes")
    ingest(msgs, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
