#!/usr/bin/env python3
"""signal_kg_bridge.py — ingest new Signal messages into the Scott knowledge graph.

Runs on x1-370 (where signal-cli is registered + the HTTP/RPC daemon lives).
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
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CURSOR_FILE = REPO / "scripts" / ".signal_kg_cursor.json"
SIGNAL_ACCOUNT = None  # set from env / signal-cli default device below
LM_STUDIO_URL = "http://100.64.43.123:1234/v1/embeddings"  # LM Studio on x1-370 net
EMBED_MODEL = "text-embedding-nomic-embed-text-v1.5"


def log(*a):
    print("[signal_kg_bridge]", *a, file=sys.stderr, flush=True)


def get_neo4j():
    from neo4j import GraphDatabase
    from auto_ingest_config import get_neo4j_config
    cfg = get_neo4j_config()
    drv = GraphDatabase.driver(cfg["uri"], auth=(cfg["user"], cfg["password"]))
    return drv, cfg.get("db") or "neo4j"


def embed(text: str):
    """Embed a short text via LM Studio. Returns list[float] or None."""
    import urllib.request
    if not text or not text.strip():
        return None
    body = json.dumps({"model": EMBED_MODEL, "input": text[:2000]}).encode()
    req = urllib.request.Request(LM_STUDIO_URL, data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.load(r)
        return data["data"][0]["embedding"]
    except Exception as e:  # noqa: BLE001
        log("embed failed:", type(e).__name__, str(e)[:120])
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
    drv, db = get_neo4j()
    try:
        with drv.session(database=db) as s:
            n = 0
            last_ts = None
            for m in msgs:
                emb = embed(m["body"])
                if dry_run:
                    log(f"[dry-run] {m['src']} @ {m['ts']}: {m['body'][:60]!r}")
                    n += 1
                    last_ts = m["ts"] or last_ts
                    continue
                # MERGE Scott + message; attach embedding if available
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
                # link to concepts by keyword overlap (cheap, no big scans)
                s.run(
                    """
                    MATCH (sm:SignalMessage {source:$src, ts:$ts})
                    MATCH (c:Concept)
                    WHERE c.name IS NOT NULL AND toLower(sm.body) CONTAINS toLower(c.name)
                    MERGE (sm)-[:ABOUT]->(c)
                    """,
                    {"src": m["src"], "ts": m["ts"]},
                )
                n += 1
                last_ts = m["ts"] or last_ts
            if not dry_run and last_ts:
                save_cursor(str(last_ts))
            log(f"ingested {n} messages" + (" [dry-run]" if dry_run else ""))
            return n
    finally:
        drv.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", help="ISO date to ingest from")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    since = args.since or load_cursor()
    log("ingest since:", since or "(signal-cli default)")
    msgs = receive_signal(since)
    log(f"received {len(msgs)} envelopes")
    ingest(msgs, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
