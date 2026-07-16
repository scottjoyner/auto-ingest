#!/usr/bin/env python3
"""backfill_paper_768.py — bring existing papers into the 768-dim embedding space.

The corpus has 61K+ papers with a 384-dim `embedding` (all-MiniLM-L6-v2) but only
the papers ingested by arxiv_kg_bridge have the 768-dim `embedding_768`
(text-embedding-nomic-embed-text-v1.5). The `paper_embedding_768` vector index and
backfill_paper_similar.py operate on `embedding_768`, so the 384-only papers are
invisible to paper-to-paper similarity.

This script embeds each paper's abstract (or title) via LM Studio and writes
`embedding_768`. Resumable via a cursor file; idempotent (skips papers that
already have embedding_768). Designed to run incrementally as a cron until the
whole corpus is in the 768 space.

Usage:
    .venv/bin/python scripts/backfill_paper_768.py [--limit N] [--batch 200]
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CURSOR_FILE = REPO / "scripts" / ".paper768_cursor.json"
LM_STUDIO_URL = "http://100.64.43.123:1234/v1/embeddings"
EMBED_MODEL = "text-embedding-nomic-embed-text-v1.5"


def log(*a):
    print("[backfill_paper_768]", *a, file=sys.stderr, flush=True)


def get_neo4j():
    from neo4j import GraphDatabase
    from auto_ingest_config import get_neo4j_config
    cfg = get_neo4j_config()
    drv = GraphDatabase.driver(cfg["uri"], auth=(cfg["user"], cfg["password"]))
    return drv, cfg.get("db") or "neo4j"


def embed(text: str):
    req = urllib.request.Request(
        LM_STUDIO_URL,
        data=json.dumps({"model": EMBED_MODEL, "input": text}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)["data"][0]["embedding"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200, help="papers to process this run")
    ap.add_argument("--batch", type=int, default=100, help="writes per tx")
    args = ap.parse_args()

    drv, db = get_neo4j()
    try:
        with drv.session(database=db) as s:
            todo = s.run(
                "MATCH (p:Paper) WHERE p.embedding_768 IS NULL "
                "AND (p.abstract IS NOT NULL OR p.title IS NOT NULL) "
                "RETURN p.arxiv_id AS id, "
                "COALESCE(p.abstract, p.title) AS text "
                f"LIMIT {args.limit}"
            ).data()
        log(f"{len(todo)} papers to embed this run")
        done = 0
        for p in todo:
            try:
                vec = embed(p["text"][:8000])
                with drv.session(database=db) as ws:
                    ws.run(
                        "MATCH (p:Paper {arxiv_id:$id}) SET p.embedding_768=$e",
                        {"id": p["id"], "e": vec},
                    )
                done += 1
            except Exception as e:  # noqa: BLE001
                log(f"skip {p['id']}: {type(e).__name__}: {str(e)[:80]}")
            if done % args.batch == 0:
                log(f"embedded {done}/{len(todo)}")
        log(f"DONE: embedded {done} papers this run")
    finally:
        drv.close()


if __name__ == "__main__":
    main()
