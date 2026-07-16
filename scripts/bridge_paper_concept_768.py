#!/usr/bin/env python3
"""bridge_paper_concept_768.py — semantically link papers to Concepts via 768-vec index.

Papers (embedding_768) and Concepts (embedding_768) share a 768-dim space
(text-embedding-nomic-embed-text-v1.5). This links each paper to its K nearest
Concepts using the `concept_embedding_768` HNSW index, writing
(:Paper)-[:DISCUSSES {method:'vector', score}]->(:Concept).

This is the missing semantic bridge between the research corpus and the activity
/ concept graph: it lets "research themes at HOME" and "papers about X" queries
actually traverse paper->concept. The existing DISCUSSES edges (lexical keyword)
are kept; this adds the vector ones (method:'vector' distinguishes them).

Idempotent & memory-safe: drops prior method:'vector' DISCUSSES in BATCHES (a
single big DELETE OOMs Neo4j's 2.8 GiB per-tx limit), then streams papers in
chunks, vector-querying + MERGing per paper. Re-running is safe.
Usage:
    .venv/bin/python scripts/bridge_paper_concept_768.py [--k 5] [--threshold 0.70] [--batch 400] [--drop-batch 5000]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def log(*a):
    print("[bridge_paper_concept_768]", *a, file=sys.stderr, flush=True)


def get_neo4j():
    from neo4j import GraphDatabase
    from auto_ingest_config import get_neo4j_config
    cfg = get_neo4j_config()
    drv = GraphDatabase.driver(cfg["uri"], auth=(cfg["user"], cfg["password"]))
    return drv, cfg.get("db") or "neo4j"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.70)
    ap.add_argument("--limit", type=int, default=0, help="max papers (0=all)")
    ap.add_argument("--batch", type=int, default=400, help="papers per processing chunk")
    ap.add_argument("--drop-batch", type=int, default=5000, help="edges deleted per tx")
    args = ap.parse_args()

    drv, db = get_neo4j()
    try:
        # 1. Batched drop of existing vector DISCUSSES (one big DELETE OOMs).
        total_dropped = 0
        while True:
            with drv.session(database=db) as s:
                rec = s.run(
                    "MATCH (p:Paper)-[r:DISCUSSES]->(c:Concept) WHERE r.method='vector' "
                    "WITH r LIMIT $lim DELETE r RETURN count(r) AS c",
                    {"lim": args.drop_batch},
                ).single()
                c = rec["c"] if rec else 0
            total_dropped += c
            if c == 0:
                break
            log(f"  dropped {total_dropped} vector DISCUSSES so far...")
        log(f"dropped {total_dropped} old vector DISCUSSES")

        # 2. Stream papers in batches; vector-query + MERGE per paper.
        skip = 0
        created = 0
        processed = 0
        skipped = 0
        while True:
            with drv.session(database=db) as rs:
                q = ("MATCH (p:Paper) WHERE p.embedding_768 IS NOT NULL "
                     "RETURN p.arxiv_id AS id, p.embedding_768 AS e "
                     f"SKIP {skip} LIMIT {args.batch}")
                papers = rs.run(q).data()
            if not papers:
                break
            for p in papers:
                aid, emb = p["id"], p["e"]
                try:
                    with drv.session(database=db) as ws:
                        neigh = ws.run(
                            "CALL db.index.vector.queryNodes('concept_embedding_768', $k, $q) "
                            "YIELD node, score WHERE score >= $t RETURN node.name AS name, score",
                            {"k": args.k, "q": emb, "t": args.threshold},
                        ).data()
                        for nb in neigh:
                            ws.run(
                                "MATCH (p:Paper {arxiv_id:$a}), (c:Concept {name:$n}) "
                                "MERGE (p)-[:DISCUSSES {method:'vector', score:$s}]->(c)",
                                {"a": aid, "n": nb["name"], "s": float(nb["score"])},
                            )
                            created += 1
                except Exception as e:  # noqa: BLE001 - one paper's vector query OOMs; skip & continue
                    log(f"  skip {aid}: {type(e).__name__}: {str(e)[:80]}")
                    skipped += 1
                processed += 1
            skip += args.batch
            if args.limit and processed >= args.limit:
                break
            log(f"processed {processed}, created {created} DISCUSSES edges (skipped {skipped})")

        log(f"DONE: {processed} papers, {created} vector DISCUSSES edges (skipped {skipped})")
    finally:
        drv.close()


if __name__ == "__main__":
    main()
