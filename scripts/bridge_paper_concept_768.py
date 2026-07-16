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

Idempotent: drops prior method:'vector' DISCUSSES from papers, rebuilds.
Usage:
    .venv/bin/python scripts/bridge_paper_concept_768.py [--k 5] [--threshold 0.70]
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
    args = ap.parse_args()

    drv, db = get_neo4j()
    try:
        with drv.session(database=db) as s:
            total = s.run(
                "MATCH (p:Paper) WHERE p.embedding_768 IS NOT NULL RETURN count(p) AS c"
            ).single()["c"]
            if args.limit:
                total = min(total, args.limit)
            dropped = s.run(
                "MATCH (p:Paper)-[r:DISCUSSES]->(c:Concept) WHERE r.method='vector' "
                "DELETE r RETURN count(r) AS c"
            ).consume().counters.relationships_deleted
            log(f"{total} papers w/ embedding_768; dropped {dropped} old vector DISCUSSES")
            q = "MATCH (p:Paper) WHERE p.embedding_768 IS NOT NULL RETURN p.arxiv_id AS id, p.embedding_768 AS e"
            if args.limit:
                q += f" LIMIT {args.limit}"
            papers = s.run(q).data()

        created = 0
        processed = 0
        for p in papers:
            aid, emb = p["id"], p["e"]
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
            processed += 1
            if processed % 500 == 0:
                log(f"processed {processed}/{total}, created {created} edges")

        log(f"DONE: {processed} papers, {created} vector DISCUSSES edges")
    finally:
        drv.close()


if __name__ == "__main__":
    main()
