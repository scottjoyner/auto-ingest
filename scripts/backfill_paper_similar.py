#!/usr/bin/env python3
"""backfill_paper_similar.py — build Paper→Paper SIMILAR edges via the vector index.

The graph has a `paper_embedding_768` HNSW vector index (768-dim, COSINE) over
Paper.embedding_768. This script uses `db.index.vector.queryNodes` (proper
approximate k-NN — NOT O(n^2)) to connect each paper to its K nearest neighbors,
writing (:Paper)-[:SIMILAR {method:'vector', score}]->(:Paper).

Idempotent & memory-safe:
- Drops prior method:'vector' SIMILAR edges in BATCHES (a single big DELETE OOMs
  Neo4j's 2.8 GiB per-tx limit).
- Streams papers in chunks so no single transaction exceeds the heap.
- Per-paper vector queries are wrapped in try/except: if one paper's k-NN query
  hits MemoryPoolOutOfMemory (intermittent on a large HNSW index), that paper is
  skipped and the rebuild continues, so the run completes (~99.9% coverage) instead
  of dying at paper ~9,600.

Usage:
    .venv/bin/python scripts/backfill_paper_similar.py [--k 10] [--threshold 0.70] [--batch 400] [--drop-batch 4000]
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def log(*a):
    print("[backfill_paper_similar]", *a, file=sys.stderr, flush=True)


def get_neo4j():
    from neo4j import GraphDatabase
    from auto_ingest_config import get_neo4j_config
    cfg = get_neo4j_config()
    drv = GraphDatabase.driver(cfg["uri"], auth=(cfg["user"], cfg["password"]))
    return drv, cfg.get("db") or "neo4j"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10, help="neighbors per paper")
    ap.add_argument("--threshold", type=float, default=0.70, help="min cosine score")
    ap.add_argument("--limit", type=int, default=0, help="max papers to process (0=all)")
    ap.add_argument("--batch", type=int, default=400, help="papers per processing chunk")
    ap.add_argument("--drop-batch", type=int, default=4000, help="edges deleted per tx")
    args = ap.parse_args()

    drv, db = get_neo4j()
    try:
        # 1. Batched drop of existing vector SIMILAR edges.
        total_dropped = 0
        while True:
            with drv.session(database=db) as s:
                rec = s.run(
                    "MATCH ()-[r:SIMILAR]->() WHERE r.method='vector' "
                    "WITH r LIMIT $lim DELETE r RETURN count(r) AS c",
                    {"lim": args.drop_batch},
                ).single()
                c = rec["c"] if rec else 0
            total_dropped += c
            if c == 0:
                break
            log(f"  dropped {total_dropped} vector SIMILAR edges so far...")
        log(f"dropped {total_dropped} old vector SIMILAR edges")

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
                            "CALL db.index.vector.queryNodes('paper_embedding_768', $k, $q) "
                            "YIELD node, score WHERE node.arxiv_id <> $id AND score >= $t "
                            "RETURN node.arxiv_id AS nid, score",
                            {"k": args.k + 1, "q": emb, "id": aid, "t": args.threshold},
                        ).data()
                        for n in neigh:
                            ws.run(
                                "MATCH (a:Paper {arxiv_id:$a}), (b:Paper {arxiv_id:$b}) "
                                "MERGE (a)-[:SIMILAR {method:'vector', score:$s}]->(b)",
                                {"a": aid, "b": n["nid"], "s": float(n["score"])},
                            )
                            created += 1
                except Exception as e:  # noqa: BLE001 - one paper's vector query OOMs; skip & continue
                    log(f"  skip {aid}: {type(e).__name__}: {str(e)[:80]}")
                    skipped += 1
                processed += 1
            skip += args.batch
            if args.limit and processed >= args.limit:
                break
            log(f"processed {processed}, created {created} SIMILAR edges (skipped {skipped})")

        log(f"DONE: processed {processed} papers, created {created} vector SIMILAR edges (skipped {skipped})")
    finally:
        drv.close()


if __name__ == "__main__":
    main()
