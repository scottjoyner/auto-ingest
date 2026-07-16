#!/usr/bin/env python3
"""backfill_paper_similar.py — build Paper→Paper SIMILAR edges via the vector index.

The graph has a `paper_embedding_768` HNSW vector index (768-dim, COSINE) over
Paper.embedding_768. This script uses `db.index.vector.queryNodes` (proper
approximate k-NN — NOT O(n^2)) to connect each paper to its K nearest neighbors,
writing (:Paper)-[:SIMILAR {method:'vector', score}]->(:Paper).

Why this exists:
- arxiv_kg_bridge only links freshly-ingested papers to each other (tiny N).
- The bulk of the corpus (papers with embedding_768) had NO paper-to-paper
  similarity edges after the noisy O(n^2) edges were pruned.
- RELATED_CONCEPT edges (Concept-level) are a different relationship and untouched.

Idempotent: clears only `method:'vector'` SIMILAR edges before re-merging, so
re-running is safe. Papers without embedding_768 are skipped (they live in the
384-dim space and need a separate backfill — see NOTES).

Usage:
    .venv/bin/python scripts/backfill_paper_similar.py [--k 10] [--threshold 0.70] [--limit N]
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
    ap.add_argument("--batch", type=int, default=500, help="papers per tx batch")
    args = ap.parse_args()

    drv, db = get_neo4j()
    try:
        with drv.session(database=db) as s:
            total = s.run(
                "MATCH (p:Paper) WHERE p.embedding_768 IS NOT NULL RETURN count(p) AS c"
            ).single()["c"]
            log(f"{total} papers have embedding_768")
            if args.limit:
                total = min(total, args.limit)

            # idempotent: drop existing vector SIMILAR edges before rebuild
            dropped = s.run(
                "MATCH ()-[r:SIMILAR]->() WHERE r.method='vector' DELETE r "
                "RETURN count(r) AS c"
            ).consume().counters.relationships_deleted
            log(f"dropped {dropped} old vector SIMILAR edges")

            created = 0
            processed = 0
            q = "MATCH (p:Paper) WHERE p.embedding_768 IS NOT NULL RETURN p.arxiv_id AS id, p.embedding_768 AS e"
            if args.limit:
                q += f" LIMIT {args.limit}"
            papers = s.run(q).data()

        # process outside the read session so writes get their own txns
        for p in papers:
            aid = p["id"]
            emb = p["e"]
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
            processed += 1
            if processed % args.batch == 0:
                log(f"processed {processed}/{total}, created {created} SIMILAR edges")

        log(f"DONE: processed {processed} papers, created {created} vector SIMILAR edges")
    finally:
        drv.close()


if __name__ == "__main__":
    main()
