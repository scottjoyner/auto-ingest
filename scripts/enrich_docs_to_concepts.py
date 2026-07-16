#!/usr/bin/env python3
"""Bridge knowledge-vault docs (KgNode) to research Concepts via embedding similarity.

KgNode (10,444 docs/chunks) and Concept (270) both carry 768-dim nomic-embeddings in the
SAME space. For each KgNode, find the top-K most similar Concepts (cosine) and MERGE
(KgNode)-[:ABOUT]->(Concept) above a threshold. This connects the entire knowledge vault
to the research topology -> "which documents discuss audio classification?" is a 1-hop
query. Uses existing embeddings only (no new model calls). Idempotent.

Usage:
  python enrich_docs_to_concepts.py                # full bridge
  python enrich_docs_to_concepts.py --limit 2000  # dev
  python enrich_docs_to_concepts.py --threshold 0.72 --top-k 5
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _cfg():
    from auto_ingest_config import get_neo4j_config
    return get_neo4j_config()


THRESHOLD = 0.72
TOP_K = 3
BATCH = 5000


def _cosine_rows(vecs_a, vecs_b):
    """Return list of (top_k indices, scores) per row of a against all b."""
    try:
        import numpy as np
        a = np.asarray(vecs_a, dtype=np.float32)
        b = np.asarray(vecs_b, dtype=np.float32)
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        b /= np.linalg.norm(b, axis=1, keepdims=True)
        sim = a @ b.T  # (na, nb)
        # top-k per row
        out = []
        for row in sim:
            idx = row.argsort()[::-1][:TOP_K]
            out.append([(int(i), float(row[i])) for i in idx])
        return out
    except Exception:
        # pure-python fallback
        import math
        def norm(v):
            return math.sqrt(sum(x * x for x in v))
        na, nb = len(vecs_a), len(vecs_b)
        an = [norm(v) for v in vecs_a]
        bn = [norm(v) for v in vecs_b]
        out = []
        for i in range(na):
            scores = []
            for j in range(nb):
                dot = sum(x * y for x, y in zip(vecs_a[i], vecs_b[j]))
                cos = dot / (an[i] * bn[j]) if an[i] and bn[j] else 0.0
                scores.append((j, cos))
            scores.sort(key=lambda t: t[1], reverse=True)
            out.append(scores[:TOP_K])
        return out


def run(threshold: float, top_k: int, limit: int, quiet: bool) -> int:
    global TOP_K
    TOP_K = top_k
    import neo4j
    cfg = _cfg()
    driver = neo4j.GraphDatabase.driver(cfg["uri"], auth=(cfg["user"], cfg["password"]))
    total = 0
    try:
        with driver.session(database=cfg.get("db")) as sess:
            concepts = sess.run(
                "MATCH (c:Concept) WHERE c.embedding_768 IS NOT NULL "
                "RETURN c.name AS name, c.embedding_768 AS vec"
            ).data()
            if not concepts:
                print("[docs_concepts] no Concept embeddings", flush=True)
                return 0
            cnames = [c["name"] for c in concepts]
            cvecs = [c["vec"] for c in concepts]
            if not quiet:
                print(f"[docs_concepts] {len(cnames)} concepts, thr={threshold}, top_k={top_k}", flush=True)

            lim = f"LIMIT {limit}" if limit else ""
            nodes = list(sess.run(
                f"MATCH (n:KgNode) WHERE n.embedding IS NOT NULL AND NOT (n)-[:ABOUT]->(:Concept) "
                f"RETURN elementId(n) AS eid, n.embedding AS vec {lim}"
            ))
            if not nodes:
                print("[docs_concepts] nothing to bridge", flush=True)
                return 0
            if not quiet:
                print(f"[docs_concepts] bridging {len(nodes)} docs", flush=True)

            nvecs = [n["vec"] for n in nodes]
            topk = _cosine_rows(nvecs, cvecs)
            to_write = []
            for n, hits in zip(nodes, topk):
                for idx, score in hits:
                    if score >= threshold:
                        to_write.append((n["eid"], cnames[idx], score))

            if not quiet:
                print(f"[docs_concepts] {len(to_write)} ABOUT edges above threshold", flush=True)
            for i in range(0, len(to_write), BATCH):
                batch = to_write[i:i + BATCH]
                with sess.begin_transaction() as tx:
                    for eid, cname, score in batch:
                        tx.run(
                            "MATCH (n:KgNode) WHERE elementId(n) = $eid "
                            "MATCH (c:Concept {name: $cn}) "
                            "MERGE (n)-[:ABOUT {score: $s}]->(c)",
                            eid=eid, cn=cname, s=float(score),
                        )
                    tx.commit()
                total += len(batch)
                if not quiet:
                    print(f"[docs_concepts] wrote {total}/{len(to_write)}", flush=True)
    finally:
        driver.close()
    print(f"[docs_concepts] done. wrote {total} ABOUT edges", flush=True)
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    run(args.threshold, args.top_k, args.limit, args.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
