#!/usr/bin/env python3
"""
link_entities_dedupe.py — dedupe / link entity nodes WITHIN a single Neo4j KG.

For each node of --label that carries --prop, ANN-query the same label's vector index and
link near-duplicates with an idempotent SAME_AS relationship. Because everything stays
inside one database, native relationships work.

Run after the KG's entity nodes are embedded (emb_e5_large).
"""
from __future__ import annotations
import os
import sys
import argparse
import logging
from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default=os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687"))
    ap.add_argument("--user", default=os.getenv("NEO4J_USER", "neo4j"))
    ap.add_argument("--password", default=os.getenv("NEO4J_PASSWORD", "knowledge_graph_2026"))
    ap.add_argument("--label", default="Entity")
    ap.add_argument("--prop", default="emb_e5_large")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--threshold", type=float, default=0.92)
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0, help="cap #source nodes (0=all)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    idx = f"{args.label}_{args.prop}_index"

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    created = 0
    with driver.session() as s:
        last = -1
        processed = 0
        while True:
            rows = s.run(
                f"MATCH (n:{args.label}) WHERE n.{args.prop} IS NOT NULL AND id(n) > $last "
                f"RETURN id(n) AS nid, n.{args.prop} AS vec "
                f"ORDER BY id(n) ASC LIMIT $lim",
                last=last, lim=args.batch,
            ).data()
            if not rows:
                break
            for r in rows:
                last = r["nid"]
                vec = r["vec"]
                res = s.run(
                    "CALL db.index.vector.queryNodes($idx, $k, $vec) YIELD node, score "
                    "RETURN id(node) AS cid, score",
                    idx=idx, k=args.topk, vec=vec,
                ).data()
                for c in res:
                    cid = c["cid"]
                    if cid == last:
                        continue
                    if c["score"] >= args.threshold:
                        if args.dry_run:
                            logging.info("[dry] %d <-> %d score=%.3f", last, cid, c["score"])
                        else:
                            s.run(
                                f"MATCH (a:{args.label}) WHERE id(a)=$aid "
                                f"MATCH (b:{args.label}) WHERE id(b)=$bid "
                                f"MERGE (a)-[r:SAME_AS]-(b) "
                                f"SET r.score=$score, r.prop=$prop, r.label=$label",
                                aid=last, bid=cid, score=c["score"], prop=args.prop, label=args.label,
                            )
                            created += 1
                processed += 1
                if args.limit and processed >= args.limit:
                    break
            if args.limit and processed >= args.limit:
                break
    logging.info("DONE created=%d", created)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
