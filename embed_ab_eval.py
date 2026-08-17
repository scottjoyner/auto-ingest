#!/usr/bin/env python3
"""
embed_ab_eval.py — A/B harness comparing two vector properties' retrieval behavior.

For a sample of nodes carrying BOTH --prop-x and --prop-y, query each property's own
vector index using the node's stored vector of that prop, then compare the returned
neighbor sets. Reports mean Jaccard (set overlap) and mean top-1 agreement between the
two embedding spaces' neighborhoods. Cheap: no models loaded (uses stored vectors).

Default target: Chunk with emb_e5_large vs emb_mini12 (research KG).
"""
from __future__ import annotations
import os
import sys
import random
import argparse
import logging
from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default=os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687"))
    ap.add_argument("--user", default=os.getenv("NEO4J_USER", "neo4j"))
    ap.add_argument("--password", default=os.getenv("NEO4J_PASSWORD", "knowledge_graph_2026"))
    ap.add_argument("--label", default="Chunk")
    ap.add_argument("--prop-x", default="emb_e5_large")
    ap.add_argument("--prop-y", default="emb_mini12")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    ix_x = f"{args.label}_{args.prop_x}_index"
    ix_y = f"{args.label}_{args.prop_y}_index"

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    rng = random.Random(args.seed)
    with driver.session() as s:
        nodes = s.run(
            f"MATCH (n:{args.label}) WHERE n.{args.prop_x} IS NOT NULL AND n.{args.prop_y} IS NOT NULL "
            f"RETURN id(n) AS nid, n.{args.prop_x} AS vx, n.{args.prop_y} AS vy "
            f"LIMIT 5000"
        ).data()
        if not nodes:
            logging.error("no nodes have both %s and %s; run the embedding jobs first", args.prop_x, args.prop_y)
            return 1
        if len(nodes) > args.sample:
            nodes = rng.sample(nodes, args.sample)
        logging.info("probes=%d (label=%s, %s vs %s)", len(nodes), args.label, args.prop_x, args.prop_y)

        jac_sum, top1_sum = 0.0, 0.0
        for n in nodes:
            rx = s.run(
                "CALL db.index.vector.queryNodes($i,$k,$v) YIELD node RETURN id(node) AS id",
                i=ix_x, k=args.topk, v=n["vx"],
            ).data()
            ry = s.run(
                "CALL db.index.vector.queryNodes($i,$k,$v) YIELD node RETURN id(node) AS id",
                i=ix_y, k=args.topk, v=n["vy"],
            ).data()
            sx = {r["id"] for r in rx}
            sy = {r["id"] for r in ry}
            union = len(sx | sy)
            jac = len(sx & sy) / union if union else 0.0
            top1 = 1.0 if (rx and ry and rx[0]["id"] == ry[0]["id"]) else 0.0
            jac_sum += jac
            top1_sum += top1
        n = len(nodes) or 1
        logging.info("MEAN Jaccard@%d = %.3f", args.topk, jac_sum / n)
        logging.info("MEAN top-1 agreement = %.3f", top1_sum / n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
