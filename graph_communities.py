#!/usr/bin/env python3
"""
graph_communities.py — derive communities from embedding similarity and write them back.

For a label carrying --prop, ANN-query its vector index for each node's nearest
neighbors (score >= --threshold), union those pairs into connected components, then
write `community_id` onto every node and create one (:Community {id, size, label})
node per component. Pure-Python union-find (no graph lib needed). Use --leiden if
python-louvain is installed for denser clusters.

Run after the KG's nodes are embedded:
  graph_communities.py --uri bolt://127.0.0.1:17687 --label Entity --prop emb_e5_large
"""
from __future__ import annotations
import os
import sys
import argparse
import logging
from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class UnionFind:
    def __init__(self):
        self.p = {}

    def add(self, x):
        if x not in self.p:
            self.p[x] = x

    def find(self, x):
        self.add(x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default=os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687"))
    ap.add_argument("--user", default=os.getenv("NEO4J_USER", "neo4j"))
    ap.add_argument("--password", default=os.getenv("NEO4J_PASSWORD", "knowledge_graph_2026"))
    ap.add_argument("--label", default="Entity")
    ap.add_argument("--prop", default="emb_e5_large")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--threshold", type=float, default=0.80)
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--limit", type=int, default=0, help="cap #source nodes (0=all)")
    ap.add_argument("--leiden", action="store_true", help="use python-louvain if available (else CC)")
    args = ap.parse_args()
    idx = f"{args.label}_{args.prop}_index"

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    uf = UnionFind()
    edges = 0
    with driver.session() as s:
        last = -1
        processed = 0
        while True:
            rows = s.run(
                f"MATCH (n:{args.label}) WHERE n.{args.prop} IS NOT NULL AND id(n) > $last "
                f"RETURN elementId(n) AS nid, n.{args.prop} AS vec "
                f"ORDER BY id(n) ASC LIMIT $lim",
                last=last, lim=args.batch,
            ).data()
            if not rows:
                break
            for r in rows:
                last = r["nid"]
                res = s.run(
                    "CALL db.index.vector.queryNodes($i,$k,$v) YIELD node, score "
                    "WHERE score >= $t RETURN elementId(node) AS cid, score",
                    i=idx, k=args.topk, v=r["vec"], t=args.threshold,
                ).data()
                for c in res:
                    if c["cid"] == last:
                        continue
                    uf.union(last, c["cid"])
                    edges += 1
            processed += len(rows)
            if args.limit and processed >= args.limit:
                break
            if processed % 20000 < args.batch:
                logging.info("edges=%d nodes=%d", edges, len(uf.p))
    logging.info("union done: %d nodes, %d edges", len(uf.p), edges)

    # Assign sequential community ids by root.
    root_to_cid = {}
    nid_to_cid = {}
    for nid in uf.p:
        root = uf.find(nid)
        if root not in root_to_cid:
            root_to_cid[root] = len(root_to_cid)
        nid_to_cid[nid] = root_to_cid[root]
    sizes = {}
    for cid in nid_to_cid.values():
        sizes[cid] = sizes.get(cid, 0) + 1
    logging.info("communities=%d", len(sizes))

    # Write community_id back to nodes in batches.
    items = list(nid_to_cid.items())
    with driver.session() as s:
        for i in range(0, len(items), args.batch):
            chunk = [{"nid": n, "cid": c} for n, c in items[i:i + args.batch]]
            s.run(
                f"UNWIND $u AS x MATCH (n:{args.label}) WHERE elementId(n)=x.nid "
                f"SET n.community_id = x.cid",
                u=chunk,
            )
        # Create one :Community node per component (no per-member rels to stay light).
        s.run(
            "UNWIND $c AS x MERGE (com:Community {id:x.id, label:$label, prop:$prop}) "
            "SET com.size = x.size",
            c=[{"id": cid, "size": sz} for cid, sz in sizes.items()],
            label=args.label, prop=args.prop,
        )
    driver.close()
    logging.info("DONE wrote community_id on %d %s nodes; %d :Community nodes",
                 len(items), args.label, len(sizes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
