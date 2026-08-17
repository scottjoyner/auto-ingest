#!/usr/bin/env python3
"""
link_entities_crosskg.py — link Entity nodes ACROSS two separate Neo4j KGs.

The bodycam (deathstar) and research (x1-370) graphs live in different Neo4j databases,
so a native relationship cannot span them. Instead we build an EXTERNAL cross-reference
map (sqlite) of matched entity ids: for each Entity in KG A, ANN-query KG B's Entity vector
index and record pairs above --threshold. A downstream app reads this map to join across
KGs. Optionally stamp the matched B-ids onto the A node via --write-prop.

Run after BOTH KGs' Entity nodes are embedded with --prop.
"""
from __future__ import annotations
import os
import sys
import sqlite3
import argparse
import logging
from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg-a-uri", required=True)
    ap.add_argument("--kg-b-uri", required=True)
    ap.add_argument("--user", default=os.getenv("NEO4J_USER", "neo4j"))
    ap.add_argument("--password", default=os.getenv("NEO4J_PASSWORD", "knowledge_graph_2026"))
    ap.add_argument("--label", default="Entity")
    ap.add_argument("--prop", default="emb_e5_large")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.85)
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--db", default="crosskg_links.sqlite")
    ap.add_argument("--write-prop", default="", help="stamp matched B-ids onto A node under this prop")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    idx = f"{args.label}_{args.prop}_index"

    da = GraphDatabase.driver(args.kg_a_uri, auth=(args.user, args.password))
    db = GraphDatabase.driver(args.kg_b_uri, auth=(args.user, args.password))
    con = sqlite3.connect(args.db)
    con.execute(
        "CREATE TABLE IF NOT EXISTS links ("
        "a_label TEXT, a_id INTEGER, b_label TEXT, b_id INTEGER, score REAL, "
        "PRIMARY KEY (a_label, a_id, b_label, b_id))"
    )

    linked = 0
    with da.session() as sa, db.session() as sb:
        last = -1
        while True:
            rows = sa.run(
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
                res = sb.run(
                    "CALL db.index.vector.queryNodes($idx, $k, $vec) YIELD node, score "
                    "RETURN id(node) AS bid, score",
                    idx=idx, k=args.topk, vec=vec,
                ).data()
                bids = []
                for c in res:
                    if c["score"] >= args.threshold:
                        bid = c["bid"]
                        if args.dry_run:
                            logging.info("[dry] A:%d -> B:%d score=%.3f", last, bid, c["score"])
                        else:
                            con.execute(
                                "INSERT OR IGNORE INTO links VALUES (?,?,?,?,?)",
                                (args.label, last, args.label, bid, c["score"]),
                            )
                            bids.append(bid)
                            linked += 1
                if bids and args.write_prop and not args.dry_run:
                    sa.run(
                        f"MATCH (a:{args.label}) WHERE id(a)=$aid SET a.{args.write_prop}=$ids",
                        aid=last, ids=bids,
                    )
            con.commit()
            logging.info("processed up to A id %d, linked %d", last, linked)
    con.close()
    logging.info("DONE linked=%d -> %s", linked, args.db)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
