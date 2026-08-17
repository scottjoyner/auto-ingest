#!/usr/bin/env python3
"""
crosskg_query.py — query the cross-instance entity link map produced by
link_entities_crosskg.py.

The two KGs live in separate Neo4j databases, so links are stored externally (sqlite)
rather than as native relationships. This tool reads that map:
  * given an entity id in KG A, list its matched entities in KG B (+ scores);
  * (--resolve) also connect to both KGs and print the source text of each node;
  * (--top N) list the N strongest cross-KG links overall.

Usage:
  crosskg_query.py --db crosskg_links.sqlite --a-id 12345
  crosskg_query.py --db crosskg_links.sqlite --top 20 --resolve \
      --kg-a-uri bolt://127.0.0.1:17687 --kg-b-uri bolt://100.64.43.123:7687
"""
from __future__ import annotations
import os
import sys
import sqlite3
import argparse
import logging
from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _label_prop(driver, label, nid):
    with driver.session() as s:
        r = s.run(
            f"MATCH (n:{label}) WHERE id(n)=$id "
            f"RETURN coalesce(n.text, n.name, n.title, n.label, n.display_label) AS t",
            id=nid,
        ).single()
        return (r["t"] if r and r["t"] is not None else f"<id {nid}>") if r else f"<missing {nid}>"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="crosskg_links.sqlite")
    ap.add_argument("--a-id", type=int, default=None, help="entity id in KG A to resolve")
    ap.add_argument("--top", type=int, default=0, help="list N strongest links overall")
    ap.add_argument("--kg-a-uri", default=os.getenv("NEO4J_URI", ""))
    ap.add_argument("--kg-b-uri", default="")
    ap.add_argument("--user", default=os.getenv("NEO4J_USER", "neo4j"))
    ap.add_argument("--password", default=os.getenv("NEO4J_PASSWORD", "knowledge_graph_2026"))
    ap.add_argument("--resolve", action="store_true", help="connect to both KGs and print node text")
    args = ap.parse_args()

    if not os.path.exists(args.db):
        logging.error("map %s not found; run link_entities_crosskg.py first", args.db)
        return 1
    con = sqlite3.connect(args.db)

    if args.a_id is not None:
        rows = con.execute(
            "SELECT b_label, b_id, score FROM links WHERE a_id=? ORDER BY score DESC",
            (args.a_id,),
        ).fetchall()
        print(f"KG-A entity id={args.a_id} -> {len(rows)} KG-B matches:")
        da = db = None
        if args.resolve and args.kg_a_uri:
            da = GraphDatabase.driver(args.kg_a_uri, auth=(args.user, args.password))
            print(f"  A: {_label_prop(da, 'Entity', args.a_id)}")
        if args.resolve and args.kg_b_uri:
            db = GraphDatabase.driver(args.kg_b_uri, auth=(args.user, args.password))
        for b_label, b_id, score in rows:
            txt = _label_prop(db, b_label, b_id) if (args.resolve and db) else ""
            print(f"  -> {b_label} id={b_id} score={score:.3f}  {txt}")
        if da:
            da.close()
        if db:
            db.close()
        return 0

    if args.top:
        rows = con.execute(
            "SELECT a_id, b_id, score FROM links ORDER BY score DESC LIMIT ?", (args.top,)
        ).fetchall()
        print(f"top {len(rows)} cross-KG links:")
        for a_id, b_id, score in rows:
            print(f"  A:{a_id} <-> B:{b_id}  score={score:.3f}")
        return 0

    # Default: summary stats.
    n = con.execute("SELECT count(*) FROM links").fetchone()[0]
    maxs = con.execute("SELECT max(score), min(score) FROM links").fetchone()
    logging.info("map %s: %d links  score range [%.3f, %.3f]", args.db, n, maxs[1] or 0, maxs[0] or 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
