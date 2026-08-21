#!/usr/bin/env python3
"""Semantic search over dashcam frames.

Embeds a natural-language query with the same multilingual-e5-large model used to
embed frame descriptions, then runs a vector search against
`DashcamFrame_emb_e5_large_index` (cosine). No e5 query/passage prefix is used,
matching the ingest convention in reembed.py.

Examples:
  python3 dashcam_ask.py "highway at night with oncoming headlights"
  python3 dashcam_ask.py "police car pulled over on the shoulder" --top-k 8 --view R
  python3 dashcam_ask.py "rain on the windshield" --json
"""
import os
import sys
import json
import argparse

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from auto_ingest.embed import EmbedModel
from neo4j import GraphDatabase

INDEX = "DashcamFrame_emb_e5_large_index"
EMB_MODEL = "intfloat/multilingual-e5-large"


def main():
    ap = argparse.ArgumentParser(description="Semantic search over DashcamFrame nodes.")
    ap.add_argument("query", help="natural-language query")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--view", choices=["F", "R", "FR"], help="filter by camera view")
    ap.add_argument("--json", action="store_true", help="emit JSON")
    ap.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    ap.add_argument("--neo4j-user", default="neo4j")
    ap.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD", "knowledge_graph_2026"))
    ap.add_argument("--embed-model", default=EMB_MODEL)
    args = ap.parse_args()

    model = EmbedModel(args.embed_model)
    vec = model.embed([args.query])[0]

    d = GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_password))
    try:
        with d.session() as s:
            if args.view:
                rows = s.run(
                    f"""CALL db.index.vector.queryNodes('{INDEX}', $k, $vec) YIELD node, score
                        WITH node, score WHERE node.view = $view
                        MATCH (c:DashcamClip {{key: node.key}})
                        RETURN node.key AS key, node.minute AS minute, node.view AS view,
                               node.description AS desc, node.timestamp AS ts,
                               c.path AS path, score
                        ORDER BY score DESC LIMIT $k""",
                    vec=vec, k=args.top_k, view=args.view,
                ).data()
            else:
                rows = s.run(
                    f"""CALL db.index.vector.queryNodes('{INDEX}', $k, $vec) YIELD node, score
                        MATCH (c:DashcamClip {{key: node.key}})
                        RETURN node.key AS key, node.minute AS minute, node.view AS view,
                               node.description AS desc, node.timestamp AS ts,
                               c.path AS path, score
                        ORDER BY score DESC LIMIT $k""",
                    vec=vec, k=args.top_k,
                ).data()
    finally:
        d.close()

    if args.json:
        print(json.dumps([
            {"key": r["key"], "minute": r["minute"], "view": r["view"],
             "score": round(r["score"], 4), "timestamp": str(r["ts"]) if r["ts"] else None,
             "path": r["path"], "description": r["desc"]}
            for r in rows
        ], indent=2))
        return

    if not rows:
        print("No frames found.")
        return
    for r in rows:
        ts = r["ts"].strftime("%Y-%m-%d %H:%M") if r["ts"] else "?"
        print(f"\n[score {r['score']:.3f}] {r['key']}  view={r['view']}  min={r['minute']}  {ts}")
        if r["path"]:
            print(f"  file: {r['path']}")
        print(f"  {r['desc']}")


if __name__ == "__main__":
    main()
