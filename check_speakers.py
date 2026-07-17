#!/usr/bin/env python3
"""Check current speaker state in Neo4j (speaker counts, GlobalSpeaker linkage,
is_me coverage). Uses NEO4J_* env (default bolt://localhost:7687, db neo4j)."""
import os
from neo4j import GraphDatabase

URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
USER = os.environ.get("NEO4J_USER", "neo4j")
try:
    from auto_ingest_config import get_neo4j_password
    PASS = get_neo4j_password()
except Exception:
    PASS = os.environ.get("NEO4J_PASSWORD") or os.environ.get("NEO4J_PASSWORD_DEFAULT") or "knowledge_graph_2026"
DB = os.environ.get("NEO4J_DB", "neo4j")


def main() -> None:
    with GraphDatabase.driver(URI, auth=(USER, PASS)) as driver:
        with driver.session(database=DB) as s:
            # Speaker counts by label
            rows = s.run("""
                MATCH (sp:Speaker)
                WITH sp.label AS label, count(sp) AS cnt
                ORDER BY cnt DESC
                RETURN label, cnt
            """).data()
            print("=== Current Speaker labels (top 10) ===")
            for row in rows[:10]:
                print(f"  {row['label']}: {row['cnt']}")

            total = s.run("MATCH (sp:Speaker) RETURN count(sp) AS c").single()["c"]
            print(f"\n=== Total Speaker nodes: {total} ===")

            # GlobalSpeaker linkage via SAME_PERSON
            r = s.run("""
                MATCH (sp:Speaker)-[:SAME_PERSON]->(g:GlobalSpeaker)
                RETURN count(DISTINCT sp) AS linked, count(DISTINCT g) AS globals
            """).single()
            linked = r["linked"]; globals_ = r["globals"]
            print(f"\n=== Linked speakers (SAME_PERSON): {linked} ===")
            print(f"=== Distinct GlobalSpeakers: {globals_} ===")
            cov = (linked / total * 100) if total else 0.0
            print(f"=== Linkage coverage: {cov:.1f}% ===")

            me = s.run("MATCH (sp:Speaker{is_me:true}) RETURN count(sp) AS c").single()["c"]
            gs_me = s.run("MATCH (g:GlobalSpeaker{is_me:true}) RETURN count(g) AS c").single()["c"]
            print(f"\n=== Speaker is_me: {me} | GlobalSpeaker is_me: {gs_me} ===")


if __name__ == "__main__":
    main()
