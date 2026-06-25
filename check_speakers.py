#!/usr/bin/env python3
"""Check current speaker state in Neo4j."""
from neo4j import GraphDatabase

uri = "bolt://100.64.43.123:7687"
auth = ("neo4j", "livelongandprosper")

with GraphDatabase.driver(uri, auth=auth) as driver:
    with driver.session(database="knowledge_graph_2026") as session:
        # Speaker counts by label
        r = session.run("""
            MATCH (s:Speaker)
            WITH s.label AS label, count(s) AS cnt
            ORDER BY cnt DESC
            RETURN label, cnt
        """).data()

        print("=== Current Speaker labels ===")
        for row in r[:10]:
            print(f"  {row['label']}: {row['cnt']}")

        # Total speaker count
        total = session.run("MATCH (s:Speaker) RETURN count(s)").single()["count(s)"]
        print(f"\n=== Total Speaker nodes: {total} ===")

        # GlobalSpeaker linkage
        r2 = session.run("""
            MATCH (s:Speaker)-[r:SPOKEN_BY]->(g:GlobalSpeaker)
            RETURN count(DISTINCT s) AS linked_speakers, 
                   count(g) AS global_speaker_refs
        """).data()

        print(f"\n=== Linked speakers: {r2[0]['linked_speakers']} ===")
        print(f"=== GlobalSpeaker refs: {r2[0]['global_speaker_refs']} ===")

        # Coverage percentage
        coverage = r2[0]['linked_speakers'] / total * 100 if total else 0
        print(f"=== Coverage: {coverage:.1f}% ===")
