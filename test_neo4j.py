#!/usr/bin/env python3
"""Test Neo4j connection with different passwords."""
from neo4j import GraphDatabase

passwords = [
    "knowle...2026",
    "knowledge_graph_2026",
    "livelongandprosper",
]

for pw in passwords:
    try:
        drv = GraphDatabase.driver("bolt://100.64.43.123:7687", auth=("neo4j", pw))
        with drv.session() as s:
            r = s.run("RETURN 1 AS x").single()
            print(f"SUCCESS with password '{pw}': {r['x']}")
        drv.close()
    except Exception as e:
        print(f"FAILED with password '{pw}': {e}")
