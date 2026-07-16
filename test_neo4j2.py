#!/usr/bin/env python3
import os
from neo4j import GraphDatabase, Config
import sys

_pw = os.environ.get("NEO4J_PASSWORD") or os.environ.get("NEO4J_PASSWORD_DEFAULT") or "knowledge_graph_2026"
_db = os.environ.get("NEO4J_DB", "neo4j")

drv = GraphDatabase.driver(
    'bolt://100.64.43.123:7687', 
    auth=('neo4j', _pw),
    config=Config(tls_enabled=False)
)
with drv.session(database=_db) as s:
    r = s.run('RETURN 1 AS test').single()
    print("Result:", r, flush=True)

# Check databases
with drv.session(database=_db) as s:
    dbs = s.run('SHOW DATABASES YIELD name, currentStatus').data()
    for row in dbs[:5]:
        print(f"DB: {row['name']} - {row.get('currentStatus','?')}", flush=True)

drv.close()
print("DONE", flush=True)
