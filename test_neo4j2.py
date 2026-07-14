#!/usr/bin/env python3
from neo4j import GraphDatabase, Config
import sys

drv = GraphDatabase.driver(
    'bolt://100.64.43.123:7687', 
    auth=('neo4j', 'knowledge_graph_2026'),
    config=Config(tls_enabled=False)
)
with drv.session(database='knowledge_graph_2026') as s:
    r = s.run('RETURN 1 AS test').single()
    print("Result:", r, flush=True)

# Check databases
with drv.session(database='knowledge_graph_2026') as s:
    dbs = s.run('SHOW DATABASES YIELD name, currentStatus').data()
    for row in dbs[:5]:
        print(f"DB: {row['name']} - {row.get('currentStatus','?')}", flush=True)

drv.close()
print("DONE", flush=True)
