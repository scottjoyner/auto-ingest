#!/usr/bin/env python3
"""Quick auth check before running the full linker."""
import os, sys

os.environ['NEO4J_URI'] = 'bolt://100.64.43.123:7687'
os.environ['NEO4J_USER'] = 'neo4j'
os.environ['NEO4J_PASSWORD'] = '***'  # will be set by env
os.environ['NEO4J_DB'] = 'knowledge_graph_2026'

from neo4j import GraphDatabase

uri = os.environ['NEO4J_URI']
user = os.environ['NEO4J_USER']
password = os.environ['NEO4J_PASSWORD']
db = os.environ['NEO4J_DB']

print(f"Connecting to {uri} as {user}")
print(f"Password: {password[:3]}*** (len={len(password)})")
print(f"Database: {db}")

driver = GraphDatabase.driver(uri, auth=(user, password))
try:
    with driver.session(database=db) as session:
        result = session.run("MATCH (s:Speaker) RETURN count(s) AS total")
        print(f"\nTotal Speaker nodes: {result.single()['total']}")

        result2 = session.run("""
            MATCH (g:GlobalSpeaker)<-[:LINKED_TO]-(s:Speaker) 
            WHERE g.elementId IN ['global_speaker_0', 'global_speaker_1'] 
            RETURN count(DISTINCT s) AS linked_sample
        """)
        print(f"Sample linked speakers: {result2.single()['linked_sample']}")

        result3 = session.run("""
            MATCH (g:GlobalSpeaker) 
            OPTIONAL MATCH (s:Speaker)-[:LINKED_TO]->(g)
            RETURN count(DISTINCT g) AS total_global, 
                   count(DISTINCT s) AS linked_to_any
        """)
        row = result3.single()
        print(f"Total GlobalSpeakers: {row['total_global']}")
        print(f"Linked to any GlobalSpeaker: {row['linked_to_any']}")

    print("\n✓ Connection successful!")
except Exception as e:
    print(f"\n✗ Auth/connection error: {e}")
    sys.exit(1)
finally:
    driver.close()
