#!/usr/bin/env python3
"""Debug auth connection."""
import os, sys

print(f"NEO4J_URI={os.environ.get('NEO4J_URI', 'NOT SET')}", file=sys.stderr)
print(f"NEO4J_USER={os.environ.get('NEO4J_USER', 'NOT SET')}", file=sys.stderr)
print(f"NEO4J_PASSWORD={os.environ.get('NEO4J_PASSWORD', 'NOT SET')} (len={len(os.environ.get('NEO4J_PASSWORD', ''))})", file=sys.stderr)
print(f"NEO4J_DB={os.environ.get('NEO4J_DB', 'NOT SET')}", file=sys.stderr)

from neo4j import GraphDatabase

uri = os.environ['NEO4J_URI']
user = os.environ['NEO4J_USER']
password = os.environ['NEO4J_PASSWORD']
db = os.environ['NEO4J_DB']

print(f"\nConnecting to {uri} as {user}")
print(f"Password: '{password}' (len={len(password)})")
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

    print("\n✓ Connection successful!")
except Exception as e:
    print(f"\n✗ Auth/connection error: {e}")
    sys.exit(1)
finally:
    driver.close()
