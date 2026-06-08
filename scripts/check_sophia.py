#!/usr/bin/env python3
import neo4j
from neo4j import GraphDatabase
from datetime import datetime, timedelta

uri = 'bolt://localhost:7687'
user = 'neo4j'
passwd = '${NEO4J_PASSWORD}'

driver = GraphDatabase.driver(uri, auth=(user, passwd))

print("=== SOPHIA EVENTS CHECK ===\n")

with driver.session() as session:
    # Check all Sophia events
    result = session.run("MATCH (e:SophiaVoiceUiEvent) RETURN count(e) as total")
    for record in result:
        print(f"Total SophiaVoiceUiEvent nodes: {record['total']}")
    
    # Check timestamps of recent events
    result = session.run("MATCH (e:SophiaVoiceUiEvent) RETURN e.ts_ms as ts ORDER BY ts DESC LIMIT 5")
    for record in result:
        print(f"Most recent event timestamp: {record['ts']}")
    
    # Count events from last 24 hours (epoch milliseconds)
    now = datetime.now()
    twenty_four_hours_ago = int((now - timedelta(hours=24)).timestamp() * 1000)
    result = session.run(f"MATCH (e:SophiaVoiceUiEvent) WHERE e.ts_ms > {twenty_four_hours_ago} RETURN count(e) as recent")
    for record in result:
        print(f"Events from last 24 hours: {record['recent']}")

driver.close()
