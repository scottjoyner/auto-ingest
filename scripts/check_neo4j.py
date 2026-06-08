#!/usr/bin/env python3
import neo4j
from neo4j import GraphDatabase

uri = 'bolt://localhost:7687'
user = 'neo4j'
passwd = '${NEO4J_PASSWORD}'

driver = GraphDatabase.driver(uri, auth=(user, passwd))
with driver.session() as session:
    # Count Sophia events
    result = session.run("MATCH (e:SophiaEvent) RETURN count(e) as total")
    for record in result:
        print(f"Total Sophia events: {record['total']}")
    
    # Count memories with high confidence
    result = session.run("MATCH (m:Memory) WHERE m.confidence >= 0.7 RETURN count(m) as total")
    for record in result:
        print(f"High-confidence memories: {record['total']}")
    
    # Count unprocessed transcriptions
    result = session.run("MATCH (t:Transcription) WHERE t.processed = false OR t.knowledge_added = false RETURN count(t) as total")
    for record in result:
        print(f"Unprocessed transcriptions: {record['total']}")

driver.close()
