#!/usr/bin/env python3
import neo4j
from neo4j import GraphDatabase

uri = 'bolt://localhost:7687'
user = 'neo4j'
passwd = '${NEO4J_PASSWORD}'

driver = GraphDatabase.driver(uri, auth=(user, passwd))

print("=== SCHEMA CHECK ===\n")

# Check Transcription properties
with driver.session() as session:
    result = session.run("MATCH (t:Transcription) RETURN t LIMIT 1")
    for record in result:
        node = record['t']
        print(f"Transcription properties: {list(node.keys())}")
    
    # Check SophiaVoiceUiEvent properties  
    result = session.run("MATCH (e:SophiaVoiceUiEvent) RETURN e LIMIT 1")
    for record in result:
        node = record['e']
        print(f"SophiaVoiceUiEvent properties: {list(node.keys())}")
    
    # Check VoiceprintVersion properties
    result = session.run("MATCH (v:VoiceprintVersion) RETURN v LIMIT 1")
    for record in result:
        node = record['v']
        print(f"VoiceprintVersion properties: {list(node.keys())}")

driver.close()
