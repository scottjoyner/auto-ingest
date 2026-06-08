from neo4j import GraphDatabase
import json

driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', '${NEO4J_PASSWORD}'))
with driver.session() as session:
    result = session.run("MATCH (e:SophiaVoiceUiEvent) WHERE e.ts_ms >= datetime().epochMillis - (24 * 60 * 60 * 1000) RETURN count(e) as total")
    for record in result:
        print(f"Total Sophia events (last 24h): {record['total']}")

driver.close()
