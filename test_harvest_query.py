from neo4j import GraphDatabase
import json

uri = 'bolt://localhost:7687'
user = 'neo4j'
passwd = '${NEO4J_PASSWORD}'

driver = GraphDatabase.driver(uri, auth=(user, passwd))
with driver.session() as session:
    query = "MATCH (e:SophiaVoiceUiEvent) WHERE e.ts_ms >= datetime().epochMillis - (24 * 60 * 60 * 1000) RETURN count(e) as total"
    result = session.run(query)
    records = [record.data() for record in result]
    print(json.dumps(records, default=str))

driver.close()
