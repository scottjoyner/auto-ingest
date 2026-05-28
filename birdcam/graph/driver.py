from __future__ import annotations
class GraphDriver:
    def __init__(self, uri:str, username:str, password:str, database:str="neo4j"):
        from neo4j import GraphDatabase
        self._driver = GraphDatabase.driver(uri, auth=(username, password))
        self.database = database
    def execute(self, query:str, **params):
        with self._driver.session(database=self.database) as s:
            return list(s.run(query, **params))
    def close(self):
        self._driver.close()
