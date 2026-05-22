from __future__ import annotations
import json, sqlite3
from datetime import datetime

class GraphOutbox:
    def __init__(self, sqlite_path:str):
        self.conn = sqlite3.connect(sqlite_path, check_same_thread=False)
        self.conn.execute('''CREATE TABLE IF NOT EXISTS graph_outbox(id INTEGER PRIMARY KEY, op_type TEXT, payload TEXT, created_at TEXT, retry_count INTEGER DEFAULT 0, last_error TEXT)''')
        self.conn.commit()
    def append(self, op_type:str, payload:dict, last_error:str=""):
        self.conn.execute("INSERT INTO graph_outbox(op_type,payload,created_at,last_error) VALUES(?,?,?,?)", (op_type, json.dumps(payload), datetime.utcnow().isoformat(), last_error))
        self.conn.commit()
    def pending(self, limit:int=100):
        return self.conn.execute("SELECT id,op_type,payload,retry_count FROM graph_outbox ORDER BY id LIMIT ?", (limit,)).fetchall()
    def mark_done(self, id_:int):
        self.conn.execute("DELETE FROM graph_outbox WHERE id=?", (id_,)); self.conn.commit()
    def mark_failed(self, id_:int, err:str):
        self.conn.execute("UPDATE graph_outbox SET retry_count=retry_count+1,last_error=? WHERE id=?", (err,id_)); self.conn.commit()
