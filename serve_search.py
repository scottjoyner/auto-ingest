#!/usr/bin/env python3
"""
serve_search.py — HTTP service wrapping vector_search.py semantic search.

Exposes POST /search with a JSON body:
  {"query": "...", "target": "chunk", "topk": 10, "prop": "emb_e5_large",
   "kg2_uri": null}
and returns the same rows as `vector_search.py search-text`. The query embedding
model is loaded once (lazily) and reused across requests. Configure the KG via
NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD and the model via EMBED_MODEL_NAME.

Run on a GPU box:
  serve_search.py --host 0.0.0.0 --port 8080
"""
from __future__ import annotations
import os
import sys
import json
import argparse
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vector_search import search_text  # reuse the exact retrieval path

DEFAULT_TARGETS = {"chunk", "summary", "segment", "utterance", "transcription",
                   "entity", "concept", "topic", "keyword", "kgnode", "note",
                   "speaker", "globalspeaker"}


class Handler(BaseHTTPRequestHandler):
    driver = None  # set by main()

    def _send(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if urlparse(self.path).path != "/search":
            self._send(404, {"error": "only POST /search supported"})
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            self._send(400, {"error": f"bad request: {e}"})
            return
        query = (payload.get("query") or "").strip()
        if not query:
            self._send(400, {"error": "missing 'query'"})
            return
        target = payload.get("target", "chunk")
        if target not in DEFAULT_TARGETS:
            self._send(400, {"error": f"unknown target {target}"})
            return
        topk = min(int(payload.get("topk", 10)), 100)
        prop = payload.get("prop", "emb_e5_large")
        kg2 = payload.get("kg2_uri")
        try:
            rows = search_text(self.driver, query, target, topk, False, 200)
            if kg2:
                from neo4j import GraphDatabase
                import vector_search as vs
                d2 = GraphDatabase.driver(kg2, auth=(vs.NEO4J_USER, vs.NEO4J_PASSWORD))
                r2 = search_text(d2, query, target, topk, False, 200)
                for r in r2:
                    r["kg"] = "kg2"
                for r in rows:
                    r["kg"] = "kg1"
                rows = sorted(rows + r2, key=lambda x: x.get("score", 0.0), reverse=True)[:topk]
            for r in rows:
                r.pop("embedding", None)
            self._send(200, {"query": query, "target": target, "prop": prop, "results": rows})
        except Exception as e:
            logging.exception("search failed")
            self._send(500, {"error": str(e)})

    def log_message(self, *a):
        pass


def main() -> int:
    from neo4j import GraphDatabase
    import vector_search as vs
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.getenv("SEARCH_HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=int(os.getenv("SEARCH_PORT", "8080")))
    args = ap.parse_args()

    vs.NEO4J_URI = os.getenv("NEO4J_URI", vs.NEO4J_URI)
    vs.NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", vs.NEO4J_PASSWORD)
    Handler.driver = GraphDatabase.driver(vs.NEO4J_URI, auth=(vs.NEO4J_USER, vs.NEO4J_PASSWORD))
    logging.info("serve_search listening on %s:%d  (KG=%s)", args.host, args.port, vs.NEO4J_URI)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
