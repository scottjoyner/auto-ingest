#!/usr/bin/env python3
"""
serve_app.py — a realtime, single-page search UI over the LOCAL graph.

No new dependencies: stdlib http.server + htmx (loaded from a CDN), Jinja-less
inline HTML. Query embedding reuses auto_ingest.embed so server-side search
vectors are byte-identical to ingest/re-embed vectors. Results stream back and
the worker log is tailed via Server-Sent Events so the page stays live while
new footage is being processed.

Run:
    ./.venv/bin/python3 serve_app.py            # 127.0.0.1:8090
    ./.venv/bin/python3 serve_app.py --port 80  # reverse-proxy behind nginx

Routes
------
GET /                  search page (htmx)
GET /api/search?q=..&k=10   top-k utterance hits as JSON
GET /api/media?file=..&start_ms=..   byte-range audio clip (play from hit)
GET /api/health         backend + model + graph counts
GET /api/ingest/stream  SSE stream tailing logs/pipeline.log
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import posixpath
import signal
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, unquote, urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from auto_ingest_config import get_neo4j_config, get_fileserver_root
from auto_ingest.backend import backend_info, has_rocm, gpu_target_machine
from auto_ingest.embed import get_default_model, embed_texts
from neo4j import GraphDatabase

DEFAULT_TOP_K = 25
# Which utterance vector index to query. Defaults to the live MiniLM-L6 index;
# switch to the gte-small re-embed by starting the server with
# VECTOR_PROP=emb_gte_small (INDEX_NAME resolves to <prop>_index).
EMBED_PROP = os.getenv("VECTOR_PROP", "embedding")
INDEX_NAME = f"{EMBED_PROP}_index"
_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
# SSE tailed logs (sweep worker + active re-embed). New lines from any are
# forwarded to the live page with a short tag so you see ingest + re-embed
# progress in one real-time feed.
SSE_LOGS = [
    os.path.join(_LOG_DIR, "pipeline.log"),
    os.path.join(_LOG_DIR, "reembed_gte.log"),
]
SSE_POLL_INTERVAL = 0.5

# --- Neo4j driver (module-level singleton) ---
_cfg = get_neo4j_config()
_driver = GraphDatabase.driver(_cfg["uri"], auth=(_cfg["user"], _cfg["password"]), database=_cfg.get("database"))


def db_count() -> int:
    try:
        with _driver.session() as s:
            from neo4j import Query
            res = s.run(Query("MATCH (n) RETURN count(n)", timeout=3.0))
            return int(res.single().values()[0])
    except Exception:
        return -1


def search(text: str, k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
    """Vector search utterances, joined to their Transcription for media path."""
    if not text.strip():
        return []
    qvec = embed_texts([text])[0]
    cypher = f"""
    CALL db.index.vector.queryNodes('{INDEX_NAME}', $k, $qvec)
    YIELD node AS u, score
    WHERE 'Utterance' IN labels(u)
    MATCH (u)<-[:HAS_UTTERANCE]-(t:Transcription)
    WITH u, score, t,
         coalesce(u.absolute_start, t.started_at) AS started,
         coalesce(u.absolute_end,   t.ended_at)   AS ended,
         t.source_media AS source_media
    RETURN
      u.text AS text,
      score,
      t.key AS transcript,
      source_media,
      started.epochMillis AS start_ms,
      ended.epochMillis AS end_ms,
      toFloat(u.start) AS start_rel,
      toFloat(u.end) AS end_rel
    ORDER BY score DESC
    LIMIT $k
    """
    with _driver.session() as s:
        rows = s.run(cypher, k=int(k), qvec=list(qvec)).data()
    hits = []
    for r in rows:
        src = r["source_media"] or ""
        hits.append({
            "text": (r["text"] or "")[:320],
            "score": round(float(r["score"]), 4),
            "transcript": r["transcript"],
            "source_media": src,
            "start_ms": r["start_ms"],
            "end_ms": r["end_ms"],
            "start_rel": r["start_rel"],
            "end_rel": r["end_rel"],
            "media_url": f"/api/media?file={src}" + (f"#t={r['start_rel'] or 0}" if r["start_rel"] else ""),
        })
    return hits


def serve_range(path: str, range_header: Optional[str]):
    """Serve `path` honoring an HTTP Range header (bytes).

    Returns (status, headers, open_file, size). The caller streams `open_file`
    from `start` to `end`. A missing/unsatisfiable Range yields the whole file
    (200) or 416. This lets the <audio> element issue byte-range requests so
    the browser can seek without us guessing ms->byte mappings for compressed
    audio (mp3). Seeking is handled by the client's `#t=<seconds>` fragment.
    """
    if not path or not os.path.isfile(path):
        return HTTPStatus.NOT_FOUND, {"Content-Type": "text/plain"}, None, 0, 0, 0
    size = os.path.getsize(path)
    ctype, _ = mimetypes.guess_type(path)
    ctype = ctype or "application/octet-stream"
    start, end = 0, size - 1
    if range_header and range_header.strip().lower().startswith("bytes="):
        spec = range_header.split("=", 1)[1].strip()
        try:
            sp, ep = spec.split("-", 1)
            start = int(sp) if sp else 0
            end = int(ep) if ep else size - 1
        except ValueError:
            start, end = 0, size - 1
        if start > size - 1 or end > size - 1 or start > end:
            return HTTPStatus.RANGE_NOT_SATISFIABLE, {"Content-Type": "text/plain", "Content-Range": f"bytes */{size}"}, None, size, 0, 0
    length = end - start + 1
    headers = {
        "Content-Type": ctype,
        "Accept-Ranges": "bytes",
        "Accept-Length": str(length),
        "Content-Length": str(length),
        "Content-Range": f"bytes {start}-{end}/{size}",
    }
    f = open(path, "rb")
    f.seek(start)
    status = HTTPStatus.PARTIAL_CONTENT if range_header else HTTPStatus.OK
    return status, headers, f, size, start, end


PAGE_HTML = r"""<!doctype html>
<html lang=en>
<head>
  <meta charset=utf-8>
  <meta name=viewport content="width=device-width,initial-scale=1">
  <title>LOCAL Search</title>
  <script src="https://unpkg.com/htmx.org@1.9.12"></script>
  <script src="https://cdn.tailwindcss.com"></script>  <style>
    body{font-family:sans-serif;margin:0;padding:16px;background:#0f1115;color:#e5e7eb}
    .card{background:#1e2028;border:1px solid #2e323a;border-radius:10px;padding:14px}
    .hit{border-bottom:1px solid #2e323a;padding:10px 0}
    .hit:last-child{border-bottom:none}
    .score{background:#2563eb;color:#fff;padding:1px 6px;border-radius:4px;font-size:11px}
    #q{width:100%;font-size:16px;padding:10px;border-radius:8px;border:1px solid #2e323a;background:#16181f;color:#e5e7eb}
    button{background:#2563eb;color:#fff;border:0;padding:10px 16px;border-radius:8px;cursor:pointer}
    button:hover{background:#1d57b8}
    #log{height:96px;overflow:auto;font-family:monospace;font-size:11px;background:#111318;border:1px solid #2e323a;border-radius:8px;padding:8px;resize:none;width:100%;color:#94a3b8}
    audio{width:100%;margin-top:6px}
    .sse{border:1px solid #2e323a;border-radius:8px;padding:10px;background:#111318}
  </style>
  </head>
<body>
  <div class="card mb-3">
    <h1 class="text-lg font-bold mb-2">LOCAL graph search</h1>
    <form id="sf" hx-get="/api/search" hx-trigger="keyup delay:350ms, change" hx-vars="q: q.value" hx-target="#hits" hx-swap="innerHTML">
      <input id="q" name="q" type="text" size="40" placeholder="Ask about anything you recorded… e.g. ‘quarterly numbers’, ‘router firmware’, ‘beach house’">
      <button type="submit">Search</button>
    </form>
  </div>

  <div id="hits" class="mb-4"></div>

  <div class="card">
    <h2 class="text-sm font-semibold mb-1 text-slate-400">Live ingest log</h2>
    <div class="sse">
      <pre id="log" class="text-xs text-slate-400"></pre>
    </div>
  </div>

  <script>
  function fetchResults(q){
    if(!q) return;
    fetch('/api/search?q='+encodeURIComponent(q)+'&k=25').then(r=>r.json()).then(render).catch(()=>{});
  }
  function render(hits){
    const box=document.getElementById('hits');
    if(!hits.length){ box.innerHTML='<p class=text-sm text-slate-400>no results</p>'; return; }
    box.innerHTML = hits.map(h => \`
      <div class="hit">
        <div class="flex gap-2"><span class="score">\${h.score.toFixed(3)}</span><b class="text-sm">\${h.transcript||'?'}</b></div>
        <div class="text-sm text-slate-300 mt-1">\${h.text}</div>
        \${h.media_url ? '<audio controls preload=none src="'+h.media_url+'"></audio>' : ''}
      </div>\`).join('');
  }
  document.getElementById('q').addEventListener('input', e => fetchResults(e.target.value));
  document.getElementById('sf').addEventListener('submit', e => { e.preventDefault(); fetchResults(document.getElementById('q').value); });
  // SSE: tail pipeline.log in real time
  let src=null;
  function sse(){
    src=new EventSource('/api/ingest/stream');
    src.onmessage=function(e){
      const el=document.getElementById('log');
      if(el){ el.textContent += e.data + '\\n'; el.scrollTop = el.scrollHeight; }
    };
    src.onerror=function(){ if(src){ src.close(); } setTimeout(sse, 2000); };
  }
  sse();
  document.getElementById('q').focus();
  </script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence default stderr noise
        pass

    def _send(self, status: HTTPStatus, body: bytes, ctype: str = "text/html", extra: Optional[Dict] = None):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if extra:
            for k, v in extra.items():
                self.send_header(k, str(v))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, status: HTTPStatus, obj: Any):
        body = json.dumps(obj, default=str).encode()
        self._send(status, body, "application/json")

    def do_GET(self):
        parsed = urlparse(self.path)
        path, qs = parsed.path, parse_qs(parsed.query)
        if path == "/":
            self._send(HTTPStatus.OK, PAGE_HTML.encode())
        elif path == "/api/health":
            bi = backend_info()
            self._json(HTTPStatus.OK, {
                "backend": bi["backend"], "torch_device": bi["torch_device"],
                "has_rocm": has_rocm(),
                "gpu_target": gpu_target_machine().get("name") if gpu_target_machine() else None,
                "embed_model": get_default_model().name, "embed_dim": get_default_model().dim,
                "nodes": db_count(),
            })
        elif path == "/api/search":
            q = (qs.get("q", [""])[0] or "").strip()
            k = int(qs.get("k", [str(DEFAULT_TOP_K)])[0])
            k = max(1, min(k, 200))
            self._json(HTTPStatus.OK, search(q, k))
        elif path == "/api/media":
            raw = unquote(qs.get("file", [""])[0] or "")
            rng = self.headers.get("Range")
            status, headers, fh, size, start, end = serve_range(raw, rng)
            if status == HTTPStatus.NOT_FOUND:
                self._send(HTTPStatus.NOT_FOUND, b"not found")
                return
            self.send_response(int(status))
            for k, v in headers.items():
                self.send_header(k, str(v))
            self.end_headers()
            try:
                if fh is None:
                    self.wfile.write(b"")
                    return
                remaining = end - start + 1
                while remaining > 0:
                    chunk = fh.read(min(1 << 16, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
            finally:
                fh.close()
        elif path == "/api/ingest/stream":
            self._sse_tail()
        else:
            self._send(HTTPStatus.NOT_FOUND, b"not found")

    def _sse_tail(self):
        """Long-lived SSE: tail every file in SSE_LOGS, tagging each line by
        source so the page shows the sweep + re-embed progress in one stream.
        Runs on its own thread (ThreadingHTTPServer)."""
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        # Track each log with a byte offset + incomplete-line buffer. We do NOT
        # use readline() (CPython's BufferedReader caches EOF on a file being
        # written by another process, so it returns '' even after appends).
        # Reading by size delta + splitting on newlines is the robust tail -f.
        tails = []
        for lf in SSE_LOGS:
            try:
                f = open(lf, "rb")
                f.seek(0, os.SEEK_END)
                tails.append({"path": lf, "fd": f, "off": f.tell(), "buf": b"",
                              "tag": os.path.basename(lf).replace(".log", "")})
            except FileNotFoundError:
                pass
        if not tails:
            self._sse_send("[system] (no logs found yet — has a worker run?)")
            while True:
                time.sleep(SSE_POLL_INTERVAL)
            return
        try:
            while True:
                any_line = False
                for t in tails:
                    f = t["fd"]
                    try:
                        size = os.fstat(f.fileno()).st_size
                    except OSError:
                        size = t["off"]
                    if size <= t["off"]:
                        continue
                    f.seek(t["off"])
                    chunk = f.read(size - t["off"])
                    t["off"] = f.tell()
                    t["buf"] += chunk
                    while b"\n" in t["buf"]:
                        line, t["buf"] = t["buf"].split(b"\n", 1)
                        self._sse_send(f"[{t['tag']}] {line.decode(errors='replace')}")
                        any_line = True
                if not any_line:
                    time.sleep(SSE_POLL_INTERVAL)
        except Exception:
            pass
        finally:
            for t in tails:
                try:
                    t["fd"].close()
                except Exception:
                    pass

    def _sse_send(self, data: str):
        payload = f"data: {data}\n\n".encode()
        try:
            self.wfile.write(payload)
            self.wfile.flush()
        except Exception:
            raise

    def do_HEAD(self):
        self.do_GET()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--log", default=None,
                    help="optional extra log file to tail via SSE (added to SSE_LOGS)")
    args = ap.parse_args()
    if args.log:
        SSE_LOGS.append(os.path.abspath(args.log))

    # Force the embedding model to load once at startup (and warm any GPU).
    m = get_default_model()
    print(f"LOCAL search server on http://{args.host}:{args.port}  "
          f"(backend={backend_info()['backend']} device={m.device} "
          f"model={m.name} dim={m.dim} rocm={has_rocm()})")
    server = ThreadingHTTPServer((args.host, args.port), Handler)

    def shutdown(signum, frame):
        server.shutdown()
    signal.signal(signal.SIGTERM, shutdown)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
