#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, io, json, mimetypes
from typing import List, Dict, Any, Optional
from flask import Flask, request, render_template_string, send_file, abort, url_for, jsonify
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from neo4j import GraphDatabase

# ---------------------------
# Config
# ---------------------------
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "livelongandprosper")
NEO4J_DB = os.getenv("NEO4J_DB", "neo4j")

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_DIM = int(os.getenv("EMBED_DIM", "384"))
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "32"))

# IMPORTANT: only serve audio from within this base dir
MEDIA_BASE = os.getenv("MEDIA_BASE", "/mnt/8TB_2025/fileserver/audio")

# ANN index names (must exist)
IDX_TRANS_V1 = os.getenv("IDX_TRANS_V1", "transcription_embedding_index")
IDX_TRANS_V2 = os.getenv("IDX_TRANS_V2", "transcription_embedding_v2_index")
IDX_SEGMENT  = os.getenv("IDX_SEGMENT", "segment_embedding_index")
IDX_UTTER    = os.getenv("IDX_UTTER", "utterance_embedding_index")

# ---------------------------
# Model (load once)
# ---------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
model = AutoModel.from_pretrained(EMBED_MODEL_NAME).to(DEVICE).eval()

def _normalize(v: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(v, p=2, dim=1)

def _mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return (last_hidden_state * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)

def embed_texts(texts: List[str], batch_size: int = EMBED_BATCH, max_length: int = 512) -> List[List[float]]:
    if not texts: return []
    vecs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            enc = {k: v.to(DEVICE) for k,v in enc.items()}
            out = model(**enc)
            pooled = _mean_pooling(out.last_hidden_state, enc["attention_mask"])
            pooled = _normalize(pooled)
            vecs.extend(pooled.cpu().numpy().tolist())
    return vecs

# ---------------------------
# Neo4j
# ---------------------------
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def query_transcriptions_ab_with_best_utterance(qvec, k=10, m=200, mode="both"):
    with driver.session(database=NEO4J_DB) as sess:
        CYPHER = """// PARAMS: $qvec (float[]), $k (int), $m (int), $idx (string), $utt_idx (string)
                CALL {
                WITH $qvec AS q, $k AS k, $idx AS idx
                CALL db.index.vector.queryNodes(idx, k, q)
                YIELD node AS t, score AS t_score
                RETURN collect({t: t, t_score: t_score}) AS t_hits
                }

                CALL {
                WITH $qvec AS q, $m AS m, $utt_idx AS uidx
                CALL db.index.vector.queryNodes(uidx, m, q)
                YIELD node AS u, score AS u_score
                MATCH (t:Transcription)-[:HAS_UTTERANCE]->(u)
                OPTIONAL MATCH (u)-[:OF_SEGMENT]->(s:Segment)

                // collect speakers per utterance (NO nesting, aggregate once per u)
                OPTIONAL MATCH (u)-[:SPOKEN_BY]->(sp:Speaker)
                WITH t, u, u_score, s, collect(DISTINCT {id: sp.id, label: sp.label}) AS sp_list

                RETURN collect({
                    t_id: t.id, t_key: t.key,
                    u_id: u.id, u_text: u.text, u_start: u.start, u_end: u.end,
                    u_abs_start: u.absolute_start, u_abs_end: u.absolute_end,
                    s_id: s.id, s_idx: s.idx, s_start: s.start, s_end: s.end,
                    s_abs_start: s.absolute_start, s_abs_end: s.absolute_end,
                    speakers: sp_list,
                    u_score: u_score
                }) AS utt_hits
                }

                WITH t_hits, utt_hits
                UNWIND t_hits AS th
                WITH th, [uh IN utt_hits WHERE uh.t_id = th.t.id] AS utt_for_t
                UNWIND CASE WHEN size(utt_for_t) = 0 THEN [NULL] ELSE utt_for_t END AS uh
                WITH th, uh
                ORDER BY uh.u_score DESC
                WITH th, collect(uh)[0] AS best
                RETURN
                th.t.id                    AS t_id,
                th.t.key                   AS t_key,
                th.t.started_at            AS t_started_at,
                th.t.source_media          AS t_media,
                th.t.source_rttm           AS t_rttm,
                th.t_score                 AS t_score,
                best.u_id                  AS u_id,
                best.u_text                AS u_text,
                best.u_start               AS u_start,
                best.u_end                 AS u_end,
                best.u_abs_start           AS u_abs_start,
                best.u_abs_end             AS u_abs_end,
                best.speakers              AS speakers,     // [{id,label}]
                best.s_id                  AS s_id,
                best.s_idx                 AS s_idx,
                best.s_start               AS s_start,
                best.s_end                 AS s_end,
                best.s_abs_start           AS s_abs_start,
                best.s_abs_end             AS s_abs_end,
                best.u_score               AS u_score
                ORDER BY t_score DESC
                LIMIT $k;
                """

        out = {}
        if mode in ("both", "v1"):
            out["v1"] = sess.run(
                CYPHER, qvec=qvec, k=k, m=m, idx=IDX_TRANS_V1, utt_idx=IDX_UTTER
            ).data()
        else:
            out["v1"] = []

        if mode in ("both", "v2"):
            out["v2"] = sess.run(
                CYPHER, qvec=qvec, k=k, m=m, idx=IDX_TRANS_V2, utt_idx=IDX_UTTER
            ).data()
        else:
            out["v2"] = []

        return out


# ---------------------------
# Flask
# ---------------------------
app = Flask(__name__)

HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Transcript Search A/B (v1 vs v2)</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; }
    h1 { margin-bottom: 0.25rem; }
    form { margin: 1rem 0; }
    textarea, input[type=number] { width: 100%; box-sizing: border-box; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    .card { border: 1px solid #ddd; border-radius: 12px; padding: 12px; margin-bottom: 12px; }
    .tag { display:inline-block; background:#efefef; padding:2px 6px; border-radius:8px; margin-right:6px; font-size: 12px;}
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    .muted { color: #666; font-size: 12px; }
    table { width: 100%; border-collapse: collapse; }
    th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid #eee; vertical-align: top; }
    audio { width: 100%; margin-top: 8px; }
    .col h2 { margin-top: 0; }
  </style>
</head>
<body>
  <h1>Transcript Search A/B</h1>
  <div class="muted">Compare transcription vectors <strong>v1</strong> vs <strong>v2</strong>. Best utterance + speakers per hit. Audio playback if available.</div>
  <form method="POST" action="/search">
    <label>Query</label>
    <textarea name="q" rows="3" required>{{q or ""}}</textarea>
    <div style="display:grid; grid-template-columns: 120px 120px 160px 1fr; gap: 12px; margin-top: 8px;">
    <div>
        <label>Top K</label>
        <input type="number" name="k" min="1" max="100" value="{{k or 10}}">
    </div>
    <div>
        <label>Utter pool (m)</label>
        <input type="number" name="m" min="50" max="2000" value="{{m or 200}}">
    </div>
    <div>
        <label>Mode</label>
        <select name="mode">
        <option value="both" {{ 'selected' if mode=='both' else '' }}>v1 + v2</option>
        <option value="v1"   {{ 'selected' if mode=='v1'   else '' }}>v1 only</option>
        <option value="v2"   {{ 'selected' if mode=='v2'   else '' }}>v2 only</option>
        </select>
    </div>
    <div style="align-self:end;">
        <button type="submit">Search</button>
    </div>
    </div>
  </form>

  {% if results %}
  <div class="row">
    {% for variant in ['v2','v1'] %}
    <div class="col">
      <h2>{{ 'v2 (embedding_v2)' if variant=='v2' else 'v1 (embedding)' }}</h2>
      {% if results[variant]|length == 0 %}
        <div class="muted">No results.</div>
      {% endif %}
      {% for r in results[variant] %}
      <div class="card">
        <div><strong>{{ loop.index }}.</strong> <span class="mono">{{ r.t_key }}</span> &nbsp; <span class="muted">score {{ '%.4f'|format(r.t_score or 0) }}</span></div>
        <div class="muted">Transcription: <span class="mono">{{ r.t_id }}</span> &middot; started: {{ r.t_started_at or '—' }}</div>

        {% if r.u_text %}
          <div style="margin-top:8px;"><strong>Best utterance</strong> ({{ '%.2fs'|format(r.u_start or 0) }}–{{ '%.2fs'|format(r.u_end or 0) }}):</div>
          <div>{{ r.u_text }}</div>
          <div class="muted">Abs: {{ r.u_abs_start or '—' }} → {{ r.u_abs_end or '—' }}</div>
        {% endif %}

        {% if r.speakers and r.speakers|length > 0 %}
        <div style="margin-top:6px;">
            {% for sp in r.speakers %}
            <div class="tag" title="{{ sp.id }}">{{ sp.label }}</div>
            <form method="POST" action="/speaker/rename" style="display:inline-block; margin-left:6px;">
                <input type="hidden" name="speaker_id" value="{{ sp.id }}">
                <input type="text" name="new_label" placeholder="Rename {{ sp.label }}" style="width:160px;">
                <button type="submit">Save</button>
            </form>
            <br/>
            {% endfor %}
        </div>
        {% endif %}


        {% if r.t_media %}
          <details style="margin-top:8px;">
            <summary>Audio</summary>
            <audio controls preload="none" src="{{ url_for('serve_media', path=r.t_media) }}#t={{ (r.u_start or 0)|int }}"></audio>
            <div class="muted">{{ r.t_media }}</div>
          </details>
        {% endif %}
        {% if r.t_rttm %}
          <div class="muted">RTTM: {{ r.t_rttm }}</div>
        {% endif %}
      </div>
      {% endfor %}
    </div>
    {% endfor %}
  </div>
  {% endif %}
</body>
</html>
"""

@app.get("/")
def home():
    return render_template_string(HTML)

@app.post("/search")
def search():
    q = (request.form.get("q") or "").strip()
    k = int(request.form.get("k") or 10)
    m = int(request.form.get("m") or 200)
    mode = (request.form.get("mode") or "both").lower()
    if not q:
        return render_template_string(HTML, q=q, k=k, m=m, mode=mode, results=None)
    qvec = embed_texts([q])[0]
    results = query_transcriptions_ab_with_best_utterance(qvec, k=k, m=m, mode=mode)
    return render_template_string(HTML, q=q, k=k, m=m, mode=mode, results=results)

def _is_safe_media(path: str) -> bool:
    try:
        real = os.path.realpath(path)
        base = os.path.realpath(MEDIA_BASE)
        return real.startswith(base) and os.path.isfile(real)
    except Exception:
        return False

@app.get("/media")
def serve_media():
    # /media?path=/abs/path/to/file.wav
    path = request.args.get("path", "")
    if not _is_safe_media(path):
        abort(404)
    guessed = mimetypes.guess_type(path)[0] or "application/octet-stream"
    return send_file(path, mimetype=guessed, as_attachment=False, conditional=True)

@app.post("/speaker/rename")
def speaker_rename():
    sp_id = (request.form.get("speaker_id") or "").strip()
    new_label = (request.form.get("new_label") or "").strip()
    if not sp_id or not new_label:
        return jsonify({"ok": False, "error": "speaker_id and new_label required"}), 400

    cy = """
    MATCH (sp:Speaker {id:$id})
    SET sp.label = $label, sp.updated_at = datetime()
    WITH sp
    MATCH (s:Segment) WHERE s.speaker_id = $id
    SET s.speaker_label = $label
    RETURN sp.id AS id, sp.label AS label
    """
    with driver.session(database=NEO4J_DB) as sess:
        rec = sess.run(cy, id=sp_id, label=new_label).single()
        if not rec:
            return jsonify({"ok": False, "error": "Speaker not found"}), 404
        return jsonify({"ok": True, "id": rec["id"], "label": rec["label"]})

if __name__ == "__main__":
    # Flask dev server
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")), debug=True)
