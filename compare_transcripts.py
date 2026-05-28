#!/usr/bin/env python3
from auto_ingest_config import get_fileserver_path
# -*- coding: utf-8 -*-
import os, io, json, mimetypes, re
from typing import List, Dict, Any, Optional, Tuple
from flask import Flask, request, render_template_string, send_file, abort, url_for, jsonify, redirect
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

# IMPORTANT: only serve media from within this base dir
MEDIA_BASE = os.getenv("MEDIA_BASE", get_fileserver_path("audio"))

# ANN index names (must exist)
IDX_TRANS_V1 = os.getenv("IDX_TRANS_V1", "transcription_embedding_index")
IDX_TRANS_V2 = os.getenv("IDX_TRANS_V2", "transcription_embedding_v2_index")
IDX_SEGMENT  = os.getenv("IDX_SEGMENT", "segment_embedding_index")
IDX_UTTER    = os.getenv("IDX_UTTER", "utterance_embedding_index")
IDX_GLOBAL   = os.getenv("IDX_GLOBAL", "global_speaker_embedding_index")  # NEW: GlobalSpeaker vector index

# Suggestion defaults
SUGG_TOPK   = int(os.getenv("SUGG_TOPK", "5"))
SUGG_THRESH = float(os.getenv("SUGG_THRESH", "0.62"))
SUGG_LIMIT_SPEAKERS = int(os.getenv("SUGG_LIMIT_SPEAKERS", "200"))  # how many unclustered speakers to inspect per page

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

# Core A/B query – segment speaker fallback + globals + segment flags
def query_transcriptions_ab_with_best_utterance(qvec, k=10, m=200, mode="both"):
    with driver.session(database=NEO4J_DB) as sess:
        CYPHER = """
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

            // 1) Direct SPOKEN_BY
            OPTIONAL MATCH (u)-[:SPOKEN_BY]->(sp:Speaker)

            WITH t,u,u_score,s,sp,
                 CASE WHEN sp IS NOT NULL THEN sp.id ELSE s.speaker_id END AS sp_id,
                 CASE WHEN sp IS NOT NULL THEN sp.label ELSE s.speaker_label END AS sp_label,
                 s.is_lyrics AS is_lyrics, s.lyrics_score AS lyrics_score, s.review_needed AS review_needed

            // 2) Hop to GlobalSpeaker
            OPTIONAL MATCH (spx:Speaker {id:sp_id})-[:SAME_PERSON]->(gs:GlobalSpeaker)
            WITH t,u,u_score,s,sp_id,sp_label,is_lyrics,lyrics_score,review_needed,
                 collect(DISTINCT {gs_id: gs.id, status: gs.status, method: gs.method, display_label: gs.display_label}) AS globals

            RETURN collect({
                t_id: t.id, t_key: t.key,
                t_media: t.source_media, t_rttm: t.source_rttm, t_started_at: t.started_at,
                u_id: u.id, u_text: u.text, u_start: u.start, u_end: u.end,
                u_abs_start: coalesce(u.absolute_start, s.absolute_start),
                u_abs_end:   coalesce(u.absolute_end, s.absolute_end),
                s_id: s.id, s_idx: s.idx, s_start: s.start, s_end: s.end,
                is_lyrics: is_lyrics, lyrics_score: lyrics_score, review_needed: review_needed,
                speakers: CASE WHEN sp_id IS NOT NULL
                           THEN [{id: sp_id, label: sp_label, globals: globals}]
                           ELSE []
                           END,
                u_score: u_score
            }) AS utt_hits
        }

        WITH t_hits, utt_hits
        UNWIND t_hits AS th
        WITH th, [uh IN utt_hits WHERE uh.t_id = th.t.id] AS utt_for_t
        UNWIND CASE WHEN size(utt_for_t)=0 THEN [NULL] ELSE utt_for_t END AS uh
        WITH th, uh
        ORDER BY uh.u_score DESC
        WITH th, collect(uh)[0] AS best
        RETURN
            th.t.id            AS t_id,
            th.t.key           AS t_key,
            best.t_started_at  AS t_started_at,
            best.t_media       AS t_media,
            best.t_rttm        AS t_rttm,
            th.t_score         AS t_score,
            best.u_id          AS u_id,
            best.u_text        AS u_text,
            best.u_start       AS u_start,
            best.u_end         AS u_end,
            best.u_abs_start   AS u_abs_start,
            best.u_abs_end     AS u_abs_end,
            best.speakers      AS speakers,
            best.s_id          AS s_id,
            best.s_idx         AS s_idx,
            best.s_start       AS s_start,
            best.s_end         AS s_end,
            best.is_lyrics     AS is_lyrics,
            best.lyrics_score  AS lyrics_score,
            best.review_needed AS review_needed,
            best.u_score       AS u_score
        ORDER BY t_score DESC
        LIMIT $k;
        """

        out = {}
        if mode in ("both", "v1"):
            out["v1"] = sess.run(CYPHER, qvec=qvec, k=k, m=m, idx=IDX_TRANS_V1, utt_idx=IDX_UTTER).data()
        else:
            out["v1"] = []

        if mode in ("both", "v2"):
            out["v2"] = sess.run(CYPHER, qvec=qvec, k=k, m=m, idx=IDX_TRANS_V2, utt_idx=IDX_UTTER).data()
        else:
            out["v2"] = []
        return out

# ---------------------------
# Media helpers
# ---------------------------
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".flac"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv"}

def _is_safe_media(path: str) -> bool:
    try:
        real = os.path.realpath(path)
        base = os.path.realpath(MEDIA_BASE)
        return real.startswith(base) and os.path.isfile(real)
    except Exception:
        return False

def _stem_and_dir(path: str) -> Tuple[str, str]:
    if not path: return "", ""
    d, fn = os.path.split(path)
    stem, _ = os.path.splitext(fn)
    stem = re.sub(r"_(F|R)$", "", stem)
    return stem, d

def best_audio_for_media(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    if not _is_safe_media(path):
        return None
    ext = os.path.splitext(path)[1].lower()
    if ext in AUDIO_EXTS:
        return path
    if ext in VIDEO_EXTS:
        stem, d = _stem_and_dir(path)
        if stem and d:
            for cand_ext in (".mp3", ".wav", ".m4a"):
                cand = os.path.join(d, stem + cand_ext)
                if _is_safe_media(cand):
                    return cand
        return path
    return path if _is_safe_media(path) else None

# ---------------------------
# Flask
# ---------------------------
app = Flask(__name__)

BASE_CSS = """
  body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; }
  h1 { margin-bottom: 0.25rem; }
  form { margin: 1rem 0; }
  textarea, input[type=number], input[type=text], select { width: 100%; box-sizing: border-box; }
  .row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .card { border: 1px solid #ddd; border-radius: 12px; padding: 12px; margin-bottom: 12px; }
  .tag { display:inline-block; background:#efefef; padding:2px 6px; border-radius:8px; margin-right:6px; font-size: 12px;}
  .tag.lyrics { background:#e2f3ff; }
  .tag.warn { background:#fff3cd; }
  .tag.good { background:#e7f7e7; }
  .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
  .muted { color: #666; font-size: 12px; }
  table { width: 100%; border-collapse: collapse; }
  th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid #eee; vertical-align: top; }
  audio, video { width: 100%; margin-top: 8px; }
  .col h2 { margin-top: 0; }
  details > summary { cursor: pointer; }
  .inline-form { display:inline-block; margin-left:6px; }
  .topnav { display:flex; gap:8px; align-items:center; margin:8px 0 16px; }
  .topnav a { text-decoration:none; }
"""

HTML = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Transcript Search A/B (v1 vs v2)</title>
  <style>{BASE_CSS}</style>
</head>
<body>
  <div class="topnav">
    <h1 style="margin:0;">Transcript Search A/B</h1>
    <a class="tag" href="{{ url_for('suggest_page') }}">Suggestions (Speakers → Global)</a>
  </div>
  <div class="muted">Compare transcription vectors <strong>v1</strong> vs <strong>v2</strong>. Best utterance + speakers/clusters per hit. Audio playback prefers WAV/MP3.</div>
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

        <div style="margin-top:6px;">
          {% if r.is_lyrics %}<div class="tag lyrics" title="is_lyrics=true">Lyrics</div>{% endif %}
          {% if r.review_needed %}<div class="tag warn" title="review_needed=true">Needs review</div>{% endif %}
          {% if r.lyrics_score is not none %}<div class="tag" title="lyrics_score">score {{ '%.2f'|format(r.lyrics_score) }}</div>{% endif %}
          {% if r.s_idx is not none %}<div class="tag">seg#{{ r.s_idx }}</div>{% endif %}
        </div>

        {% if r.speakers and r.speakers|length > 0 %}
          <div style="margin-top:6px;">
            {% for sp in r.speakers %}
              <div class="tag" title="{{ sp.id }}">Speaker: {{ sp.label or sp.id }}</div>
              {% if sp.globals and sp.globals|length > 0 %}
                {% for g in sp.globals %}
                  <div class="tag good" title="{{ g.gs_id }} ({{ g.method or '—' }})">Global: {{ g.display_label or g.gs_id }}</div>
                {% endfor %}
              {% else %}
                <div class="tag warn">Unclustered</div>
              {% endif %}
              <form method="POST" action="/speaker/rename" class="inline-form">
                <input type="hidden" name="speaker_id" value="{{ sp.id }}">
                <input type="text" name="new_label" placeholder="Rename {{ sp.label or sp.id }}" style="width:160px;">
                <button type="submit">Save</button>
              </form>
              <form method="POST" action="/speaker/promote_to_global" class="inline-form">
                <input type="hidden" name="speaker_id" value="{{ sp.id }}">
                <input type="text" name="display_label" placeholder="New global label (optional)" style="width:180px;">
                <button type="submit">Promote→Global</button>
              </form>
              <a class="inline-form tag" href="{{ url_for('suggest_page', seed=sp.id) }}">Suggest clusters</a>
              <br/>
            {% endfor %}
          </div>
        {% endif %}

        {% if r.t_media %}
          <details style="margin-top:8px;">
            <summary>Media</summary>
            <div class="muted">{{ r.t_media }}</div>
            <div>
              <a class="tag" href="{{ url_for('transcription_full', t_id=r.t_id) }}">Open full transcript →</a>
            </div>
            {% set play_url = url_for('serve_media', path=(best_audio_for_media(r.t_media) or r.t_media)) %}
            {% if play_url.endswith('.mp3') or play_url.endswith('.wav') or play_url.endswith('.m4a') or play_url.endswith('.aac') or play_url.endswith('.flac') %}
              <audio controls preload="none" src="{{ play_url }}#t={{ (r.u_start or 0)|int }}"></audio>
            {% else %}
              <video controls preload="metadata" src="{{ play_url }}#t={{ (r.u_start or 0)|int }}"></video>
            {% endif %}
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

@app.context_processor
def _inject_helpers():
    # Expose the helper function directly
    return dict(best_audio_for_media=best_audio_for_media)


@app.get("/")
def home():
    return render_template_string(HTML, q=None, k=10, m=200, mode="both", results=None)

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

# ---------- MEDIA ----------
@app.get("/media")
def serve_media():
    path = request.args.get("path", "")
    if not _is_safe_media(path):
        abort(404)
    guessed = mimetypes.guess_type(path)[0] or "application/octet-stream"
    return send_file(path, mimetype=guessed, as_attachment=False, conditional=True)

# ---------- LABELING & CLUSTERS (existing) ----------
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

@app.post("/global_speaker/rename")
def global_speaker_rename():
    gs_id = (request.form.get("global_id") or "").strip()
    new_label = (request.form.get("display_label") or "").strip()
    if not gs_id or not new_label:
        return jsonify({"ok": False, "error": "global_id and display_label required"}), 400
    cy = """
    MATCH (g:GlobalSpeaker {id:$id})
    SET g.display_label = $label, g.updated_at = datetime()
    WITH g
    OPTIONAL MATCH (sp:Speaker)-[:SAME_PERSON]->(g)
    SET sp.label = coalesce(sp.label, $label)
    WITH g, collect(sp.id) AS sp_ids
    MATCH (s:Segment) WHERE s.speaker_id IN sp_ids
    SET s.speaker_label = coalesce(s.speaker_label, $label)
    RETURN g.id AS id, g.display_label AS label, sp_ids
    """
    with driver.session(database=NEO4J_DB) as sess:
        rec = sess.run(cy, id=gs_id, label=new_label).single()
        if not rec:
            return jsonify({"ok": False, "error": "GlobalSpeaker not found"}), 404
        return jsonify({"ok": True, "id": rec["id"], "label": rec["label"], "affected_speakers": rec["sp_ids"]})

@app.post("/speaker/attach_global")
def speaker_attach_global():
    sp_id = (request.form.get("speaker_id") or "").strip()
    gs_id = (request.form.get("global_id") or "").strip()
    if not sp_id or not gs_id:
        return jsonify({"ok": False, "error": "speaker_id and global_id required"}), 400
    cy = """
    MATCH (sp:Speaker {id:$sp_id})
    MATCH (g:GlobalSpeaker {id:$gs_id})
    MERGE (sp)-[:SAME_PERSON]->(g)
    SET sp.updated_at = datetime(), g.updated_at = datetime()
    RETURN sp.id AS sp_id, g.id AS gs_id, g.display_label AS label
    """
    with driver.session(database=NEO4J_DB) as sess:
        rec = sess.run(cy, sp_id=sp_id, gs_id=gs_id).single()
        if not rec:
            return jsonify({"ok": False, "error": "Failed to attach"}), 400
        return jsonify({"ok": True, "speaker_id": rec["sp_id"], "global_id": rec["gs_id"], "global_label": rec["label"]})

@app.post("/speaker/promote_to_global")
def speaker_promote_to_global():
    sp_id = (request.form.get("speaker_id") or "").strip()
    display_label = (request.form.get("display_label") or "").strip()
    if not sp_id:
        return jsonify({"ok": False, "error": "speaker_id required"}), 400
    cy = """
    MERGE (sp:Speaker {id:$sp_id})
    ON MATCH SET sp.updated_at = datetime()
    WITH sp
    MERGE (g:GlobalSpeaker {id:sp.id})
    ON CREATE SET g.created_at = datetime(), g.method='manual', g.status='tentative'
    SET g.updated_at = datetime(),
        g.display_label = coalesce($display_label, g.display_label)
    MERGE (sp)-[:SAME_PERSON]->(g)
    RETURN sp.id AS sp_id, g.id AS gs_id, g.display_label AS label
    """
    with driver.session(database=NEO4J_DB) as sess:
        rec = sess.run(cy, sp_id=sp_id, display_label=display_label or None).single()
        return jsonify({"ok": True, "speaker_id": rec["sp_id"], "global_id": rec["gs_id"], "global_label": rec["label"]})

@app.post("/cluster/propagate_labels")
def cluster_propagate_labels():
    gs_id = (request.form.get("global_id") or "").strip()
    if not gs_id:
        return jsonify({"ok": False, "error": "global_id required"}), 400
    cy = """
    MATCH (g:GlobalSpeaker {id:$id})
    OPTIONAL MATCH (sp:Speaker)-[:SAME_PERSON]->(g)
    SET sp.label = coalesce(g.display_label, sp.label),
        sp.updated_at = datetime()
    WITH g, collect(sp.id) AS sp_ids, g.display_label AS lbl
    MATCH (s:Segment) WHERE s.speaker_id IN sp_ids
    SET s.speaker_label = coalesce(s.speaker_label, lbl)
    RETURN lbl AS label, sp_ids
    """
    with driver.session(database=NEO4J_DB) as sess:
        rec = sess.run(cy, id=gs_id).single()
        return jsonify({"ok": True, "label": rec["label"], "affected_speakers": rec["sp_ids"]})

# ---------- SUGGESTIONS (NEW) ----------
SUGGEST_HTML = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Speaker → Global Suggestions</title>
  <style>{BASE_CSS}</style>
</head>
<body>
  <div class="topnav">
    <a class="tag" href="{{ url_for('home') }}">← Back</a>
    <h1 style="margin:0;">Speaker → Global Suggestions</h1>
  </div>

  <form method="GET" action="{{ url_for('suggest_page') }}" style="margin-bottom:8px;">
    <div style="display:grid;grid-template-columns: 160px 160px 200px 1fr; gap:12px;">
      <div>
        <label>Top-k per speaker</label>
        <input type="number" name="topk" min="1" max="50" value="{{ topk }}">
      </div>
      <div>
        <label>Threshold (score ≥)</label>
        <input type="number" step="0.01" name="thresh" value="{{ thresh }}">
      </div>
      <div>
        <label>Max speakers (page)</label>
        <input type="number" name="limit" min="1" max="1000" value="{{ limit }}">
      </div>
      <div style="align-self:end;">
        <button type="submit">Refresh</button>
        {% if seed %}
          <input type="hidden" name="seed" value="{{ seed }}">
          <span class="tag">Seed: {{ seed }}</span>
        {% endif %}
      </div>
    </div>
  </form>

  {% if suggestions|length == 0 %}
    <div class="muted">No suggestions above threshold. Try lowering the threshold or increasing top-k.</div>
  {% else %}
    <form method="POST" action="{{ url_for('suggest_attach_bulk') }}">
      <input type="hidden" name="topk" value="{{ topk }}">
      <input type="hidden" name="thresh" value="{{ thresh }}">
      <input type="hidden" name="limit" value="{{ limit }}">
      <table>
        <thead>
          <tr>
            <th><input type="checkbox" onclick="for(const cb of document.querySelectorAll('.rowpick')) cb.checked=this.checked;"></th>
            <th>Speaker</th>
            <th>Candidates (Global)</th>
          </tr>
        </thead>
        <tbody>
          {% for row in suggestions %}
            <tr>
              <td><input class="rowpick" type="checkbox" name="pair" value="{{ row.sp_id }}|{{ row.cands[0].gs_id }}|{{ '%.6f'|format(row.cands[0].score) }}"></td>
              <td>
                <div class="tag">sp: {{ row.sp_label or row.sp_id }}</div>
                <div class="muted mono">{{ row.sp_id }}</div>
              </td>
              <td>
                {% for c in row.cands %}
                  <div class="card" style="margin:6px 0;padding:6px;">
                    <div>
                      <span class="tag good">{{ c.display_label or c.gs_id }}</span>
                      <span class="muted">score {{ '%.4f'|format(c.score) }}</span>
                      {% if c.status %}<span class="tag">{{ c.status }}</span>{% endif %}
                      {% if c.method %}<span class="tag">{{ c.method }}</span>{% endif %}
                    </div>
                    <div class="muted mono">{{ c.gs_id }}</div>
                    <div style="margin-top:6px;">
                      <form method="POST" action="{{ url_for('suggest_attach_one') }}" class="inline-form">
                        <input type="hidden" name="speaker_id" value="{{ row.sp_id }}">
                        <input type="hidden" name="global_id" value="{{ c.gs_id }}">
                        <input type="hidden" name="score" value="{{ '%.6f'|format(c.score) }}">
                        <button type="submit">Attach</button>
                      </form>
                      <form method="POST" action="{{ url_for('global_speaker_rename') }}" class="inline-form">
                        <input type="hidden" name="global_id" value="{{ c.gs_id }}">
                        <input type="text" name="display_label" placeholder="Rename cluster" style="width:180px;">
                        <button type="submit">Save label</button>
                      </form>
                    </div>
                  </div>
                {% endfor %}
              </td>
            </tr>
          {% endfor %}
        </tbody>
      </table>
      <div style="margin-top:8px;">
        <button type="submit">Attach Selected</button>
      </div>
    </form>
  {% endif %}
</body>
</html>
"""

def _vector_suggestions(seed: Optional[str], topk: int, thresh: float, limit: int):
    """
    Suggest (Speaker)->(GlobalSpeaker) links using vector ANN over GlobalSpeaker index.
    If `seed` is provided, only consider that specific Speaker id.
    Returns: List[{sp_id, sp_label, cands:[{gs_id, display_label, status, method, score}...]}]
    """
    cy = """
    // Gather candidate Speakers (unclustered or a specific seed)
    CALL {
      WITH $seed AS seed, $limit AS lim
      MATCH (sp:Speaker)
      WHERE sp.embedding IS NOT NULL
        AND (seed IS NOT NULL AND sp.id = seed
             OR seed IS NULL AND NOT (sp)-[:SAME_PERSON]->(:GlobalSpeaker))
      RETURN sp
      LIMIT CASE WHEN seed IS NULL THEN lim ELSE 1 END
    }
    WITH sp
    CALL db.index.vector.queryNodes($gs_idx, $topk, sp.embedding)
      YIELD node AS gs, score
    WITH sp, gs, score
    WHERE score >= $thresh
    RETURN sp.id AS sp_id,
           sp.label AS sp_label,
           collect({
             gs_id: gs.id,
             display_label: gs.display_label,
             status: gs.status,
             method: gs.method,
             score: score
           }) AS cands
    ORDER BY size(cands) DESC, sp_id
    """
    with driver.session(database=NEO4J_DB) as sess:
        rows = sess.run(cy, seed=seed, gs_idx=IDX_GLOBAL, topk=topk, thresh=thresh, limit=limit).data()
        # Sort each row's candidates by score desc
        for r in rows:
            r["cands"] = sorted(r["cands"], key=lambda x: x["score"], reverse=True)
        return rows

@app.get("/suggest")
def suggest_page():
    topk = int(request.args.get("topk") or SUGG_TOPK)
    thresh = float(request.args.get("thresh") or SUGG_THRESH)
    limit = int(request.args.get("limit") or SUGG_LIMIT_SPEAKERS)
    seed = (request.args.get("seed") or "").strip() or None
    suggestions = _vector_suggestions(seed, topk, thresh, limit)
    return render_template_string(SUGGEST_HTML, suggestions=suggestions, topk=topk, thresh=thresh, limit=limit, seed=seed)

@app.post("/suggest/attach")
def suggest_attach_one():
    sp_id = (request.form.get("speaker_id") or "").strip()
    gs_id = (request.form.get("global_id") or "").strip()
    score = (request.form.get("score") or "").strip()
    if not sp_id or not gs_id:
        return jsonify({"ok": False, "error": "speaker_id and global_id required"}), 400
    cy = """
    MATCH (sp:Speaker {id:$sp_id})
    MATCH (g:GlobalSpeaker {id:$gs_id})
    MERGE (sp)-[:SAME_PERSON]->(g)
    SET sp.updated_at = datetime(), g.updated_at = datetime()
    RETURN sp.id AS sp_id, g.id AS gs_id, g.display_label AS label
    """
    with driver.session(database=NEO4J_DB) as sess:
        rec = sess.run(cy, sp_id=sp_id, gs_id=gs_id).single()
        if not rec:
            return jsonify({"ok": False, "error": "Failed to attach"}), 400
        return redirect(url_for("suggest_page"))

@app.post("/suggest/attach_bulk")
def suggest_attach_bulk():
    """
    Accepts multiple "pair" form fields: each is "sp_id|gs_id|score".
    """
    pairs = request.form.getlist("pair")
    if not pairs:
        return jsonify({"ok": False, "error": "no selections"}), 400
    cy = """
    UNWIND $pairs AS pr
    WITH split(pr, "|") AS parts
    WITH parts[0] AS sp_id, parts[1] AS gs_id
    MATCH (sp:Speaker {id:sp_id})
    MATCH (g:GlobalSpeaker {id:gs_id})
    MERGE (sp)-[:SAME_PERSON]->(g)
    SET sp.updated_at = datetime(), g.updated_at = datetime()
    RETURN count(*) AS attached
    """
    with driver.session(database=NEO4J_DB) as sess:
        rec = sess.run(cy, pairs=pairs).single()
        attached = rec["attached"] if rec else 0
    return redirect(url_for("suggest_page"))

# ---------- FULL TRANSCRIPT PAGE ----------
TRANSCRIPT_HTML = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Transcript {{ t.key }}</title>
  <style>{BASE_CSS}</style>
</head>
<body>
  <div class="topnav">
    <a href="{{ url_for('home') }}" class="tag">← Back</a>
    <a class="tag" href="{{ url_for('suggest_page') }}">Suggestions</a>
  </div>
  <h1>Transcript <span class="mono">{{ t.key }}</span></h1>
  <div class="muted">id={{ t.id }} &middot; created={{ t.created_at or '—' }} &middot; started={{ t.started_at or '—' }}</div>

  {% if t.source_media %}
    <details style="margin-top:8px;" open>
      <summary>Media</summary>
      <div class="muted">{{ t.source_media }}</div>
      {% set play = best_audio_for_media(t.source_media) %}
      {% if play and (play.endswith('.mp3') or play.endswith('.wav') or play.endswith('.m4a') or play.endswith('.aac') or play.endswith('.flac')) %}
        <audio controls preload="metadata" src="{{ url_for('serve_media', path=play) }}"></audio>
      {% else %}
        <video controls preload="metadata" src="{{ url_for('serve_media', path=(play or t.source_media)) }}"></video>
      {% endif %}
    </details>
  {% endif %}

  {% if t.source_json %}
    <div class="muted">source_json: {{ t.source_json }}</div>
  {% endif %}
  {% if t.source_rttm %}
    <div class="muted">RTTM: {{ t.source_rttm }}</div>
  {% endif %}

  <h2>Utterances</h2>
  <table>
    <thead>
      <tr><th>#</th><th>Time</th><th>Speaker</th><th>Global Cluster</th><th>Text</th><th>Flags</th></tr>
    </thead>
    <tbody>
      {% for u in utterances %}
        <tr>
          <td class="mono">{{ u.idx or loop.index }}</td>
          <td class="mono">{{ '%.2f'|format(u.start or 0) }}–{{ '%.2f'|format(u.end or 0) }}</td>
          <td>
            {% if u.speaker %}
              <div class="tag"> {{ u.speaker.label or u.speaker.id }} </div>
            {% else %}
              <div class="tag warn">unknown</div>
            {% endif %}
          </td>
          <td>
            {% if u.globals and u.globals|length>0 %}
              {% for g in u.globals %}
                <div class="tag good" title="{{ g.gs_id }}">{{ g.display_label or g.gs_id }}</div>
              {% endfor %}
            {% else %}
              <div class="tag warn">unclustered</div>
            {% endif %}
          </td>
          <td>{{ u.text }}</td>
          <td>
            {% if u.is_lyrics %}<div class="tag lyrics">lyrics</div>{% endif %}
            {% if u.review_needed %}<div class="tag warn">needs review</div>{% endif %}
            {% if u.lyrics_score is not none %}<div class="tag">score {{ '%.2f'|format(u.lyrics_score) }}</div>{% endif %}
            {% if u.seg_idx is not none %}<div class="tag">seg#{{ u.seg_idx }}</div>{% endif %}
          </td>
        </tr>
      {% endfor %}
    </tbody>
  </table>
</body>
</html>
"""

@app.get("/t/<t_id>")
def transcription_full(t_id: str):
    cy = """
    MATCH (t:Transcription {id:$id})
    OPTIONAL MATCH (t)-[:HAS_UTTERANCE]->(u:Utterance)
    OPTIONAL MATCH (u)-[:OF_SEGMENT]->(s:Segment)
    OPTIONAL MATCH (u)-[:SPOKEN_BY]->(sp:Speaker)
    WITH t,u,s,sp,
         CASE WHEN sp IS NOT NULL THEN sp ELSE NULL END AS direct_sp,
         s.speaker_id AS seg_sid, s.speaker_label AS seg_slabel,
         s.is_lyrics AS is_lyrics, s.lyrics_score AS lyrics_score, s.review_needed AS review_needed
    WITH t,u,s,
         CASE WHEN direct_sp IS NOT NULL
              THEN {id: direct_sp.id, label: direct_sp.label}
              ELSE CASE WHEN seg_sid IS NOT NULL
                        THEN {id: seg_sid, label: seg_slabel}
                        ELSE NULL END
         END AS sp_map,
         is_lyrics, lyrics_score, review_needed
    OPTIONAL MATCH (sp2:Speaker {id:coalesce(sp_map.id, '__none__')})-[:SAME_PERSON]->(g:GlobalSpeaker)
    WITH t,u,s,sp_map,
         collect(DISTINCT {gs_id:g.id, display_label:g.display_label, method:g.method, status:g.status}) AS globals,
         is_lyrics, lyrics_score, review_needed
    ORDER BY u.start ASC
    RETURN t AS t,
           collect({
             id: u.id,
             idx: u.idx,
             start: u.start, end: u.end,
             text: u.text,
             speaker: sp_map,
             globals: [g IN globals WHERE g.gs_id IS NOT NULL],
             is_lyrics:is_lyrics, lyrics_score:lyrics_score, review_needed:review_needed,
             seg_idx: s.idx
           }) AS utterances
    """
    with driver.session(database=NEO4J_DB) as sess:
        rec = sess.run(cy, id=t_id).single()
        if not rec or not rec["t"]:
            abort(404)
        t = rec["t"]
        utts = rec["utterances"] or []
    return render_template_string(TRANSCRIPT_HTML, t=t, utterances=utts)

# ---------- MAIN ----------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")), debug=True)
