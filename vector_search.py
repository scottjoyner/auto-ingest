#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
vector_search.py
Standalone, high-signal CLI for querying your Neo4j media graph.

Subcommands
-----------
1) search-text:
   Semantic search over Utterance / Segment / Transcription with location snap.

2) similar-frames:
   ANN similarity for Frame embeddings, seeded by:
     --frame-id <id>    OR
     --file-key <key> --frame <int>

3) geo-frames:
   Geospatial pull of Frame nodes near a lat/lon (+ optional time/speed filters).

Outputs
-------
- Pretty table (default)
- JSON (--json)
- CSV  (--csv path)

Environment
-----------
NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DB
EMBED_BATCH, LOCAL_TZ, FRAME_DIM (default 256)
"""

import os, sys, csv, json, argparse, logging, math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError

try:
    from zoneinfo import ZoneInfo  # py>=3.9
except Exception:
    ZoneInfo = None

# =========================
# Config
# =========================
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOCAL_TZ = os.getenv("LOCAL_TZ", "America/New_York")

# Embeddings (for text queries)
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dims
EMBED_DIM = 384
DEFAULT_BATCH_SIZE = int(os.getenv("EMBED_BATCH", "32"))

# Frame embedding dim (set by your YOLO pipeline; override via env)
FRAME_DIM = int(os.getenv("FRAME_DIM", "256"))

# Neo4j
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "livelongandprosper")
NEO4J_DB = os.getenv("NEO4J_DB", "neo4j")

# =========================
# Embedding model
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass

_tokenizer = None
_model = None

def load_text_model():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        logging.info(f"Loading text embedding model on {DEVICE}…")
        _tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
        _model = AutoModel.from_pretrained(EMBED_MODEL_NAME).to(DEVICE).eval()
    return _tokenizer, _model

def _normalize(v: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(v, p=2, dim=1)

def _mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    summed = torch.sum(masked, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts

def embed_texts(texts: List[str], batch_size: int = DEFAULT_BATCH_SIZE, max_length: int = 512) -> List[List[float]]:
    if not texts:
        return []
    tok, mdl = load_text_model()
    vecs: List[List[float]] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            out = mdl(**enc)
            pooled = _mean_pooling(out.last_hidden_state, enc["attention_mask"])
            pooled = _normalize(pooled)
            vecs.extend(pooled.cpu().numpy().tolist())
    return vecs

# =========================
# Neo4j
# =========================
def neo4j_driver():
    if not (NEO4J_URI and NEO4J_USER and NEO4J_PASSWORD):
        raise RuntimeError("Neo4j not configured. Set NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD.")
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

SCHEMA_QUERIES = [
    "CREATE CONSTRAINT transcription_id IF NOT EXISTS FOR (t:Transcription) REQUIRE t.id IS UNIQUE",
    "CREATE CONSTRAINT segment_id IF NOT EXISTS FOR (s:Segment) REQUIRE s.id IS UNIQUE",
    "CREATE CONSTRAINT utterance_id IF NOT EXISTS FOR (u:Utterance) REQUIRE u.id IS UNIQUE",
    "CREATE CONSTRAINT frame_id IF NOT EXISTS FOR (f:Frame) REQUIRE f.id IS UNIQUE",
    "CREATE INDEX transcription_key IF NOT EXISTS FOR (t:Transcription) ON (t.key)",
    "CREATE INDEX frame_key IF NOT EXISTS FOR (f:Frame) ON (f.key)",
    "CREATE INDEX frame_key_frame IF NOT EXISTS FOR (f:Frame) ON (f.key, f.frame)"
]

def _create_vector_index(sess, label: str, prop: str, name: str, dim: int):
    q = f"""
    CREATE VECTOR INDEX {name} IF NOT EXISTS
    FOR (n:{label}) ON (n.{prop})
    OPTIONS {{
      indexConfig: {{
        `vector.dimensions`: {dim},
        `vector.similarity_function`: 'cosine'
      }}
    }}
    """
    sess.run(q)

def ensure_indexes(driver, frame_label: str, embed_prop: str):
    with driver.session(database=NEO4J_DB) as sess:
        for q in SCHEMA_QUERIES:
            sess.run(q)
        # Text indices
        _create_vector_index(sess, "Segment", "embedding", "segment_embedding_index", EMBED_DIM)
        _create_vector_index(sess, "Transcription", "embedding", "transcription_embedding_index", EMBED_DIM)
        _create_vector_index(sess, "Utterance", "embedding", "utterance_embedding_index", EMBED_DIM)
        # Frame index for selected property
        try:
            name = "frame_embedding_index" if embed_prop == "embedding" else f"frame_{embed_prop}_index"
            _create_vector_index(sess, frame_label, embed_prop, name, FRAME_DIM)
        except Exception as e:
            logging.warning(f"Frame vector index may not be available: {e}")

# =========================
# Pretty output helpers
# =========================
def _fmt(v: Any) -> str:
    if v is None: return ""
    if isinstance(v, float):
        if math.isnan(v): return ""
        return f"{v:.4f}" if abs(v) < 1000 else f"{v:.2f}"
    return str(v)

def print_table(rows: List[Dict[str, Any]], cols: List[Tuple[str,str]]):
    if not rows:
        print("(no results)")
        return
    widths = []
    headers = [h for h,_ in cols]
    for i,(_,k) in enumerate(cols):
        w = max(len(headers[i]), max(len(_fmt(r.get(k,""))) for r in rows))
        widths.append(w)
    line = " | ".join(h.ljust(widths[i]) for i,h in enumerate(headers))
    sep  = "-+-".join("-"*widths[i] for i in range(len(widths)))
    print(line)
    print(sep)
    for r in rows:
        print(" | ".join(_fmt(r.get(k,"")).ljust(widths[i]) for i,(_,k) in enumerate(cols)))

def maybe_dump(rows: List[Dict[str,Any]], as_json: bool, csv_path: Optional[str], csv_cols: Optional[List[str]] = None):
    if as_json:
        print(json.dumps(rows, ensure_ascii=False, indent=2, default=str))
    if csv_path:
        keys = csv_cols or sorted({k for r in rows for k in r.keys()})
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in keys})

# =========================
# Text semantic search (Utterance/Segment/Transcription)
# =========================
def search_text(driver, query: str, target: str, top_k: int, win_minutes: int, include_embedding: bool) -> List[Dict[str,Any]]:
    qvec = embed_texts([query])[0]
    if target == "utterance":
        index_name = "utterance_embedding_index"
        # (same logic you validated; left compact)
        cypher = f"""
        CALL db.index.vector.queryNodes('{index_name}', $k, $qvec)
          YIELD node, score
        WHERE 'Utterance' IN labels(node)
        MATCH (t:Transcription)-[:HAS_UTTERANCE]->(u:Utterance)
        WHERE u = node
        WITH node, score, t, u,
             coalesce(u.absolute_start, t.started_at) AS ts_start,
             coalesce(u.absolute_end,   t.ended_at)   AS ts_end,
             toInteger($win) * 60000 AS win_ms
        WITH node, score, t, u, win_ms,
             CASE WHEN ts_start IS NULL THEN NULL ELSE ts_start.epochMillis END AS ts_start_ms,
             CASE WHEN ts_end   IS NULL THEN NULL ELSE ts_end.epochMillis   END AS ts_end_ms,
             CASE WHEN t.started_at IS NULL THEN NULL ELSE t.started_at.epochMillis END AS t0_ms,
             CASE WHEN t.ended_at   IS NULL THEN NULL ELSE t.ended_at.epochMillis   END AS t1_ms
        WITH node, score, t, u, win_ms, ts_start_ms, ts_end_ms, t0_ms, t1_ms,
             CASE
               WHEN ts_start_ms IS NOT NULL AND ts_end_ms IS NOT NULL
                 THEN toInteger((ts_start_ms + ts_end_ms) / 2)
               WHEN t0_ms IS NOT NULL
                 THEN t0_ms + toInteger(((coalesce(toFloat(u.start), toFloat(u.end), 0.0) + coalesce(toFloat(u.end), toFloat(u.start), 0.0))/2.0)*1000)
               WHEN t0_ms IS NOT NULL AND t1_ms IS NOT NULL
                 THEN toInteger((t0_ms + t1_ms) / 2)
               ELSE NULL
             END AS ts_mid_ms
        WITH node, score, t, u, win_ms, ts_start_ms, ts_end_ms, ts_mid_ms,
             coalesce(ts_start_ms, ts_mid_ms) AS left_base,
             coalesce(ts_end_ms,   ts_mid_ms) AS right_base
        OPTIONAL MATCH (pl:PhoneLog)
          WHERE ts_mid_ms IS NOT NULL
            AND pl.epoch_millis >= left_base  - win_ms
            AND pl.epoch_millis <= right_base + win_ms
        WITH node, score, t, u, ts_mid_ms, pl
        ORDER BY CASE WHEN pl IS NULL OR ts_mid_ms IS NULL THEN 9223372036854775807 ELSE abs(pl.epoch_millis - ts_mid_ms) END ASC
        WITH node, score, t, u, ts_mid_ms, collect(pl)[0] AS nearest_pl
        OPTIONAL MATCH (t)-[:HAS_SEGMENT]->(allSeg)
        WITH node, score, t, u, ts_mid_ms, nearest_pl, max(allSeg.end) AS t_dur_s
        WITH node, score, t, u, ts_mid_ms, nearest_pl, t_dur_s,
             CASE WHEN t_dur_s IS NOT NULL AND t_dur_s > 0
               THEN ((coalesce(toFloat(u.start),0.0) + coalesce(toFloat(u.end),0.0))/2.0) / toFloat(t_dur_s)
               ELSE 0.5 END AS pos_raw
        WITH node, score, t, u, ts_mid_ms, nearest_pl,
             CASE WHEN pos_raw < 0 THEN 0.0 WHEN pos_raw > 1 THEN 1.0 ELSE pos_raw END AS pos_clamped
        OPTIONAL MATCH (fmax:Frame {{key: t.key}})
        WITH node, score, t, u, ts_mid_ms, nearest_pl, pos_clamped, max(fmax.frame) AS max_frame
        WITH node, score, t, u, ts_mid_ms, nearest_pl,
             CASE WHEN max_frame IS NULL THEN NULL ELSE toInteger(round(pos_clamped * toFloat(max_frame))) END AS target_frame
        OPTIONAL MATCH (f:Frame {{key: t.key}})
        WHERE target_frame IS NOT NULL
        WITH node, score, t, u, ts_mid_ms, nearest_pl, target_frame, f
        ORDER BY abs(f.frame - target_frame) ASC
        WITH node, score, t, u, ts_mid_ms, nearest_pl, collect(f)[0] AS nearest_frame
        RETURN
            node.id AS id,
            score,
            node.text AS text,
            node.start AS start,
            node.end AS end,
            t.id AS transcription_id,
            t.key AS file_key,
            t.started_at AS started_at,
            CASE WHEN $include_embedding THEN node.embedding ELSE NULL END AS embedding,
            coalesce(nearest_pl.latitude,  nearest_frame.lat)  AS latitude,
            coalesce(nearest_pl.longitude, nearest_frame.long) AS longitude,
            CASE WHEN nearest_pl.timestamp IS NOT NULL THEN nearest_pl.timestamp
                 WHEN nearest_pl.epoch_millis IS NOT NULL THEN datetime({{epochMillis: toInteger(nearest_pl.epoch_millis)}})
                 WHEN ts_mid_ms IS NOT NULL THEN datetime({{epochMillis: toInteger(ts_mid_ms)}})
                 ELSE NULL END AS location_ts,
            CASE WHEN nearest_pl IS NOT NULL THEN 'PhoneLog'
                 WHEN nearest_frame IS NOT NULL THEN 'Frame'
                 WHEN ts_mid_ms IS NOT NULL THEN 'TranscriptionMid'
                 ELSE NULL END AS location_source
        ORDER BY score DESC
        """
    elif target == "segment":
        index_name = "segment_embedding_index"
        cypher = f"""
        CALL db.index.vector.queryNodes('{index_name}', $k, $qvec)
          YIELD node, score
        WHERE 'Segment' IN labels(node)
        MATCH (t:Transcription)-[:HAS_SEGMENT]->(s:Segment)
        WHERE s = node
        WITH node, score, t, s
        OPTIONAL MATCH (t)-[:HAS_SEGMENT]->(allSeg)
        WITH node, score, t, s, max(allSeg.end) AS t_dur_s
        WITH node, score, t, s,
             CASE WHEN t_dur_s IS NOT NULL AND t_dur_s > 0
               THEN ((coalesce(toFloat(s.start),0.0) + coalesce(toFloat(s.end),0.0))/2.0) / toFloat(t_dur_s)
               ELSE 0.5 END AS pos_raw
        WITH node, score, t, s,
             CASE WHEN pos_raw < 0 THEN 0.0 WHEN pos_raw > 1 THEN 1.0 ELSE pos_raw END AS pos_clamped
        OPTIONAL MATCH (fmax:Frame {{key: t.key}})
        WITH node, score, t, s, pos_clamped, max(fmax.frame) AS max_frame
        WITH node, score, t, s,
             CASE WHEN max_frame IS NULL THEN NULL ELSE toInteger(round(pos_clamped * toFloat(max_frame))) END AS target_frame
        OPTIONAL MATCH (f:Frame {{key: t.key}})
        WHERE target_frame IS NOT NULL
        WITH node, score, t, s, target_frame, f
        ORDER BY abs(f.frame - target_frame) ASC
        WITH node, score, t, s, collect(f)[0] AS nearest_frame
        RETURN
            node.id AS id,
            score,
            node.text AS text,
            node.start AS start,
            node.end AS end,
            t.id AS transcription_id,
            t.key AS file_key,
            t.started_at AS started_at,
            CASE WHEN $include_embedding THEN node.embedding ELSE NULL END AS embedding,
            nearest_frame.lat  AS latitude,
            nearest_frame.long AS longitude,
            CASE WHEN nearest_frame.millis IS NOT NULL THEN datetime({{epochMillis: toInteger(nearest_frame.millis)}}) ELSE NULL END AS location_ts,
            'Frame' AS location_source
        ORDER BY score DESC
        """
    else:
        index_name = "transcription_embedding_index"
        cypher = f"""
        CALL db.index.vector.queryNodes('{index_name}', $k, $qvec)
          YIELD node, score
        WHERE 'Transcription' IN labels(node)
        RETURN
            node.id AS id,
            score,
            node.text AS text,
            null AS start,
            null AS end,
            node.id AS transcription_id,
            node.key AS file_key,
            node.started_at AS started_at,
            CASE WHEN $include_embedding THEN node.embedding ELSE NULL END AS embedding,
            NULL AS latitude,
            NULL AS longitude,
            node.started_at AS location_ts,
            'TranscriptionMid' AS location_source
        ORDER BY score DESC
        """
    with driver.session(database=NEO4J_DB) as sess:
        res = sess.run(cypher, k=top_k, qvec=qvec, include_embedding=include_embedding, win=win_minutes)
        return [r.data() for r in res]

# =========================
# Frame Similarity (ANN over Frame.embedding)
# =========================
def get_frame_embedding(driver, frame_id: Optional[str], file_key: Optional[str], frame_no: Optional[int],
                        frame_label: str, embed_prop: str) -> Tuple[List[float], Dict[str,Any]]:
    def first_hit_with_vector(sess, where, params):
        q = f"""
        MATCH (f:`{frame_label}` {where})
        RETURN f.id AS id, f.key AS key, f.frame AS frame, f.{embed_prop} AS emb, keys(f) AS props
        LIMIT 1
        """
        rec = sess.run(q, **params).single()
        if rec and rec.get("emb") is not None:
            return rec
        return None

    def first_hit_any(sess, where, params):
        q = f"""
        MATCH (f:`{frame_label}` {where})
        RETURN f.id AS id, f.key AS key, f.frame AS frame, keys(f) AS props, f
        LIMIT 1
        """
        return sess.run(q, **params).single()

    with driver.session(database=NEO4J_DB) as sess:
        # 1) Try direct lookups with the chosen embedding property
        if frame_id:
            rec = first_hit_with_vector(sess, "{id:$id}", {"id": frame_id})
            # Also try alternate id property some pipelines use
            if rec is None:
                rec = first_hit_with_vector(sess, "{frame_id:$id}", {"id": frame_id})
        else:
            rec = first_hit_with_vector(sess, "{key:$key, frame:$frame}", {"key": file_key, "frame": int(frame_no)})

        if rec:
            return rec["emb"], {"id": rec["id"], "key": rec["key"], "frame": rec["frame"]}

        # 2) Not found or missing vector: fetch *any* matching node and report what it has
        if frame_id:
            anyrec = first_hit_any(sess, "{id:$id}", {"id": frame_id}) \
                     or first_hit_any(sess, "{frame_id:$id}", {"id": frame_id})
        else:
            anyrec = first_hit_any(sess, "{key:$key, frame:$frame}", {"key": file_key, "frame": int(frame_no)})

        if anyrec:
            props = anyrec.get("props") or []
            # common embedding prop fallbacks
            for alt in [embed_prop, "vec", "vector", "features", "feat", "embedding_l2", "embedding_cos"]:
                if alt == embed_prop:
                    continue
                q = f"""
                MATCH (f:`{frame_label}` {{id:$id}})
                RETURN f.`{alt}` AS emb
                """
                if frame_id:
                    r2 = sess.run(q, id=anyrec["id"]).single()
                else:
                    q2 = f"""
                    MATCH (f:`{frame_label}` {{key:$key, frame:$frame}})
                    RETURN f.`{alt}` AS emb
                    """
                    r2 = sess.run(q2, key=anyrec["key"], frame=anyrec["frame"]).single()
                if r2 and r2.get("emb") is not None:
                    # Found an alternative property; return it
                    return r2["emb"], {"id": anyrec["id"], "key": anyrec["key"], "frame": anyrec["frame"]}

            raise RuntimeError(
                f"Seed frame found but has no vector at '{embed_prop}'. "
                f"Available properties: {props}. "
                f"Try: --frame-embed-prop one of {{vec, vector, features, feat, embedding_l2, embedding_cos}}."
            )

        # 3) No node found at all → provide counts for guidance
        stats = sess.run(f"""
            MATCH (f:`{frame_label}`)
            RETURN count(f) as total,
                   count(CASE WHEN exists(f.{embed_prop}) THEN 1 END) as with_vec
        """).single()
        total = stats["total"] if stats else 0
        with_vec = stats["with_vec"] if stats else 0
        raise RuntimeError(
            f"Seed frame not found. Checked by "
            f"{'id' if frame_id else 'key+frame'}. "
            f"({frame_label} total={total}, with {embed_prop}={with_vec}). "
            f"If your label/prop differ, use --frame-label and/or --frame-embed-prop."
        )

def similar_frames(driver, seed_emb: List[float], top_k: int, include_seed: bool, frame_label: str, embed_prop: str) -> List[Dict[str,Any]]:
    index_name = "frame_embedding_index" if embed_prop == "embedding" else f"frame_{embed_prop}_index"
    cypher = f"""
    CALL db.index.vector.queryNodes('{index_name}', $k, $qvec)
      YIELD node, score
    WHERE '{frame_label}' IN labels(node)
    RETURN
      node.id    AS id,
      score      AS score,
      node.key   AS file_key,
      node.frame AS frame,
      node.lat   AS latitude,
      node.long  AS longitude,
      CASE WHEN node.millis IS NOT NULL THEN datetime({{epochMillis: toInteger(node.millis)}}) ELSE NULL END AS ts,
      node.mph   AS mph
    ORDER BY score DESC
    """
    with driver.session(database=NEO4J_DB) as sess:
        res = sess.run(cypher, k=top_k + (1 if not include_seed else 0), qvec=seed_emb)
        rows = [r.data() for r in res]
        return rows[:top_k]

# =========================
# Geo Frames (lat/lon/radius + optional time/speed filters)
# =========================
def geo_frames(driver, lat: float, lon: float, radius_m: float,
               start: Optional[str], end: Optional[str],
               min_mph: Optional[float], max_mph: Optional[float],
               limit: int) -> List[Dict[str,Any]]:
    # Simple haversine in-cypher approximation using degrees → meters via 111_320m/deg (~latitude)
    cypher = """
    WITH $lat AS lat0, $lon AS lon0, $radius AS R
    MATCH (f:Frame)
    WHERE f.lat IS NOT NULL AND f.long IS NOT NULL
    // bounding box prefilter (cheap)
      AND f.lat  >= lat0 - (R/111320.0) AND f.lat  <= lat0 + (R/111320.0)
      AND f.long >= lon0 - (R/(111320.0 * cos(radians(lat0)))) AND f.long <= lon0 + (R/(111320.0 * cos(radians(lat0))))
    WITH f, lat0, lon0, R,
         6371000.0 * 2 * asin(sqrt( pow(sin(radians((f.lat-lat0)/2)),2)
                                   + cos(radians(lat0))*cos(radians(f.lat))*pow(sin(radians((f.long-lon0)/2)),2) )) AS dist
    WHERE dist <= R
    WITH f, dist
    WHERE ($start IS NULL OR (f.millis IS NOT NULL AND f.millis >= $start_ms))
      AND ($end   IS NULL OR (f.millis IS NOT NULL AND f.millis <= $end_ms))
      AND ($min_mph IS NULL OR (f.mph IS NOT NULL AND f.mph >= $min_mph))
      AND ($max_mph IS NULL OR (f.mph IS NOT NULL AND f.mph <= $max_mph))
    RETURN f.id AS id, f.key AS file_key, f.frame AS frame,
           f.lat AS latitude, f.long AS longitude,
           CASE WHEN f.millis IS NOT NULL THEN datetime({epochMillis: toInteger(f.millis)}) ELSE NULL END AS ts,
           f.mph AS mph, dist AS meters
    ORDER BY dist ASC
    LIMIT $limit
    """
    def to_ms(iso: Optional[str]) -> Optional[int]:
        if not iso: return None
        # accept either epoch millis or ISO string
        if iso.isdigit():
            return int(iso)
        dt = datetime.fromisoformat(iso.replace("Z","+00:00"))
        return int(dt.timestamp()*1000)

    params = {
        "lat": float(lat),
        "lon": float(lon),
        "radius": float(radius_m),
        "start": start,
        "end": end,
        "start_ms": to_ms(start),
        "end_ms": to_ms(end),
        "min_mph": None if min_mph is None else float(min_mph),
        "max_mph": None if max_mph is None else float(max_mph),
        "limit": int(limit),
    }
    with driver.session(database=NEO4J_DB) as sess:
        res = sess.run(cypher, **params)
        return [r.data() for r in res]

# =========================
# CLI
# =========================
def main():
    p = argparse.ArgumentParser(description="Location-aware vector & frame search (standalone).")
    p.add_argument("--json", action="store_true", help="Output JSON instead of table.")
    p.add_argument("--csv", type=str, default=None, help="Also write results to this CSV path.")
    p.add_argument("--no-index-check", action="store_true", help="Skip vector-index creation (faster startup).")

    sub = p.add_subparsers(dest="cmd", required=True)

    # 1) search-text
    sp1 = sub.add_parser("search-text", help="Semantic search over utterance/segment/transcription.")
    sp1.add_argument("--q", required=True, help="Text query.")
    sp1.add_argument("--target", choices=["utterance","segment","transcription"], default="utterance")
    sp1.add_argument("--topk", type=int, default=10)
    sp1.add_argument("--win-mins", type=int, default=10, help="± minutes for PhoneLog snap.")
    sp1.add_argument("--include-emb", action="store_true", help="Return node embeddings.")
    sp1.add_argument("--text-chars", type=int, default=160, help="Max chars of text snippet in table output.")


    # 2) similar-frames
    sp2 = sub.add_parser("similar-frames", help="Find visually similar frames via ANN over Frame.embedding.")
    g = sp2.add_mutually_exclusive_group(required=True)
    g.add_argument("--frame-id", type=str, help="Seed frame id.")
    g.add_argument("--file-key", type=str, help="Seed file key (use with --frame).")
    sp2.add_argument("--frame", type=int, help="Seed frame number (use with --file-key).")
    sp2.add_argument("--topk", type=int, default=20)
    sp2.add_argument("--include-seed", action="store_true", help="Include the seed frame in results (default off).")

    # 3) geo-frames
    sp3 = sub.add_parser("geo-frames", help="Find frames near a lat/lon within R meters (optional time/speed filters).")
    sp3.add_argument("--lat", type=float, required=True)
    sp3.add_argument("--lon", type=float, required=True)
    sp3.add_argument("--radius-m", type=float, default=200.0)
    sp3.add_argument("--start", type=str, default=None, help="ISO8601 or epochMillis (optional).")
    sp3.add_argument("--end", type=str, default=None, help="ISO8601 or epochMillis (optional).")
    sp3.add_argument("--min-mph", type=float, default=None)
    sp3.add_argument("--max-mph", type=float, default=None)
    sp3.add_argument("--limit", type=int, default=100)

    args = p.parse_args()

    driver = neo4j_driver()
    if not args.no_index_check:
        ensure_indexes(driver, args.frame_label, args.frame_embed_prop)

    if args.cmd == "search-text":
        rows = search_text(driver, args.q, args.target, args.topk, args.win_mins, args.include_emb)

        # Create a trimmed snippet for table display (full text still available via --json/--csv)
        snip_len = getattr(args, "text_chars", 160)
        for r in rows:
            t = (r.get("text") or "").replace("\n", " ").strip()
            r["text_snip"] = (t[:snip_len - 1] + "…") if len(t) > snip_len else t

        if not args.json and not args.csv:
            cols = [
                ("score","score"),
                ("file_key","file_key"),
                ("id","id"),
                ("start","start"),
                ("end","end"),
                ("text","text_snip"),          # 👈 show the snippet
                ("latitude","latitude"),
                ("longitude","longitude"),
                ("location_ts","location_ts"),
                ("source","location_source"),
            ]
            print_table(rows, cols)

        maybe_dump(rows, args.json, args.csv)


    elif args.cmd == "similar-frames":
        if not args.frame_id and not (args.file_key and args.frame is not None):
            raise SystemExit("--frame-id OR (--file-key and --frame) required.")
        seed_emb, meta = get_frame_embedding(driver, args.frame_id, args.file_key, args.frame)
        rows = similar_frames(driver, seed_emb, args.topk, args.include_seed, args.frame_label, args.frame_embed_prop)
        # Optionally drop the seed from results by ID match:
        if not args.include_seed and args.frame_id:
            rows = [r for r in rows if r.get("id") != args.frame_id][:args.topk]
        if not args.json and not args.csv:
            cols = [
                ("score","score"),
                ("file_key","file_key"),
                ("frame","frame"),
                ("latitude","latitude"),
                ("longitude","longitude"),
                ("ts","ts"),
                ("mph","mph"),
                ("id","id"),
            ]
            print_table(rows, cols)
        maybe_dump(rows, args.json, args.csv)

    elif args.cmd == "geo-frames":
        rows = geo_frames(driver, args.lat, args.lon, args.radius_m, args.start, args.end, args.min_mph, args.max_mph, args.limit)
        if not args.json and not args.csv:
            cols = [
                ("meters","meters"),
                ("file_key","file_key"),
                ("frame","frame"),
                ("latitude","latitude"),
                ("longitude","longitude"),
                ("ts","ts"),
                ("mph","mph"),
                ("id","id"),
            ]
            print_table(rows, cols)
        maybe_dump(rows, args.json, args.csv)

if __name__ == "__main__":
    main()
