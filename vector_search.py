#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
vector_search.py
Standalone, high-signal CLI for querying the Neo4j media graph.

Subcommands
-----------
1) search-text:
   Semantic search over any text-bearing label (Chunk / Summary / Segment /
   Utterance / Transcription / Entity / Concept / Topic / Keyword / KgNode /
   Note / Speaker / GlobalSpeaker). Uses the SAME embedding model + pooling as
   the ingest / re-embed path (auto_ingest.embed.EmbedModel), so query vectors
   are directly comparable to the stored `emb_e5_large` vectors.

2) similar-frames:
   ANN similarity for Frame embeddings, seeded by --frame-id OR --file-key --frame.

3) geo-frames:
   Geospatial pull of Frame nodes near a lat/lon (+ optional time/speed filters).

Environment
-----------
NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DB   (which KG to hit)
EMBED_MODEL_NAME  default intfloat/multilingual-e5-large (must match the model
                  used to WRITE the vectors you are searching)
EMBED_PROP        default emb_e5_large (the vector property to query)
FRAME_DIM         default 256
LOCAL_TZ          default America/New_York
"""

import os, sys, csv, json, argparse, logging, math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

try:
    from auto_ingest_config import get_neo4j_env
except Exception:  # packaged import fallback
    from auto_ingest._config import get_neo4j_env

import numpy as np
import torch
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError

from auto_ingest.embed import load_embed_model
from transformers import AutoConfig

# =========================
# Config
# =========================
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOCAL_TZ = os.getenv("LOCAL_TZ", "America/New_York")

# MUST match the model used to WRITE the vectors being searched.
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "intfloat/multilingual-e5-large")
EMBED_PROP = os.getenv("EMBED_PROP", "emb_e5_large")
FRAME_DIM = int(os.getenv("FRAME_DIM", "256"))

NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DB = get_neo4j_env()

# label -> (neo4j label, text property) — mirrors reembed.py SCHEMA so the
# source text returned matches what was embedded.
TEXT_TARGETS: Dict[str, Tuple[str, str]] = {
    "chunk":         ("Chunk", "text"),
    "summary":       ("Summary", "text"),
    "segment":       ("Segment", "text"),
    "utterance":     ("Utterance", "text"),
    "transcription": ("Transcription", "text"),
    "entity":        ("Entity", "text"),
    "concept":       ("Concept", "name"),
    "topic":         ("Topic", "name"),
    "keyword":       ("Keyword", "name"),
    "kgnode":        ("KgNode", "title"),
    "note":          ("Note", "text"),
    "speaker":       ("Speaker", "label"),
    "globalspeaker": ("GlobalSpeaker", "display_label"),
}


# =========================
# Query embedding model
# =========================
_MODEL = None

def query_embedder():
    global _MODEL
    if _MODEL is None:
        logging.info("Loading query embedding model %s", EMBED_MODEL_NAME)
        _MODEL = load_embed_model(EMBED_MODEL_NAME)
    return _MODEL

def embed_query(text: str) -> List[float]:
    return query_embedder().embed([text])[0]

def _model_dim(name: str) -> int:
    return int(getattr(AutoConfig.from_pretrained(name), "hidden_size", 0) or 384)


# =========================
# Neo4j
# =========================
def neo4j_driver():
    if not (NEO4J_URI and NEO4J_USER and NEO4J_PASSWORD):
        raise RuntimeError("Neo4j not configured. Set NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD.")
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def ensure_indexes(driver, frame_label: str, frame_embed_prop: str):
    """Create the text + frame vector indexes this CLI queries (idempotent)."""
    dim = _model_dim(EMBED_MODEL_NAME)
    with driver.session(database=NEO4J_DB) as sess:
        for label, _ in TEXT_TARGETS.values():
            name = f"{label}_{EMBED_PROP}_index"
            sess.run(
                f"CREATE VECTOR INDEX {name} IF NOT EXISTS "
                f"FOR (n:{label}) ON (n.{EMBED_PROP}) "
                f"OPTIONS {{ indexConfig: {{ `vector.dimensions`: {dim}, "
                f"`vector.similarity_function`: 'cosine' }} }}"
            )
        try:
            fname = "frame_embedding_index" if frame_embed_prop == "embedding" else f"frame_{frame_embed_prop}_index"
            sess.run(
                f"CREATE VECTOR INDEX {fname} IF NOT EXISTS "
                f"FOR (n:{frame_label}) ON (n.{frame_embed_prop}) "
                f"OPTIONS {{ indexConfig: {{ `vector.dimensions`: {FRAME_DIM}, "
                f"`vector.similarity_function`: 'cosine' }} }}"
            )
        except Exception as e:
            logging.warning("Frame vector index may not be available: %s", e)


# =========================
# Pretty output helpers
# =========================
def _fmt(v: Any) -> str:
    if v is None: return ""
    if isinstance(v, float):
        if math.isnan(v): return ""
        return f"{v:.4f}" if abs(v) < 1000 else f"{v:.2f}"
    return str(v)

def print_table(rows: List[Dict[str, Any]], cols: List[Tuple[str, str]]):
    if not rows:
        print("(no results)")
        return
    widths = []
    headers = [h for h, _ in cols]
    for i, (_, k) in enumerate(cols):
        w = max(len(headers[i]), max(len(_fmt(r.get(k, ""))) for r in rows))
        widths.append(w)
    line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    sep = "-+-".join("-" * widths[i] for i in range(len(widths)))
    print(line); print(sep)
    for r in rows:
        print(" | ".join(_fmt(r.get(k, "")).ljust(widths[i]) for i, (_, k) in enumerate(cols)))

def maybe_dump(rows: List[Dict[str, Any]], as_json: bool, csv_path: Optional[str], csv_cols: Optional[List[str]] = None):
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
# Text semantic search
# =========================
def search_text(driver, query: str, target: str, top_k: int,
                include_embedding: bool, text_chars: int) -> List[Dict[str, Any]]:
    label, text_prop = TEXT_TARGETS[target]
    index_name = f"{label}_{EMBED_PROP}_index"
    qvec = embed_query(query)
    cypher = f"""
    CALL db.index.vector.queryNodes('{index_name}', $k, $qvec)
      YIELD node, score
    WHERE '{label}' IN labels(node)
    RETURN
        node.id AS id,
        score,
        node.{text_prop} AS text,
        '{label}' AS label,
        CASE WHEN $include_embedding THEN node.{EMBED_PROP} ELSE NULL END AS embedding
    ORDER BY score DESC
    """
    with driver.session(database=NEO4J_DB) as sess:
        res = sess.run(cypher, k=top_k, qvec=qvec, include_embedding=include_embedding)
        rows = [r.data() for r in res]
    for r in rows:
        t = (r.get("text") or "").replace("\n", " ").strip()
        r["text_snip"] = (t[:text_chars - 1] + "…") if len(t) > text_chars else t
    return rows


# =========================
# Frame Similarity (ANN over Frame.<embed_prop>)
# =========================
def get_frame_embedding(driver, frame_id: Optional[str], file_key: Optional[str], frame_no: Optional[int],
                        frame_label: str, embed_prop: str):
    def first_hit_with_vector(sess, where, params):
        q = f"MATCH (f:`{frame_label}` {where}) RETURN f.id AS id, f.key AS key, f.frame AS frame, f.{embed_prop} AS emb, keys(f) AS props LIMIT 1"
        rec = sess.run(q, **params).single()
        if rec and rec.get("emb") is not None:
            return rec
        return None

    def first_hit_any(sess, where, params):
        q = f"MATCH (f:`{frame_label}` {where}) RETURN f.id AS id, f.key AS key, f.frame AS frame, keys(f) AS props, f LIMIT 1"
        return sess.run(q, **params).single()

    with driver.session(database=NEO4J_DB) as sess:
        if frame_id:
            rec = first_hit_with_vector(sess, "{id:$id}", {"id": frame_id}) \
                or first_hit_with_vector(sess, "{frame_id:$id}", {"id": frame_id})
        else:
            rec = first_hit_with_vector(sess, "{key:$key, frame:$frame}", {"key": file_key, "frame": int(frame_no)})
        if rec:
            return rec["emb"], {"id": rec["id"], "key": rec["key"], "frame": rec["frame"]}
        anyrec = (first_hit_any(sess, "{id:$id}", {"id": frame_id}) \
                  or first_hit_any(sess, "{frame_id:$id}", {"id": frame_id})) if frame_id \
            else first_hit_any(sess, "{key:$key, frame:$frame}", {"key": file_key, "frame": int(frame_no)})
        if anyrec:
            props = anyrec.get("props") or []
            for alt in [embed_prop, "vec", "vector", "features", "feat", "embedding_l2", "embedding_cos"]:
                if alt == embed_prop:
                    continue
                q = f"MATCH (f:`{frame_label}` {{id:$id}}) RETURN f.`{alt}` AS emb"
                r2 = sess.run(q, id=anyrec["id"]).single()
                if r2 and r2.get("emb") is not None:
                    return r2["emb"], {"id": anyrec["id"], "key": anyrec["key"], "frame": anyrec["frame"]}
            raise RuntimeError(f"Seed frame found but has no vector at '{embed_prop}'. Available: {props}")
        stats = sess.run(f"MATCH (f:`{frame_label}`) RETURN count(f) as total, count(CASE WHEN exists(f.{embed_prop}) THEN 1 END) as with_vec").single()
        total = stats["total"] if stats else 0
        with_vec = stats["with_vec"] if stats else 0
        raise RuntimeError(f"Seed frame not found. ({frame_label} total={total}, with {embed_prop}={with_vec}).")


def similar_frames(driver, seed_emb: List[float], top_k: int, include_seed: bool, frame_label: str, embed_prop: str):
    index_name = "frame_embedding_index" if embed_prop == "embedding" else f"frame_{embed_prop}_index"
    cypher = f"""
    CALL db.index.vector.queryNodes('{index_name}', $k, $qvec)
      YIELD node, score
    WHERE '{frame_label}' IN labels(node)
    RETURN
      node.id AS id, score AS score, node.key AS file_key, node.frame AS frame,
      node.lat AS latitude, node.long AS longitude,
      CASE WHEN node.millis IS NOT NULL THEN datetime({{epochMillis: toInteger(node.millis)}}) ELSE NULL END AS ts,
      node.mph AS mph
    ORDER BY score DESC
    """
    with driver.session(database=NEO4J_DB) as sess:
        res = sess.run(cypher, k=top_k + (1 if not include_seed else 0), qvec=seed_emb)
        rows = [r.data() for r in res]
    return rows[:top_k]


# =========================
# Geo Frames
# =========================
def geo_frames(driver, lat: float, lon: float, radius_m: float,
               start: Optional[str], end: Optional[str],
               min_mph: Optional[float], max_mph: Optional[float], limit: int):
    cypher = """
    WITH $lat AS lat0, $lon AS lon0, $radius AS R
    MATCH (f:Frame)
    WHERE f.lat IS NOT NULL AND f.long IS NOT NULL
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
    def to_ms(iso):
        if not iso: return None
        if iso.isdigit(): return int(iso)
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return int(dt.timestamp() * 1000)
    params = {"lat": float(lat), "lon": float(lon), "radius": float(radius_m),
              "start": start, "end": end, "start_ms": to_ms(start), "end_ms": to_ms(end),
              "min_mph": None if min_mph is None else float(min_mph),
              "max_mph": None if max_mph is None else float(max_mph), "limit": int(limit)}
    with driver.session(database=NEO4J_DB) as sess:
        return [r.data() for r in sess.run(cypher, **params)]


# =========================
# CLI
# =========================
def main():
    global EMBED_MODEL_NAME, EMBED_PROP
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--json", action="store_true", help="Output JSON instead of table.")
    parent.add_argument("--csv", type=str, default=None, help="Also write results to this CSV path.")
    parent.add_argument("--no-index-check", action="store_true", help="Skip vector-index creation (faster startup).")
    parent.add_argument("--model", default=EMBED_MODEL_NAME, help="Query embedding model (must match the vectors' model).")
    parent.add_argument("--prop", default=EMBED_PROP, help="Vector property to search (e.g. emb_e5_large, emb_mini12).")
    parent.add_argument("--frame-label", default="Frame", help="Node label for frame embeddings.")
    parent.add_argument("--frame-embed-prop", default="embedding", help="Property holding frame vectors.")

    p = argparse.ArgumentParser(parents=[parent], description="Semantic + frame/geo search over the Neo4j media graph.")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp1 = sub.add_parser("search-text", parents=[parent], help="Semantic search over a text label.")
    sp1.add_argument("--q", required=True, help="Text query.")
    sp1.add_argument("--target", choices=list(TEXT_TARGETS.keys()), default="utterance")
    sp1.add_argument("--topk", type=int, default=10)
    sp1.add_argument("--include-emb", action="store_true", help="Return node embeddings.")
    sp1.add_argument("--text-chars", type=int, default=200, help="Max chars of text snippet in table output.")

    sp2 = sub.add_parser("similar-frames", parents=[parent], help="Find visually similar frames via ANN over Frame vectors.")
    g = sp2.add_mutually_exclusive_group(required=True)
    g.add_argument("--frame-id", type=str, help="Seed frame id.")
    g.add_argument("--file-key", type=str, help="Seed file key (use with --frame).")
    sp2.add_argument("--frame", type=int, help="Seed frame number (use with --file-key).")
    sp2.add_argument("--topk", type=int, default=20)
    sp2.add_argument("--include-seed", action="store_true", help="Include the seed frame in results.")

    sp3 = sub.add_parser("geo-frames", parents=[parent], help="Find frames near a lat/lon within R meters (optional time/speed filters).")
    sp3.add_argument("--lat", type=float, required=True)
    sp3.add_argument("--lon", type=float, required=True)
    sp3.add_argument("--radius-m", type=float, default=200.0)
    sp3.add_argument("--start", type=str, default=None, help="ISO8601 or epochMillis (optional).")
    sp3.add_argument("--end", type=str, default=None)
    sp3.add_argument("--min-mph", type=float, default=None)
    sp3.add_argument("--max-mph", type=float, default=None)
    sp3.add_argument("--limit", type=int, default=100)

    args = p.parse_args()

    # Let --model override the module default for this invocation.
    EMBED_MODEL_NAME = args.model
    EMBED_PROP = args.prop

    driver = neo4j_driver()
    if not args.no_index_check:
        ensure_indexes(driver, args.frame_label, args.frame_embed_prop)

    if args.cmd == "search-text":
        rows = search_text(driver, args.q, args.target, args.topk, args.include_emb, args.text_chars)
        if not args.json and not args.csv:
            cols = [("score", "score"), ("label", "label"), ("file/id", "id"), ("text", "text_snip"),
                    ("source", "label")]
            # de-dup column keys (label appears twice) -> use unique
            cols = [("score", "score"), ("label", "label"), ("id", "id"), ("text", "text_snip")]
            print_table(rows, cols)
        maybe_dump(rows, args.json, args.csv)

    elif args.cmd == "similar-frames":
        if not args.frame_id and not (args.file_key and args.frame is not None):
            raise SystemExit("--frame-id OR (--file-key and --frame) required.")
        seed_emb, meta = get_frame_embedding(driver, args.frame_id, args.file_key, args.frame,
                                            args.frame_label, args.frame_embed_prop)
        rows = similar_frames(driver, seed_emb, args.topk, args.include_seed, args.frame_label, args.frame_embed_prop)
        if not args.include_seed and args.frame_id:
            rows = [r for r in rows if r.get("id") != args.frame_id][:args.topk]
        if not args.json and not args.csv:
            cols = [("score", "score"), ("file_key", "file_key"), ("frame", "frame"),
                    ("latitude", "latitude"), ("longitude", "longitude"), ("ts", "ts"), ("mph", "mph"), ("id", "id")]
            print_table(rows, cols)
        maybe_dump(rows, args.json, args.csv)

    elif args.cmd == "geo-frames":
        rows = geo_frames(driver, args.lat, args.lon, args.radius_m, args.start, args.end,
                          args.min_mph, args.max_mph, args.limit)
        if not args.json and not args.csv:
            cols = [("meters", "meters"), ("file_key", "file_key"), ("frame", "frame"),
                    ("latitude", "latitude"), ("longitude", "longitude"), ("ts", "ts"), ("mph", "mph"), ("id", "id")]
            print_table(rows, cols)
        maybe_dump(rows, args.json, args.csv)


if __name__ == "__main__":
    main()
