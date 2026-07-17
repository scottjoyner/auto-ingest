#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
recall.py
Personal "memory recall" CLI over the user's own iPhone photos/videos
stored as :MediaFile nodes in Neo4j.

Subcommands
-----------
1) similar-media:
   ANN similarity for MediaFile.embedding, seeded by sha256 or file path.

2) recall:
   Natural-language text query -> CLIP text embedding -> ANN over
   MediaFile.text_embedding, then filtered by graph links (AT_PLACE /
   DURING / NEAR) and an optional date range.

3) geo-media:
   Radius search by gps_lat/gps_lon (haversine) on MediaFile.

Outputs
-------
- Pretty table (default)
- JSON (--json)
- CSV  (--csv PATH)

Environment
-----------
NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DB
"""

import argparse
import csv
import hashlib
import json
import logging
import math
from typing import Any, Dict, List, Optional

try:
    from auto_ingest_config import get_neo4j_env
except Exception:  # packaged import fallback
    from auto_ingest._config import get_neo4j_env

from auto_ingest.personal.embed import (
    CLIP_DIM,
    embed_text,
    ensure_media_indexes_with_retry,
)
from auto_ingest.shorts.db_retry import with_driver

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DB = get_neo4j_env()

MEDIA_LABEL = "MediaFile"


# =========================
# Helpers
# =========================
def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if math.isnan(v):
            return ""
        return f"{v:.4f}" if abs(v) < 1000 else f"{v:.2f}"
    return str(v)


def print_table(rows: List[Dict[str, Any]], cols: List[tuple]):
    if not rows:
        print("(no results)")
        return
    headers = [h for h, _ in cols]
    widths = []
    for i, (_, k) in enumerate(cols):
        w = max(len(headers[i]), max(len(_fmt(r.get(k, ""))) for r in rows))
        widths.append(w)
    line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    sep = "-+-".join("-" * widths[i] for i in range(len(widths)))
    print(line)
    print(sep)
    for r in rows:
        print(" | ".join(_fmt(r.get(k, "")).ljust(widths[i]) for i, (_, k) in enumerate(cols)))


def maybe_dump(rows: List[Dict[str, Any]], as_json: bool, csv_path: Optional[str],
              csv_cols: Optional[List[str]] = None):
    if as_json:
        print(json.dumps(rows, ensure_ascii=False, indent=2, default=str))
    if csv_path:
        keys = csv_cols or sorted({k for r in rows for k in r.keys()})
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in keys})


def _resolve_sha256(driver, sha256: Optional[str], path: Optional[str]) -> str:
    if path:
        return sha256_of(path)
    if not sha256:
        raise ValueError("One of --sha256 or --path is required.")
    return sha256


# =========================
# 1) similar-media
# =========================
def get_media_embedding(driver, sha256: str) -> List[float]:
    with driver.session(database=NEO4J_DB) as sess:
        rec = sess.run(
            f"MATCH (m:{MEDIA_LABEL} {{sha256:$s}}) RETURN m.embedding AS emb",
            s=sha256,
        ).single()
    if rec is None:
        raise RuntimeError(f"MediaFile with sha256={sha256} not found.")
    emb = rec.get("emb")
    if emb is None:
        raise RuntimeError(
            f"MediaFile sha256={sha256} has no 'embedding' property. "
            f"Ensure CLIP image embedding was ingested."
        )
    return list(emb)


def similar_media(driver, seed_sha256: str, top_k: int, include_seed: bool) -> List[Dict[str, Any]]:
    seed_emb = get_media_embedding(driver, seed_sha256)
    cypher = """
    CALL db.index.vector.queryNodes('media_embedding_index', $k, $qvec)
      YIELD node, score
    WHERE 'MediaFile' IN labels(node)
    RETURN
      node.sha256 AS sha256,
      node.path    AS path,
      node.kind    AS kind,
      node.date    AS date,
      node.gps_lat AS lat,
      node.gps_lon AS lon,
      score        AS score
    ORDER BY score DESC
    """
    with driver.session(database=NEO4J_DB) as sess:
        res = sess.run(cypher, k=top_k + (0 if include_seed else 1), qvec=seed_emb)
        rows = [r.data() for r in res]
    if not include_seed:
        rows = [r for r in rows if r.get("sha256") != seed_sha256][:top_k]
    else:
        rows = rows[:top_k]
    return rows


# =========================
# 2) recall
# =========================
def recall_media(driver, text: str, top_k: int,
                place: Optional[str] = None, trip: Optional[str] = None,
                since: Optional[str] = None, until: Optional[str] = None,
                kind: Optional[str] = None) -> List[Dict[str, Any]]:
    ensure_media_indexes_with_retry()
    qvec = embed_text(text)
    if qvec is None:
        raise RuntimeError("Text embedding failed (embed_text returned None).")
    if len(qvec) != CLIP_DIM:
        raise RuntimeError(f"embed_text returned {len(qvec)} dims, expected {CLIP_DIM}.")

    cypher = """
    CALL db.index.vector.queryNodes('media_text_embedding_index', $k, $qvec)
      YIELD node, score
    WHERE 'MediaFile' IN labels(node)
    WITH node AS m, score
    WHERE ($kind IS NULL OR m.kind = $kind)
      AND ($since IS NULL OR m.date >= $since)
      AND ($until IS NULL OR m.date <= $until)
    OPTIONAL MATCH (m)-[:AT_PLACE]->(p:SummaryPlace)
      WHERE $place IS NULL OR p.name = $place
    OPTIONAL MATCH (m)-[:DURING]->(t:Trip)
      WHERE $trip IS NULL OR t.uniqueKey = $trip
    WITH m, score,
         collect(DISTINCT p.name)[0] AS place_name,
         collect(DISTINCT t.uniqueKey)[0] AS trip_key
    WHERE ($place IS NULL OR place_name IS NOT NULL)
      AND ($trip IS NULL OR trip_key IS NOT NULL)
    RETURN
      m.sha256 AS sha256,
      m.path    AS path,
      m.kind    AS kind,
      m.date    AS date,
      m.gps_lat AS lat,
      m.gps_lon AS lon,
      score     AS score,
      place_name AS place,
      trip_key   AS trip
    ORDER BY score DESC
    LIMIT $lim
    """
    with driver.session(database=NEO4J_DB) as sess:
        res = sess.run(
            cypher,
            k=top_k * 4,
            qvec=qvec,
            kind=kind,
            since=since,
            until=until,
            place=place,
            trip=trip,
            lim=top_k,
        )
        return [r.data() for r in res]


# =========================
# 3) geo-media
# =========================
def geo_media(driver, lat: float, lon: float, radius_m: float = 200.0,
             since: Optional[str] = None, until: Optional[str] = None,
             kind: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    cypher = """
    WITH $lat AS lat0, $lon AS lon0, $radius AS R
    MATCH (m:MediaFile)
    WHERE m.gps_lat IS NOT NULL AND m.gps_lon IS NOT NULL
      AND m.gps_lat >= lat0 - (R/111320.0) AND m.gps_lat <= lat0 + (R/111320.0)
      AND m.gps_lon >= lon0 - (R/(111320.0 * cos(radians(lat0))))
      AND m.gps_lon <= lon0 + (R/(111320.0 * cos(radians(lat0))))
    WITH m, lat0, lon0, R,
         6371000.0 * 2 * asin(sqrt(
             pow(sin(radians((m.gps_lat-lat0)/2)),2)
             + cos(radians(lat0))*cos(radians(m.gps_lat))
               * pow(sin(radians((m.gps_lon-lon0)/2)),2) )) AS dist
    WHERE dist <= R
      AND ($kind IS NULL OR m.kind = $kind)
      AND ($since IS NULL OR m.date >= $since)
      AND ($until IS NULL OR m.date <= $until)
    RETURN
      m.sha256 AS sha256,
      m.path    AS path,
      m.kind    AS kind,
      m.date    AS date,
      m.gps_lat AS lat,
      m.gps_lon AS lon,
      dist      AS meters
    ORDER BY dist ASC
    LIMIT $limit
    """
    with driver.session(database=NEO4J_DB) as sess:
        res = sess.run(
            cypher,
            lat=float(lat),
            lon=float(lon),
            radius=float(radius_m),
            kind=kind,
            since=since,
            until=until,
            limit=int(limit),
        )
        return [r.data() for r in res]


# =========================
# CLI
# =========================
def main():
    p = argparse.ArgumentParser(description="Personal media recall (vector + geo).")
    p.add_argument("--json", action="store_true", help="Output JSON instead of table.")
    p.add_argument("--csv", type=str, default=None, help="Also write results to this CSV path.")

    sub = p.add_subparsers(dest="cmd", required=True)

    # 1) similar-media
    sp1 = sub.add_parser("similar-media", help="Find visually similar MediaFile via ANN.")
    g = sp1.add_mutually_exclusive_group(required=True)
    g.add_argument("--sha256", type=str, help="Seed MediaFile sha256.")
    g.add_argument("--path", type=str, help="Seed MediaFile path (content-hashed to sha256).")
    sp1.add_argument("--topk", type=int, default=12)
    sp1.add_argument("--include-seed", action="store_true", help="Include the seed in results.")

    # 2) recall
    sp2 = sub.add_parser("recall", help="Natural-language text recall over MediaFile.")
    sp2.add_argument("--text", required=True, help="Natural-language query.")
    sp2.add_argument("--topk", type=int, default=12)
    sp2.add_argument("--place", type=str, default=None, help="SummaryPlace name filter.")
    sp2.add_argument("--trip", type=str, default=None, help="Trip uniqueKey filter.")
    sp2.add_argument("--since", type=str, default=None, help="YYYY-MM-DD lower bound (m.date).")
    sp2.add_argument("--until", type=str, default=None, help="YYYY-MM-DD upper bound (m.date).")
    sp2.add_argument("--kind", choices=["movie", "picture"], default=None)

    # 3) geo-media
    sp3 = sub.add_parser("geo-media", help="MediaFile within radius of lat/lon.")
    sp3.add_argument("--lat", type=float, required=True)
    sp3.add_argument("--lon", type=float, required=True)
    sp3.add_argument("--radius-m", type=float, default=200.0)
    sp3.add_argument("--since", type=str, default=None)
    sp3.add_argument("--until", type=str, default=None)
    sp3.add_argument("--kind", choices=["movie", "picture"], default=None)
    sp3.add_argument("--limit", type=int, default=100)

    args = p.parse_args()

    if args.cmd == "similar-media":
        seed = _resolve_sha256(None, getattr(args, "sha256", None), getattr(args, "path", None))

        def run(drv):
            return similar_media(drv, seed, args.topk, args.include_seed)

        rows = with_driver(run)
        if rows is None:
            raise SystemExit("similar-media failed (see logs).")
        if not args.json and not args.csv:
            print_table(rows, [
                ("score", "score"),
                ("sha256", "sha256"),
                ("kind", "kind"),
                ("date", "date"),
                ("lat", "lat"),
                ("lon", "lon"),
                ("path", "path"),
            ])
        maybe_dump(rows, args.json, args.csv,
                   ["sha256", "path", "kind", "date", "lat", "lon", "score"])

    elif args.cmd == "recall":
        def run(drv):
            return recall_media(drv, args.text, args.topk,
                                place=args.place, trip=args.trip,
                                since=args.since, until=args.until, kind=args.kind)

        rows = with_driver(run)
        if rows is None:
            raise SystemExit("recall failed (see logs).")
        if not args.json and not args.csv:
            print_table(rows, [
                ("score", "score"),
                ("sha256", "sha256"),
                ("kind", "kind"),
                ("date", "date"),
                ("lat", "lat"),
                ("lon", "lon"),
                ("place", "place"),
                ("trip", "trip"),
                ("path", "path"),
            ])
        maybe_dump(rows, args.json, args.csv,
                   ["sha256", "path", "kind", "date", "lat", "lon", "score", "place", "trip"])

    elif args.cmd == "geo-media":
        def run(drv):
            return geo_media(drv, args.lat, args.lon, radius_m=args.radius_m,
                             since=args.since, until=args.until,
                             kind=args.kind, limit=args.limit)

        rows = with_driver(run)
        if rows is None:
            raise SystemExit("geo-media failed (see logs).")
        if not args.json and not args.csv:
            print_table(rows, [
                ("meters", "meters"),
                ("sha256", "sha256"),
                ("kind", "kind"),
                ("date", "date"),
                ("lat", "lat"),
                ("lon", "lon"),
                ("path", "path"),
            ])
        maybe_dump(rows, args.json, args.csv,
                   ["sha256", "path", "kind", "date", "lat", "lon", "meters"])


if __name__ == "__main__":
    main()
