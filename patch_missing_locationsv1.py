#!/usr/bin/env python3
import os, re, csv, math, argparse, logging, datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
from neo4j import GraphDatabase

# -----------------------
# Logging
# -----------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# -----------------------
# Cypher helpers / indexes
# -----------------------
NQ_CREATE_INDEXES = [
    # Already recommended, but safe to (re)run:
    """
    CREATE CONSTRAINT dashcam_clip_key IF NOT EXISTS
    FOR (c:DashcamClip) REQUIRE c.key IS UNIQUE
    """,
    """
    CREATE INDEX phone_event_time IF NOT EXISTS
    FOR (p:PhoneLog) ON (p.timestamp)
    """,
    """
    CREATE INDEX loc_event_time IF NOT EXISTS
    FOR (l:LocationEvent) ON (l.eventTime)
    """,
    """
    CREATE INDEX frame_by_key_frame IF NOT EXISTS
    FOR (f:Frame) ON (f.key, f.frame)
    """,
    """
    CREATE INDEX dashcam_embedding_sec_lookup IF NOT EXISTS
    FOR (e:DashcamEmbedding) ON (e.key, e.view, e.level, e.t0)
    """,
    # NEW: speeds up selecting targets by loc_source and minute lookups
    """
    CREATE INDEX dashcam_embedding_loc_source IF NOT EXISTS
    FOR (e:DashcamEmbedding) ON (e.loc_source)
    """,
    """
    CREATE INDEX dashcam_embedding_by_key_view_level IF NOT EXISTS
    FOR (e:DashcamEmbedding) ON (e.key, e.view, e.level)
    """,
]

# Candidates (seconds) with no location
NQ_FETCH_CANDIDATE_KEYS = """
MATCH (e:DashcamEmbedding {level:'second'})
WHERE coalesce(e.loc_source,'none') = 'none'
WITH e.key AS key, e.view AS view, collect(e.t0) AS secs
MATCH (c:DashcamClip {key:key})
RETURN key, view, secs, c.path AS path, coalesce(c.fps,30.0) AS fps
ORDER BY key
LIMIT $key_limit
"""

# If filtering to a specific key pattern (prefix or regex)
NQ_FETCH_CANDIDATE_KEYS_FILTERED = """
MATCH (e:DashcamEmbedding {level:'second'})
WHERE coalesce(e.loc_source,'none') = 'none' AND e.key STARTS WITH $key_prefix
WITH e.key AS key, e.view AS view, collect(e.t0) AS secs
MATCH (c:DashcamClip {key:key})
RETURN key, view, secs, c.path AS path, coalesce(c.fps,30.0) AS fps
ORDER BY key
LIMIT $key_limit
"""

# Fetch detail for a single embedding (to patch vec tail safely)
NQ_GET_EMBED_DETAIL = """
MATCH (e:DashcamEmbedding {id:$eid})
RETURN e.vec AS vec, coalesce(e.loc_dim,0) AS loc_dim, e.l2 AS l2, e.dim AS dim, e.key AS key, e.view AS view
"""

# Update embedding loc fields + full vec
NQ_UPDATE_EMBED = """
MATCH (e:DashcamEmbedding {id:$eid})
SET e.lat = $lat, e.lon = $lon, e.mph = $mph,
    e.loc_vec = $loc_vec, e.loc_dim = $loc_dim, e.loc_source = $loc_source,
    e.vec = $vec, e.updated_at = timestamp()
RETURN e
"""

# Create NEAR link to an existing node by elementId (PhoneLog/LocationEvent/etc.)
NQ_ATTACH_NEAR_BY_ELEMID = """
MATCH (e:DashcamEmbedding {id:$eid})
MATCH (n) WHERE elementId(n) = $elem_id
MERGE (e)-[:NEAR {seconds:$seconds, source:$source}]->(n)
"""

# Bulk nearest PhoneLog for many timestamps (ISO strings)
NQ_BULK_NEAREST_PHONELOG = """
UNWIND $times AS t
WITH datetime(t) AS et0, t
MATCH (p:PhoneLog)
WHERE p.timestamp >= et0 - duration({minutes:$win_mins})
  AND p.timestamp <= et0 + duration({minutes:$win_mins})
WITH t, p, abs(duration.between(p.timestamp, et0).seconds) AS dt
ORDER BY t, dt ASC
WITH t, collect({elem_id:elementId(p), lat:p.lat, lon:p.lon, mph:toFloat(p.mph), seconds:dt})[0] AS best
RETURN t AS t_iso, best
"""

# Get all seconds loc_vec for recomputing minute
NQ_FETCH_SECONDS_LOCVECS = """
MATCH (e:DashcamEmbedding {key:$key, view:$view, level:'second'})
RETURN e.loc_vec AS loc_vec
"""

# Fetch minute embedding for a key/view (if present)
NQ_GET_MINUTE_EMBED = """
MATCH (m:DashcamEmbedding {key:$key, view:$view, level:'minute'})
RETURN m.id AS id, m.vec AS vec, coalesce(m.loc_dim,0) AS loc_dim
"""

# Update minute embedding
NQ_UPDATE_MINUTE = """
MATCH (m:DashcamEmbedding {id:$mid})
SET m.loc_vec = $loc_vec, m.lat = $lat, m.lon = $lon, m.mph = $mph,
    m.vec = $vec, m.loc_source = 'mixed', m.updated_at = timestamp()
RETURN m
"""

# -----------------------
# Time / vector helpers
# -----------------------
def parse_key_datetime(key: str) -> Optional[datetime.datetime]:
    # key like YYYY_MMDD_HHMMSS_[F|R|FR]
    m = re.match(r"^(\d{4})_(\d{4})_(\d{6})", key)
    if not m: return None
    y = int(m.group(1))
    mm = int(m.group(2)[:2]); dd = int(m.group(2)[2:])
    HH = int(m.group(3)[:2]); MM = int(m.group(3)[2:4]); SS = int(m.group(3)[4:])
    try:
        return datetime.datetime(y, mm, dd, HH, MM, SS, tzinfo=datetime.timezone.utc)
    except Exception:
        return None

def clip_base_key(key: str) -> str:
    return re.sub(r'_(F|R|FR)$', '', key)

def ensure_unit_l2(vec: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    if n == 0.0 or not np.isfinite(n): return vec
    return (vec / n).astype(np.float32)

def cyc(v: float, period: float) -> Tuple[float,float]:
    ang = 2*np.pi*(v/period)
    return float(np.sin(ang)), float(np.cos(ang))

def location_feature(lat, lon, mph, t_utc, include_time=True):
    slat=clat=slon=clon=0.0
    if lat is not None and lon is not None:
        lat_wrapped = (float(lat) + 90.0) / 180.0
        lon_wrapped = (float(lon) + 180.0) / 360.0
        slat, clat = cyc(lat_wrapped, 1.0)
        slon, clon = cyc(lon_wrapped, 1.0)
    mph_norm = 0.0
    if mph is not None and np.isfinite(mph):
        mph_norm = max(0.0, min(1.0, float(mph)/80.0))
    tod_sin = tod_cos = 0.0
    if include_time and t_utc is not None:
        secs = t_utc.hour*3600 + t_utc.minute*60 + t_utc.second
        tod_sin, tod_cos = cyc(secs, 86400.0)
    vec = np.array([slat,clat, slon,clon, mph_norm, tod_sin, tod_cos], dtype=np.float32)
    scalars = {"lat": (float(lat) if lat is not None else None),
               "lon": (float(lon) if lon is not None else None),
               "mph": (float(mph) if mph is not None else None)}
    return vec, scalars

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = phi2 - phi1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi/2.0)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dl/2.0)**2
    return float(2*R*np.arcsin(np.sqrt(a)))

def plausible_fix(lat, lon, mph, max_speed=110.0):
    if lat is None or lon is None: return False
    if not (np.isfinite(lat) and np.isfinite(lon)): return False
    if mph is not None and (mph < 0 or mph > max_speed): return False
    return True

# -----------------------
# Metadata: folder-level "YYYY_MMDD_HHMMSS_metadata.csv"
# -----------------------
def coerce_float(val) -> Optional[float]:
    if val is None: return None
    s = str(val).strip()
    if s == "": return None
    try:
        return float(s)
    except Exception:
        # e.g. "O0O" → 0
        digits = "".join(ch for ch in s if ch.isdigit() or ch in ".-")
        try:
            return float(digits) if digits not in ("", ".", "-", "-.", ".-") else None
        except Exception:
            return None

def read_base_metadata_csv(folder: str, base_key: str, fps: float) -> Dict[int, Dict[str, Optional[float]]]:
    """
    Reads {base_key}_metadata.csv with columns like Key,MPH,Lat,Long,Frame (case-insensitive).
    Returns mapping: sec -> {lat, lon, mph} averaged over frames in that second.
    """
    path = os.path.join(folder, f"{base_key}_metadata.csv")
    if not os.path.exists(path):
        return {}
    by_sec: Dict[int, Dict[str, List[float]]] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        # normalize header keys to lowercase
        field_map = {k.lower(): k for k in reader.fieldnames or []}
        k_mph = field_map.get("mph") or field_map.get("speed")
        k_lat = field_map.get("lat") or field_map.get("latitude")
        k_lon = field_map.get("long") or field_map.get("lon") or field_map.get("longitude")
        k_frame = field_map.get("frame")
        if not k_frame:
            return {}
        for row in reader:
            frame = coerce_float(row.get(k_frame))
            if frame is None: continue
            sec = int(max(0, math.floor(frame / max(fps, 1.0))))
            lat = coerce_float(row.get(k_lat)) if k_lat else None
            lon = coerce_float(row.get(k_lon)) if k_lon else None
            mph = coerce_float(row.get(k_mph)) if k_mph else None
            d = by_sec.setdefault(sec, {"lat": [], "lon": [], "mph": []})
            if lat is not None: d["lat"].append(lat)
            if lon is not None: d["lon"].append(lon)
            if mph is not None: d["mph"].append(mph)
    # reduce
    out: Dict[int, Dict[str, Optional[float]]] = {}
    for s, d in by_sec.items():
        lat = float(np.nanmean(d["lat"])) if d["lat"] else None
        lon = float(np.nanmean(d["lon"])) if d["lon"] else None
        mph = float(np.nanmean(d["mph"])) if d["mph"] else None
        out[s] = {"lat": lat, "lon": lon, "mph": mph}
    return out

# -----------------------
# Neo4j access
# -----------------------
def neo4j_session(uri: str, user: str, pwd: str):
    return GraphDatabase.driver(uri, auth=(user, pwd))

def create_indexes(sess):
    for stmt in NQ_CREATE_INDEXES:
        try:
            sess.run(stmt)
        except Exception as e:
            logging.warning(f"Index creation failed or already exists: {e}")

def fetch_candidate_keys(sess, key_limit: int, key_prefix: Optional[str]) -> List[dict]:
    if key_prefix:
        res = sess.run(NQ_FETCH_CANDIDATE_KEYS_FILTERED, key_limit=int(key_limit), key_prefix=key_prefix)
    else:
        res = sess.run(NQ_FETCH_CANDIDATE_KEYS, key_limit=int(key_limit))
    return [dict(r) for r in res]

def bulk_nearest_phonelog(sess, times_iso: List[str], win_mins: int) -> Dict[str, Optional[dict]]:
    if not times_iso:
        return {}
    recs = sess.run(NQ_BULK_NEAREST_PHONELOG, times=times_iso, win_mins=int(win_mins))
    out = {}
    for r in recs:
        best = r["best"]
        out[r["t_iso"]] = (dict(best) if best and best.get("elem_id") else None)
    return out

def get_embed_detail(sess, emb_id: str) -> Tuple[List[float], int, bool, int, str, str]:
    r = sess.run(NQ_GET_EMBED_DETAIL, eid=emb_id).single()
    if not r:
        raise RuntimeError(f"Embed not found: {emb_id}")
    return (r["vec"] or [], int(r["loc_dim"] or 0), bool(r["l2"]), int(r["dim"] or 0), r["key"], r["view"])

def update_embed(sess, *, emb_id: str, vec: List[float], loc_vec: List[float],
                 lat: Optional[float], lon: Optional[float], mph: Optional[float], source: str):
    sess.run(NQ_UPDATE_EMBED,
             eid=emb_id,
             vec=vec,
             loc_vec=loc_vec,
             loc_dim=len(loc_vec),
             lat=lat, lon=lon, mph=mph,
             loc_source=source)

def attach_near(sess, *, emb_id: str, elem_id: str, seconds: int, source: str):
    sess.run(NQ_ATTACH_NEAR_BY_ELEMID,
             eid=emb_id, elem_id=elem_id, seconds=int(seconds), source=source)

def fetch_seconds_locvecs(sess, key: str, view: str) -> List[List[float]]:
    res = sess.run(NQ_FETCH_SECONDS_LOCVECS, key=key, view=view)
    return [r["loc_vec"] for r in res if r.get("loc_vec")]

def get_minute_embed(sess, key: str, view: str) -> Optional[dict]:
    r = sess.run(NQ_GET_MINUTE_EMBED, key=key, view=view).single()
    return (dict(r) if r else None)

def update_minute(sess, *, mid: str, vec: List[float], loc_vec: List[float],
                  lat: Optional[float], lon: Optional[float], mph: Optional[float]):
    sess.run(NQ_UPDATE_MINUTE,
             mid=mid, vec=vec, loc_vec=loc_vec, lat=lat, lon=lon, mph=mph)

# -----------------------
# Core patch logic
# -----------------------
def choose_location(meta: Optional[dict], phonelog: Optional[dict],
                    t_utc: datetime.datetime,
                    max_validate_dist_m: float) -> Tuple[Optional[float], Optional[float], Optional[float], str, Optional[int], Optional[str]]:
    """
    Returns (lat, lon, mph, source, seconds_delta, near_elem_id)
    Policy:
      - If metadata is plausible and (no PL OR within max_validate_dist_m of PL) -> metadata (validated if PL present)
      - Else if PhoneLog exists -> PhoneLog
      - Else -> none
    """
    # phone log candidate if present
    if phonelog and phonelog.get("lat") is not None and phonelog.get("lon") is not None:
        pl_lat = float(phonelog["lat"]); pl_lon = float(phonelog["lon"])
        pl_mph = float(phonelog.get("mph")) if phonelog.get("mph") is not None else None
        pl_seconds = int(phonelog.get("seconds") or 0)
        pl_elem = phonelog.get("elem_id")

    else:
        pl_lat = pl_lon = pl_mph = None
        pl_seconds = 0
        pl_elem = None

    # metadata candidate?
    if meta and plausible_fix(meta.get("lat"), meta.get("lon"), meta.get("mph")):
        # if phonelog exists, validate distance
        if pl_lat is not None and pl_lon is not None and meta.get("lat") is not None and meta.get("lon") is not None:
            try:
                d = haversine_m(meta["lat"], meta["lon"], pl_lat, pl_lon)
            except Exception:
                d = float("inf")
            if d <= max_validate_dist_m:
                return (meta["lat"], meta["lon"], meta.get("mph"), "metadata_csv_validated", None, None)
            # Far → prefer PhoneLog
            if pl_lat is not None:
                return (pl_lat, pl_lon, pl_mph, "PhoneLog", pl_seconds, pl_elem)
        # No PhoneLog to compare -> use metadata
        return (meta["lat"], meta["lon"], meta.get("mph"), "metadata_csv", None, None)

    # No plausible metadata → PhoneLog if available
    if pl_lat is not None and pl_lon is not None:
        return (pl_lat, pl_lon, pl_mph, "PhoneLog", pl_seconds, pl_elem)

    # Nothing
    return (None, None, None, "none", None, None)

def patch_seconds_for_key(sess, *, key: str, view: str, secs: List[int], clip_path: str,
                          fps: float, win_mins: int, validate_m: float, dry_run: bool) -> int:
    """
    Update loc for the provided seconds of a single (key,view).
    Returns count of seconds patched.
    """
    folder = os.path.dirname(clip_path)
    base_key = clip_base_key(key)
    dt0 = parse_key_datetime(key) or datetime.datetime.fromtimestamp(0, tz=datetime.timezone.utc)

    # Load folder-level metadata once
    meta_by_sec = read_base_metadata_csv(folder, base_key, fps)

    # Build timestamps, bulk fetch nearest PhoneLogs
    times_iso = []
    sec_index = []
    for s in sorted(secs):
        t_utc = dt0 + datetime.timedelta(seconds=int(s))
        times_iso.append(t_utc.isoformat())
        sec_index.append(s)
    pl_lookup = bulk_nearest_phonelog(sess, times_iso, win_mins)

    patched = 0
    for s, t_iso in zip(sec_index, times_iso):
        pl = pl_lookup.get(t_iso)
        meta = meta_by_sec.get(s)
        t_utc = dt0 + datetime.timedelta(seconds=int(s))

        lat, lon, mph, source, seconds_delta, near_elem = choose_location(meta, pl, t_utc, validate_m)

        if source == "none":
            continue  # nothing to write

        emb_id = f"{key}|{view}|sec|{s}"

        # Fetch vec + loc_dim to patch only the tail
        vec, loc_dim, _l2, _dim, _k, _v = get_embed_detail(sess, emb_id)
        if loc_dim <= 0:
            # If no dedicated loc tail recorded, we can't patch vec safely; still update loc_* properties.
            loc_dim = 7

        # Compute loc_vec
        loc_vec_np, _ = location_feature(lat, lon, mph, t_utc, include_time=True)
        loc_vec = loc_vec_np.astype(np.float32).tolist()

        # Patch vec tail if sized correctly
        if vec and len(vec) >= loc_dim:
            new_vec = list(vec)
            new_vec[-loc_dim:] = loc_vec
        else:
            # Fallback: leave vec as-is if dimensions are unexpected
            new_vec = vec

        if dry_run:
            logging.info(f"[DRY] {emb_id} -> source={source} lat={lat} lon={lon} mph={mph}")
        else:
            update_embed(sess, emb_id=emb_id, vec=new_vec, loc_vec=loc_vec,
                         lat=lat, lon=lon, mph=mph, source=source)
            if near_elem is not None and seconds_delta is not None:
                attach_near(sess, emb_id=emb_id, elem_id=near_elem,
                            seconds=int(seconds_delta), source=source)
        patched += 1

    return patched

def recompute_minute_for_key(sess, *, key: str, view: str, dry_run: bool):
    m = get_minute_embed(sess, key, view)
    if not m:
        logging.info(f"[minute] No minute node for {key}|{view}; skipping.")
        return
    loc_vecs = fetch_seconds_locvecs(sess, key, view)
    if not loc_vecs:
        logging.info(f"[minute] No seconds loc_vecs for {key}|{view}; skipping.")
        return
    # Aggregate (mean) then L2
    M = np.array(loc_vecs, dtype=np.float32)
    mloc = np.nanmean(M, axis=0)
    mloc = ensure_unit_l2(mloc).astype(np.float32)
    # Summary scalars: first non-None lat/lon/mph (best-effort)
    lat = lon = mph = None
    # Try reverse order (more recent first)
    for v in reversed(loc_vecs):
        # We cannot recover scalars from loc_vec alone. Keep scalars None to avoid lying.
        # (They’re not used for retrieval; loc_vec is.)
        break
    # Patch minute vec tail
    vec = m["vec"] or []
    loc_dim = int(m["loc_dim"] or 7)
    if vec and len(vec) >= loc_dim:
        new_vec = list(vec)
        new_vec[-loc_dim:] = mloc.tolist()
    else:
        new_vec = vec

    if dry_run:
        logging.info(f"[DRY] minute {key}|{view} updated (loc_dim={loc_dim})")
    else:
        update_minute(sess, mid=m["id"], vec=new_vec, loc_vec=mloc.tolist(),
                      lat=lat, lon=lon, mph=mph)

# -----------------------
# CLI runner
# -----------------------
def main():
    ap = argparse.ArgumentParser(description="Patch DashcamEmbedding nodes missing location using folder metadata + PhoneLog fallback")
    ap.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    ap.add_argument("--neo4j-user", default="neo4j")
    ap.add_argument("--neo4j-pass", default="password")
    ap.add_argument("--key-limit", type=int, default=500, help="Max (key,view) groups to process")
    ap.add_argument("--key-prefix", default=None, help="Optional key prefix filter, e.g. 2025_0405")
    ap.add_argument("--win-mins", type=int, default=10, help="± minutes for nearest PhoneLog search")
    ap.add_argument("--validate-m", type=float, default=50.0, help="Max distance to validate metadata against PhoneLog")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-minute", action="store_true", help="Do not recompute minute loc_vec after patching seconds")
    args = ap.parse_args()

    driver = neo4j_session(args.neo4j_uri, args.neo4j_user, args.neo4j_pass)
    patched_total = 0
    keys_processed = 0
    with driver.session() as sess:
        create_indexes(sess)

        rows = fetch_candidate_keys(sess, args.key_limit, args.key_prefix)
        if not rows:
            logging.info("No candidate seconds with loc_source=none found.")
            return

        for row in rows:
            key = row["key"]; view = row["view"]
            secs = sorted(int(s) for s in (row["secs"] or []))
            if not secs:
                continue
            clip_path = row["path"]; fps = float(row["fps"] or 30.0)
            logging.info(f"[{key}|{view}] patching {len(secs)} seconds (fps={fps})")

            count = patch_seconds_for_key(
                sess, key=key, view=view, secs=secs, clip_path=clip_path, fps=fps,
                win_mins=args.win_mins, validate_m=args.validate_m, dry_run=args.dry_run
            )
            patched_total += count
            keys_processed += 1

            if not args.no-minute and count > 0 and not args.dry_run:
                recompute_minute_for_key(sess, key=key, view=view, dry_run=args.dry_run)

    driver.close()
    logging.info(f"Done. keys_processed={keys_processed}, seconds_patched={patched_total}")

if __name__ == "__main__":
    main()
