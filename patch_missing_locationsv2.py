# v2 includes the additional system with locations from linxup if they exist
#!/usr/bin/env python3
import os, re, csv, math, argparse, logging, datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# -----------------------
# DB indexes / constraints
# -----------------------
NQ_CREATE_INDEXES = [
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
    # Speed up filtering/joins done here
    """
    CREATE INDEX dashcam_embedding_loc_source IF NOT EXISTS
    FOR (e:DashcamEmbedding) ON (e.loc_source)
    """,
    """
    CREATE INDEX dashcam_embedding_by_key_view_level IF NOT EXISTS
    FOR (e:DashcamEmbedding) ON (e.key, e.view, e.level)
    """
]

# -----------------------
# Candidate selection
# -----------------------
NQ_FETCH_CANDIDATE_KEYS = """
MATCH (e:DashcamEmbedding {level:'second'})
WHERE coalesce(e.loc_source,'none') = 'none'
WITH e.key AS key, e.view AS view, collect(e.t0) AS secs
MATCH (c:DashcamClip {key:key})
RETURN key, view, secs, c.path AS path, coalesce(c.fps,30.0) AS fps
ORDER BY key
LIMIT $key_limit
"""

NQ_FETCH_CANDIDATE_KEYS_FILTERED = """
MATCH (e:DashcamEmbedding {level:'second'})
WHERE coalesce(e.loc_source,'none') = 'none' AND e.key STARTS WITH $key_prefix
WITH e.key AS key, e.view AS view, collect(e.t0) AS secs
MATCH (c:DashcamClip {key:key})
RETURN key, view, secs, c.path AS path, coalesce(c.fps,30.0) AS fps
ORDER BY key
LIMIT $key_limit
"""

# -----------------------
# Bulk nearest lookups
# -----------------------
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

# NOTE: we search at t, t+1h, t-1h and pick the closest. This soaks DST drift.
NQ_BULK_NEAREST_LOCEVENT = """
UNWIND $times AS t
UNWIND [0, 3600, -3600] AS off
WITH datetime(t) + duration({seconds: off}) AS et0, t, off
MATCH (l:LocationEvent)-[:BELONGS_TO]->(:Trip)
WHERE l.eventTime >= et0 - duration({minutes:$win_mins})
  AND l.eventTime <= et0 + duration({minutes:$win_mins})
WITH t, off, l, abs(duration.between(l.eventTime, et0).seconds) AS dt
ORDER BY t, dt ASC
WITH t, collect({
  elem_id: elementId(l),
  lat: toFloat(l.latitude),
  lon: toFloat(l.longitude),
  mph: toFloat(coalesce(l.speed, replace(l.speed,'mph',''))),
  seconds: dt,
  offset: off
})[0] AS best
RETURN t AS t_iso, best
"""

# -----------------------
# Embedding read/write
# -----------------------
NQ_GET_EMBED_DETAIL = """
MATCH (e:DashcamEmbedding {id:$eid})
RETURN e.vec AS vec, coalesce(e.loc_dim,0) AS loc_dim, e.l2 AS l2, e.dim AS dim, e.key AS key, e.view AS view
"""

NQ_UPDATE_EMBED = """
MATCH (e:DashcamEmbedding {id:$eid})
SET e.lat = $lat, e.lon = $lon, e.mph = $mph,
    e.loc_vec = $loc_vec, e.loc_dim = $loc_dim, e.loc_source = $loc_source,
    e.vec = $vec, e.updated_at = timestamp()
RETURN e
"""

NQ_ATTACH_NEAR_BY_ELEMID = """
MATCH (e:DashcamEmbedding {id:$eid})
MATCH (n) WHERE elementId(n) = $elem_id
MERGE (e)-[:NEAR {seconds:$seconds, source:$source}]->(n)
"""

# Minute helpers
NQ_FETCH_SECONDS_LOCVECS = """
MATCH (e:DashcamEmbedding {key:$key, view:$view, level:'second'})
RETURN e.loc_vec AS loc_vec
"""

NQ_GET_MINUTE_EMBED = """
MATCH (m:DashcamEmbedding {key:$key, view:$view, level:'minute'})
RETURN m.id AS id, m.vec AS vec, coalesce(m.loc_dim,0) AS loc_dim
"""

NQ_UPDATE_MINUTE = """
MATCH (m:DashcamEmbedding {id:$mid})
SET m.loc_vec = $loc_vec, m.lat = $lat, m.lon = $lon, m.mph = $mph,
    m.vec = $vec, m.loc_source = 'mixed', m.updated_at = timestamp()
RETURN m
"""

# -----------------------
# Utilities
# -----------------------
def neo4j_session(uri: str, user: str, pwd: str):
    return GraphDatabase.driver(uri, auth=(user, pwd))

def create_indexes(sess):
    for stmt in NQ_CREATE_INDEXES:
        try:
            sess.run(stmt)
        except Exception as e:
            logging.warning(f"Index creation failed or already exists: {e}")

def parse_key_datetime(key: str) -> Optional[datetime.datetime]:
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

def coerce_float(val) -> Optional[float]:
    if val is None: return None
    s = str(val).strip()
    if s == "": return None
    try:
        return float(s)
    except Exception:
        # tolerate garbage like "O0O"
        digits = "".join(ch for ch in s if ch.isdigit() or ch in ".-")
        try:
            return float(digits) if digits not in ("", ".", "-", "-.", ".-") else None
        except Exception:
            return None

# -----------------------
# Folder-level metadata CSV: {YYYY_MMDD_HHMMSS}_metadata.csv
# -----------------------
def read_base_metadata_csv(folder: str, base_key: str, fps: float) -> Dict[int, Dict[str, Optional[float]]]:
    path = os.path.join(folder, f"{base_key}_metadata.csv")
    if not os.path.exists(path):
        return {}
    by_sec: Dict[int, Dict[str, List[float]]] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        headers = {k.lower(): k for k in (reader.fieldnames or [])}
        k_mph = headers.get("mph") or headers.get("speed")
        k_lat = headers.get("lat") or headers.get("latitude")
        k_lon = headers.get("long") or headers.get("lon") or headers.get("longitude")
        k_frame = headers.get("frame")
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
    out: Dict[int, Dict[str, Optional[float]]] = {}
    for s, d in by_sec.items():
        lat = float(np.nanmean(d["lat"])) if d["lat"] else None
        lon = float(np.nanmean(d["lon"])) if d["lon"] else None
        mph = float(np.nanmean(d["mph"])) if d["mph"] else None
        out[s] = {"lat": lat, "lon": lon, "mph": mph}
    return out

# -----------------------
# Neo4j helpers
# -----------------------
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

def bulk_nearest_locevent(sess, times_iso: List[str], win_mins: int) -> Dict[str, Optional[dict]]:
    if not times_iso:
        return {}
    recs = sess.run(NQ_BULK_NEAREST_LOCEVENT, times=times_iso, win_mins=int(win_mins))
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
             eid=emb_id, vec=vec, loc_vec=loc_vec, loc_dim=len(loc_vec),
             lat=lat, lon=lon, mph=mph, loc_source=source)

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
# Selection policy (now with LocationEvent)
# -----------------------
def choose_location(meta: Optional[dict],
                    le: Optional[dict],
                    pl: Optional[dict],
                    validate_m: float) -> Tuple[Optional[float], Optional[float], Optional[float], str, Optional[int], Optional[str]]:
    """
    Returns: (lat, lon, mph, source, seconds_delta, near_elem_id)
    Priority:
      1) metadata_csv_validated (vs LocationEvent if present, else vs PhoneLog)
      2) LocationEvent
      3) PhoneLog
      4) metadata_csv (raw, last resort)
      5) none
    """
    # Normalize candidates
    def valid(d):
        return d and d.get("lat") is not None and d.get("lon") is not None

    # 1) Metadata validated
    if meta and plausible_fix(meta.get("lat"), meta.get("lon"), meta.get("mph")):
        if valid(le):
            try:
                d = haversine_m(meta["lat"], meta["lon"], le["lat"], le["lon"])
            except Exception:
                d = float("inf")
            if d <= validate_m:
                return (meta["lat"], meta["lon"], meta.get("mph"), "metadata_csv_validated", None, None)
        elif valid(pl):
            try:
                d = haversine_m(meta["lat"], meta["lon"], pl["lat"], pl["lon"])
            except Exception:
                d = float("inf")
            if d <= validate_m:
                return (meta["lat"], meta["lon"], meta.get("mph"), "metadata_csv_validated", None, None)

    # 2) LocationEvent
    if valid(le) and plausible_fix(le.get("lat"), le.get("lon"), le.get("mph")):
        return (le["lat"], le["lon"], le.get("mph"), "LocationEvent",
                int(le.get("seconds") or 0), le.get("elem_id"))

    # 3) PhoneLog
    if valid(pl) and plausible_fix(pl.get("lat"), pl.get("lon"), pl.get("mph")):
        return (pl["lat"], pl["lon"], pl.get("mph"), "PhoneLog",
                int(pl.get("seconds") or 0), pl.get("elem_id"))

    # 4) Metadata raw
    if meta and plausible_fix(meta.get("lat"), meta.get("lon"), meta.get("mph")):
        return (meta["lat"], meta["lon"], meta.get("mph"), "metadata_csv", None, None)

    # 5) none
    return (None, None, None, "none", None, None)

# -----------------------
# Patching seconds for a (key,view)
# -----------------------
def patch_seconds_for_key(sess, *, key: str, view: str, secs: List[int], clip_path: str,
                          fps: float, win_mins: int, validate_m: float, dry_run: bool) -> int:
    folder = os.path.dirname(clip_path)
    base_key = clip_base_key(key)
    dt0 = parse_key_datetime(key) or datetime.datetime.fromtimestamp(0, tz=datetime.timezone.utc)

    # Load folder metadata
    meta_by_sec = read_base_metadata_csv(folder, base_key, fps)

    # Build timestamps list
    times_iso = [(dt0 + datetime.timedelta(seconds=int(s))).isoformat() for s in sorted(secs)]
    # Bulk fetch references
    pl_lookup = bulk_nearest_phonelog(sess, times_iso, win_mins)
    le_lookup = bulk_nearest_locevent(sess, times_iso, win_mins)

    patched = 0
    for s, t_iso in zip(sorted(secs), times_iso):
        meta = meta_by_sec.get(s)
        pl = pl_lookup.get(t_iso)
        le = le_lookup.get(t_iso)

        lat, lon, mph, source, seconds_delta, near_elem = choose_location(meta, le, pl, validate_m)

        if source == "none":
            continue

        emb_id = f"{key}|{view}|sec|{s}"
        vec, loc_dim, _l2, _dim, _k, _v = get_embed_detail(sess, emb_id)
        if loc_dim <= 0:
            loc_dim = 7  # default tail length

        # Compose new loc_vec + patch vec tail
        t_utc = dt0 + datetime.timedelta(seconds=int(s))
        loc_vec_np, _ = location_feature(lat, lon, mph, t_utc, include_time=True)
        loc_vec = loc_vec_np.astype(np.float32).tolist()

        if vec and len(vec) >= loc_dim:
            new_vec = list(vec)
            new_vec[-loc_dim:] = loc_vec
        else:
            new_vec = vec  # unexpected shape; keep as is but still set loc_* props

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

# -----------------------
# Recompute minute loc_vec from seconds
# -----------------------
def recompute_minute_for_key(sess, *, key: str, view: str, dry_run: bool):
    m = get_minute_embed(sess, key, view)
    if not m:
        logging.info(f"[minute] No minute node for {key}|{view}; skipping.")
        return
    loc_vecs = fetch_seconds_locvecs(sess, key, view)
    if not loc_vecs:
        logging.info(f"[minute] No seconds loc_vecs for {key}|{view}; skipping.")
        return
    M = np.array(loc_vecs, dtype=np.float32)
    mloc = ensure_unit_l2(np.nanmean(M, axis=0)).astype(np.float32)

    # Patch minute vec tail
    vec = m["vec"] or []
    loc_dim = int(m["loc_dim"] or 7)
    if vec and len(vec) >= loc_dim:
        new_vec = list(vec)
        new_vec[-loc_dim:] = mloc.tolist()
    else:
        new_vec = vec

    # We leave minute lat/lon/mph as None (not reliably aggregable)
    if dry_run:
        logging.info(f"[DRY] minute {key}|{view} updated (loc_dim={loc_dim})")
    else:
        update_minute(sess, mid=m["id"], vec=new_vec, loc_vec=mloc.tolist(),
                      lat=None, lon=None, mph=None)

# -----------------------
# CLI
# -----------------------
def main():
    ap = argparse.ArgumentParser(description="Patch DashcamEmbedding seconds missing location using metadata + LocationEvent (±1h) + PhoneLog")
    ap.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    ap.add_argument("--neo4j-user", default="neo4j")
    ap.add_argument("--neo4j-pass", default="password")
    ap.add_argument("--key-limit", type=int, default=1000, help="Max (key,view) groups to process")
    ap.add_argument("--key-prefix", default=None, help="Optional key prefix filter, e.g. 2025_0405")
    ap.add_argument("--win-mins", type=int, default=10, help="± minutes window for nearest event search")
    ap.add_argument("--validate-m", type=float, default=50.0, help="Max distance to validate metadata against LE/PL")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-minute", action="store_true", help="Skip minute recompute")
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

# ./.venv/bin/python3 patch_missing_locations.py \
#   --neo4j-uri bolt://localhost:7687 \
#   --neo4j-user neo4j \
#   --neo4j-pass livelongandprosper \
#   --key-limit 2000 \
#   --win-mins 10 \
#   --validate-m 50