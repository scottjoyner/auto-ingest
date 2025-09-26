#!/usr/bin/env python3
import os, re, ast, json, math, logging, argparse, datetime, subprocess, shlex
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np
import pandas as pd
from moviepy.editor import VideoFileClip
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# =========================
# Design constants & defaults
# =========================
DEFAULT_VEHICLE_CLASSES = {
    "car","truck","bus","motorbike","motorcycle","bicycle","van","pickup","suv","trailer"
}
DEFAULT_CLASS_ID_MAP = {1:"bicycle",2:"car",3:"motorbike",5:"bus",7:"truck"}

@dataclass
class GridSpec:
    gw: int
    gh: int

@dataclass
class EmbeddingSpec:
    grids: List[GridSpec]
    include_conf: bool = True
    l2_normalize: bool = True
    concat_views: bool = True
    per_second: bool = True
    per_minute: bool = True
    append_location: bool = True
    time_features: bool = True
    density_heatmap: bool = True  # default to density (original heatmap.py behavior)


# =========================
# Utility
# =========================
def clip_base_key(key: str) -> str:
    """Strip trailing _F/_R/_FR so we can match existing Clip/Frame keys."""
    return re.sub(r'_(F|R|FR)$', '', key)

def assign_seconds(df: pd.DataFrame, fps: float, dur: float, key: str) -> pd.DataFrame:
    """
    Adds a __sec__ column to df using the best available time signal:
    1) frame -> floor(frame/fps)
    2) t     -> floor(t)
    3) time  -> floor(time)
    4) timestamp -> floor(timestamp)
    Fallback: random seconds in [0, ceil(dur)), with a warning.
    """
    df = df.copy()
    s_count = max(1, int(math.ceil(dur)))

    # Normalize likely column names once (lowercase)
    cols = {c.lower(): c for c in df.columns}

    def col(name):  # get original-cased col if present
        return cols.get(name)

    if col("frame"):
        df[col("frame")] = pd.to_numeric(df[col("frame")], errors="coerce")
        df["__sec__"] = df[col("frame")].apply(lambda fr: second_from_frame(fr, fps))
        logging.info(f"[{key}] seconds derived from 'frame' @ fps={fps:.3f}")
    elif col("t"):
        df[col("t")] = pd.to_numeric(df[col("t")], errors="coerce")
        df["__sec__"] = df[col("t")].floordiv(1).astype("Int64")
        logging.info(f"[{key}] seconds derived from 't'")
    elif col("time"):
        df[col("time")] = pd.to_numeric(df[col("time")], errors="coerce")
        df["__sec__"] = df[col("time")].floordiv(1).astype("Int64")
        logging.info(f"[{key}] seconds derived from 'time'")
    elif col("timestamp"):
        df[col("timestamp")] = pd.to_numeric(df[col("timestamp")], errors="coerce")
        df["__sec__"] = df[col("timestamp")].floordiv(1).astype("Int64")
        logging.info(f"[{key}] seconds derived from 'timestamp'")
    else:
        # Fallback: randomized buckets (last resort)
        df["__sec__"] = np.random.randint(0, s_count, size=len(df))
        logging.warning(f"[{key}] no time columns found (frame/t/time/timestamp); "
                        f"assigned seconds randomly across {s_count} buckets")

    # Clean up impossible/NaN seconds
    df["__sec__"] = pd.to_numeric(df["__sec__"], errors="coerce").astype("Int64")
    df = df[(df["__sec__"].notna()) & (df["__sec__"] >= 0) & (df["__sec__"] < s_count)].copy()
    return df


def haversine_m(lat1, lon1, lat2, lon2):
    # meters
    R = 6371000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = phi2 - phi1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi/2.0)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dl/2.0)**2
    return float(2*R*np.arcsin(np.sqrt(a)))

def plausible_fix(prev_fix, lat, lon, mph, max_jump_m=250.0, max_speed=110.0):
    # reject obviously bad points
    if lat is None or lon is None:
        return False
    if mph is not None and (mph < 0 or mph > max_speed):
        return False
    if prev_fix and all(v is not None for v in prev_fix[:2]):
        d = haversine_m(prev_fix[0], prev_fix[1], lat, lon)
        if d > max_jump_m:  # >250 m/sec jump = sketchy
            return False
    return True

def safe_literal_eval(s: str):
    try:
        return ast.literal_eval(s)
    except Exception:
        return None

def ensure_unit_l2(vec: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec)
    if n == 0.0 or not np.isfinite(n): return vec
    return vec / n

def flatten_grid(grid: np.ndarray) -> np.ndarray:
    return grid.astype(np.float32).ravel()

def bbox_overlap_area(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2) -> float:
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1: return 0.0
    return float(ix2 - ix1) * float(iy2 - iy1)

def add_bbox_to_grid(
    grid: np.ndarray,
    bbox,
    img_w: int,
    img_h: int,
    weight: float,
    gs: GridSpec,
    density: bool = False,
):
    """
    Accumulate detection into grid cells.
    - density=False: add raw overlapped pixel area * weight  (area mass)
    - density=True : add (overlap / bbox_area) * weight      (normalized per box)
    """
    gw, gh = gs.gw, gs.gh
    x1, y1, x2, y2 = bbox

    # clamp bbox to image
    x1 = max(0, min(img_w, x1)); x2 = max(0, min(img_w, x2))
    y1 = max(0, min(img_h, y1)); y2 = max(0, min(img_h, y2))
    if x2 <= x1 or y2 <= y1:
        return

    bbox_area = float(x2 - x1) * float(y2 - y1)
    if bbox_area <= 0.0:
        return

    cell_w = img_w / gw
    cell_h = img_h / gh

    cx1 = int(np.floor(x1 / cell_w)); cx2 = int(np.floor((x2 - 1e-6) / cell_w))
    cy1 = int(np.floor(y1 / cell_h)); cy2 = int(np.floor((y2 - 1e-6) / cell_h))
    cx1 = max(0, min(gw - 1, cx1)); cx2 = max(0, min(gw - 1, cx2))
    cy1 = max(0, min(gh - 1, cy1)); cy2 = max(0, min(gh - 1, cy2))

    for cy in range(cy1, cy2 + 1):
        cell_y1 = cy * cell_h
        cell_y2 = (cy + 1) * cell_h
        for cx in range(cx1, cx2 + 1):
            cell_x1 = cx * cell_w
            cell_x2 = (cx + 1) * cell_w
            overlap = bbox_overlap_area(x1, y1, x2, y2, cell_x1, cell_y1, cell_x2, cell_y2)
            if overlap <= 0.0:
                continue

            if density:
                contrib = float(weight) * (overlap / bbox_area)   # normalized per box
            else:
                contrib = float(weight) * float(overlap)          # area mass in pixels

            grid[cy, cx] += contrib


# --- robust CSV parsing helpers ---
def _split_ignoring_brackets(line: str) -> List[str]:
    """Split a CSV line on commas that are NOT inside [...]"""
    parts, buf = [], []
    depth = 0
    for ch in line.rstrip("\r\n"):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf).strip())
    return parts

def _maybe_list4(x):
    """ast-literal-eval lists; otherwise try to pull 4 numbers out of a string."""
    if isinstance(x, (list, tuple, np.ndarray)) and len(x) == 4:
        return [float(v) for v in x]
    if pd.isna(x):
        return None
    s = str(x)
    if "[" in s and "]" in s:
        try:
            v = ast.literal_eval(s)
            if isinstance(v, (list, tuple, np.ndarray)) and len(v) == 4:
                return [float(xx) for xx in v]
        except Exception:
            pass
    # fallback like "[1040 500 1150 620]" or "813,171,1527,1100"
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if len(nums) == 4:
        return [float(n) for n in nums]
    return None

def _percent_to_float01(val):
    """'79.48%' -> 0.7948 ; '0.85' -> 0.85 ; bad -> np.nan"""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    s = str(val).strip()
    try:
        if s.endswith("%"):
            return float(s[:-1].strip()) / 100.0
        return float(s)
    except Exception:
        return np.nan

def parse_yolo_csv(csv_path: str) -> pd.DataFrame:
    """
    Parse YOLO CSV lines with bracketed vectors. Unifies:
      - name/label in one column: df['name']
      - confidence in [0..1]: df['confidence']
      - bboxes: df['xyxy'], df['xyxyn'], df['xywh'], df['xywhn'] parsed to 4-float lists
      - 'frame' coerced to integer
    Handles cases where 'confidence' may actually contain a label, or 'classification' is a percent.
    """
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        header = [h.strip() for h in _split_ignoring_brackets(f.readline())]
        for line in f:
            if not line.strip():
                continue
            parts = _split_ignoring_brackets(line)
            if len(parts) < len(header):
                parts += [""] * (len(header) - len(parts))
            elif len(parts) > len(header):
                parts = parts[:len(header)-1] + [",".join(parts[len(header)-1:])]
            rows.append(dict(zip(header, parts)))

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.columns = [c.strip().lower() for c in df.columns]

    # Coerce numeric-ish columns safely
    for c in ["frame", "t", "time", "timestamp", "cls", "class"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Name / label resolution
    name_col = None
    if "name" in df.columns:
        name_col = "name"
    elif "label" in df.columns:
        name_col = "label"
    elif "confidence" in df.columns:
        # if 'confidence' is mostly numeric/percent, it's not a label
        is_numericish = df["confidence"].map(lambda x: pd.notna(pd.to_numeric(str(x).replace("%",""), errors="coerce"))).all()
        if not is_numericish:
            name_col = "confidence"
    elif "classification" in df.columns:
        looks_like_num = df["classification"].map(
            lambda x: str(x).strip().endswith("%") or pd.notna(pd.to_numeric(str(x), errors="coerce"))
        ).all()
        if not looks_like_num:
            name_col = "classification"

    if name_col:
        df["name"] = df[name_col].astype(str).str.strip().str.lower()
    else:
        df["name"] = ""

    # Confidence numeric
    conf = np.full(len(df), np.nan, dtype=float)
    if "confidence" in df.columns:
        conf = df["confidence"].map(_percent_to_float01).to_numpy(dtype=float)
    if np.isnan(conf).all() and "classification" in df.columns:
        conf2 = df["classification"].map(_percent_to_float01).to_numpy(dtype=float)
        mask = np.isnan(conf) & ~np.isnan(conf2)
        conf[mask] = conf2[mask]
    conf = np.where(np.isnan(conf), 1.0, conf).clip(0,1)
    df["confidence"] = conf

    # Parse bbox-like columns
    for c in [x for x in ["xyxy", "xyxyn", "xywh", "xywhn", "bbox"] if x in df.columns]:
        df[c] = df[c].apply(_maybe_list4)

    # scalar bbox parts
    for c in ["x1","y1","x2","y2","xcenter","xc","ycenter","yc","width","w","height","h"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "cls" in df.columns and "class" not in df.columns:
        df["class"] = df["cls"]

    return df


def row_to_dict(row, columns=None):
    if isinstance(row, dict):
        return row
    if hasattr(row, "_asdict"):  # namedtuple
        return dict(row._asdict())
    if isinstance(row, pd.Series):
        return row.to_dict()
    if columns is not None:
        try:
            return {c: row[i] for i, c in enumerate(columns)}
        except Exception:
            return {}
    return {}

def to_xyxy_abs(row: dict, img_w: int, img_h: int) -> Optional[Tuple[int,int,int,int]]:
    get = row.get

    def as_vec4(val):
        v = val
        if isinstance(v, (list, tuple, np.ndarray)) and len(v) == 4:
            return [float(x) for x in v]
        return None

    # Prefer direct vector fields
    for key in ("xyxy", "xyxyn", "xywh", "xywhn", "bbox"):
        if key in row and row[key] is not None:
            v = as_vec4(row[key])
            if v is None:
                continue
            if key in ("xyxy", "bbox") and (v[2] > v[0]) and (v[3] > v[1]):
                x1, y1, x2, y2 = v
                return int(x1), int(y1), int(x2), int(y2)
            if key in ("xyxyn",):
                x1, y1, x2, y2 = v[0]*img_w, v[1]*img_h, v[2]*img_w, v[3]*img_h
                return int(x1), int(y1), int(x2), int(y2)
            # xywh / xywhn / bbox (as center-based) -> convert
            xc, yc, w, h = v
            if key in ("xywhn",) or max(xc, yc, w, h) <= 1.0001:
                xc *= img_w; yc *= img_h; w *= img_w; h *= img_h
            x1, y1, x2, y2 = xc - w/2.0, yc - h/2.0, xc + w/2.0, yc + h/2.0
            return int(x1), int(y1), int(x2), int(y2)

    # Scalar forms (x1,y1,x2,y2)
    x1, y1, x2, y2 = get("x1"), get("y1"), get("x2"), get("y2")
    if all(v is not None for v in (x1, y1, x2, y2)):
        return int(x1), int(y1), int(x2), int(y2)

    # Center-size scalars
    xc = get("xcenter") if get("xcenter") is not None else get("xc")
    yc = get("ycenter") if get("ycenter") is not None else get("yc")
    w  = get("width")   if get("width")   is not None else get("w")
    h  = get("height")  if get("height")  is not None else get("h")
    if all(v is not None for v in (xc, yc, w, h)):
        xc, yc, w, h = map(float, (xc, yc, w, h))
        if max(xc, yc, w, h) <= 1.0001:  # normalized
            xc *= img_w; yc *= img_h; w *= img_w; h *= img_h
        return int(xc - w/2.0), int(yc - h/2.0), int(xc + w/2.0), int(yc + h/2.0)

    return None


def keep_detection(row, keep_names: set, id_map: Dict[int,str]) -> bool:
    name = (row.get("name") or "").lower()
    if name and name in keep_names:
        return True
    # accept raw names like 'motorcycle' even if id_map says 'motorbike'
    if name and name.replace("motorcycle","motorbike") in keep_names:
        return True
    cid = row.get("class")
    if pd.notna(cid):
        try:
            mapped = id_map.get(int(cid))
            if mapped in keep_names:
                return True
        except Exception:
            pass
    return False

def second_from_frame(frame: Optional[float], fps: float) -> Optional[int]:
    if frame is None or not np.isfinite(frame): return None
    try: return int(frame // fps)
    except Exception: return None

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

# =========================
# Location features (NEW)
# =========================
def cyc(v: float, period: float) -> Tuple[float,float]:
    # cyclic encoding sin/cos
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

def find_best_locevent_then_phonelog(sess, t_utc: datetime.datetime, win_mins: int,
                                     prev_fix: Optional[Tuple[float,float,float]]) -> Tuple[Optional[dict], Optional[dict]]:
    """
    Returns (primary, fallback)
    primary = LocationEvent candidate dict or None
    fallback = PhoneLog candidate dict or None
    Each dict: {"elem_id": str, "seconds": int, "lat": float|None, "lon": float|None, "mph": float|None}
    Tries exact time first, then ±1h for LocationEvent. PhoneLog only at exact time window.
    """
    def q_locevent(t0):
        rec = sess.run(NQ_FIND_NEAREST_LOCEVENT, t_utc=t0.isoformat(), win_mins=int(win_mins)).single()
        return dict(rec) if rec else None

    def q_phonelog(t0):
        rec = sess.run(NQ_FIND_NEAREST_PHONELOG, t_utc=t0.isoformat(), win_mins=int(win_mins)).single()
        return dict(rec) if rec else None

    # LocationEvent at t, then ±1 hour variants; pick the closest dt
    candidates = []
    for offs in [0, +3600, -3600]:
        t_try = t_utc + datetime.timedelta(seconds=offs)
        r = q_locevent(t_try)
        if r: 
            r["_offset"] = offs
            candidates.append(r)
    primary = None
    if candidates:
        primary = sorted(candidates, key=lambda d: d["seconds"])[0]

    # Fallback PhoneLog (no offset tricks; PL is usually correctly timed)
    fallback = q_phonelog(t_utc)
    return primary, fallback

def choose_location(primary: Optional[dict], fallback: Optional[dict],
                    meta: Optional[dict], prev_fix: Optional[Tuple[float,float,float]]) -> Tuple[Optional[dict], str]:
    """
    Applies your policy:
    1) Use LocationEvent if plausible
    2) else PhoneLog if plausible
    3) else metadata if plausible
    Returns (chosen_dict, source_str)
    chosen_dict has lat/lon/mph and optionally elem_id/seconds for linking.
    """
    def ext(d):
        if not d: return (None,None,None)
        return d.get("lat"), d.get("lon"), d.get("mph")

    # 1) LocationEvent
    lat, lon, mph = ext(primary)
    if plausible_fix(prev_fix, lat, lon, mph):
        return primary, "LocationEvent"

    # 2) PhoneLog
    lat, lon, mph = ext(fallback)
    if plausible_fix(prev_fix, lat, lon, mph):
        return fallback, "PhoneLog"

    # 3) metadata.csv but only if close to phonelog (if we had one)
    if meta:
        mlat, mlon, mmph = meta.get("lat"), meta.get("lon"), meta.get("mph")
        if fallback and all(v is not None for v in (fallback.get("lat"), fallback.get("lon"))):
            d = haversine_m(mlat, mlon, fallback["lat"], fallback["lon"])
            if d <= 50.0 and plausible_fix(prev_fix, mlat, mlon, mmph):
                return {"lat":mlat, "lon":mlon, "mph":mmph}, "metadata_csv_validated"
        if plausible_fix(prev_fix, mlat, mlon, mmph):
            return {"lat":mlat, "lon":mlon, "mph":mmph}, "metadata_csv"

    return {"lat":None, "lon":None, "mph":None}, "none"

def resolve_location_for_second(
    sess,
    key: str,
    fps: float,
    second: int,
    dt0_utc: datetime.datetime,
    win_mins: int,
    meta_scalars: Optional[Dict[str, Optional[float]]],
    prev_fix: Optional[Tuple[float, float, Optional[float]]],
    max_frame_delta: int = 15,
) -> Tuple[Dict[str, Optional[float]], str, Optional[str], Optional[float]]:
    """
    Returns (chosen_scalars, source, elem_id_for_NEAR, seconds_delta_if_known)
    """
    base_key = clip_base_key(key)
    t_utc = dt0_utc + datetime.timedelta(seconds=int(second))
    frame_idx = int(round(second * fps))

    # 1) Frame match by key / relations
    f = neo4j_find_nearest_frame(sess, base_key, frame_idx, max_delta=max_frame_delta, dc_key=key)
    if f and plausible_fix(prev_fix, f.get("lat"), f.get("lon"), f.get("mph")):
        dframes = abs(int(f.get("frame", frame_idx)) - frame_idx)
        sdelta = float(dframes) / float(max(fps, 1.0))
        return ({"lat": f.get("lat"), "lon": f.get("lon"), "mph": f.get("mph")},
                "Frame", f.get("elem_id"), sdelta)

    # 2) LocationEvent / 3) PhoneLog (reuse your query helpers)
    primary, fallback = find_best_locevent_then_phonelog(sess, t_utc, win_mins, prev_fix)
    if primary and plausible_fix(prev_fix, primary.get("lat"), primary.get("lon"), primary.get("mph")):
        return ({"lat": primary.get("lat"), "lon": primary.get("lon"), "mph": primary.get("mph")},
                "LocationEvent", primary.get("elem_id"), float(primary.get("seconds", 0) or 0))
    if fallback and plausible_fix(prev_fix, fallback.get("lat"), fallback.get("lon"), fallback.get("mph")):
        return ({"lat": fallback.get("lat"), "lon": fallback.get("lon"), "mph": fallback.get("mph")},
                "PhoneLog", fallback.get("elem_id"), float(fallback.get("seconds", 0) or 0))

    # 4) metadata.csv (validate vs PL if present)
    if meta_scalars:
        mlat, mlon, mmph = meta_scalars.get("lat"), meta_scalars.get("lon"), meta_scalars.get("mph")
        if fallback and all(v is not None for v in (fallback.get("lat"), fallback.get("lon"))):
            try:
                d = haversine_m(mlat, mlon, fallback["lat"], fallback["lon"])
            except Exception:
                d = float("inf")
            if d <= 50.0 and plausible_fix(prev_fix, mlat, mlon, mmph):
                return ({"lat": mlat, "lon": mlon, "mph": mmph}, "metadata_csv_validated", None, None)
        if plausible_fix(prev_fix, mlat, mlon, mmph):
            return ({"lat": mlat, "lon": mlon, "mph": mmph}, "metadata_csv", None, None)

    return ({"lat": None, "lon": None, "mph": None}, "none", None, None)

# =========================
# Neo4j I/O (expanded)
# =========================
NQ_FIND_NEAREST_FRAME_BY_KEY = """
MATCH (f:Frame {key:$base_key})
WITH f, toInteger(f.frame) AS ff
WHERE f.lat IS NOT NULL AND coalesce(f.lon, f.long, f.longitude) IS NOT NULL
  AND ff >= $frame - $max_delta AND ff <= $frame + $max_delta
RETURN elementId(f) AS elem_id,
       ff AS frame,
       toFloat(f.lat) AS lat,
       toFloat(coalesce(f.lon, f.long, f.longitude)) AS lon,
       toFloat(f.mph) AS mph,
       abs(ff - $frame) AS dframe
ORDER BY dframe ASC
LIMIT 1
"""

NQ_FIND_NEAREST_FRAME_BY_CLIPID = """
MATCH (c:Clip {id:$base_key})<-[:BELONGS_TO]-(f:Frame)
WITH f, toInteger(f.frame) AS ff
WHERE f.lat IS NOT NULL AND coalesce(f.lon, f.long, f.longitude) IS NOT NULL
  AND ff >= $frame - $max_delta AND ff <= $frame + $max_delta
RETURN elementId(f) AS elem_id,
       ff AS frame,
       toFloat(f.lat) AS lat,
       toFloat(coalesce(f.lon, f.long, f.longitude)) AS lon,
       toFloat(f.mph) AS mph,
       abs(ff - $frame) AS dframe
ORDER BY dframe ASC
LIMIT 1
"""

NQ_FIND_NEAREST_FRAME_BY_DASHCAM = """
MATCH (dc:DashcamClip {key:$dc_key})<-[:BELONGS_TO]-(f:Frame)
WITH f, toInteger(f.frame) AS ff
WHERE f.lat IS NOT NULL AND coalesce(f.lon, f.long, f.longitude) IS NOT NULL
  AND ff >= $frame - $max_delta AND ff <= $frame + $max_delta
RETURN elementId(f) AS elem_id,
       ff AS frame,
       toFloat(f.lat) AS lat,
       toFloat(coalesce(f.lon, f.long, f.longitude)) AS lon,
       toFloat(f.mph) AS mph,
       abs(ff - $frame) AS dframe
ORDER BY dframe ASC
LIMIT 1
"""

NQ_FIND_NEAREST_LOCEVENT = """
WITH datetime($t_utc) AS et0
MATCH (l:LocationEvent)-[:BELONGS_TO]->(:Trip)
WHERE l.eventTime >= et0 - duration({minutes:$win_mins})
  AND l.eventTime <= et0 + duration({minutes:$win_mins})
WITH l, abs(duration.between(l.eventTime, et0).seconds) AS dt
ORDER BY dt ASC
LIMIT 1
RETURN elementId(l) AS elem_id, dt AS seconds, l.latitude AS lat, l.longitude AS lon,
       toFloat(coalesce(l.speed, replace(l.speed,'mph',''))) AS mph
"""

NQ_FIND_NEAREST_PHONELOG = """
WITH datetime($t_utc) AS et0
MATCH (p:PhoneLog)
WHERE p.timestamp >= et0 - duration({minutes:$win_mins})
  AND p.timestamp <= et0 + duration({minutes:$win_mins})
WITH p, abs(duration.between(p.timestamp, et0).seconds) AS dt
ORDER BY dt ASC
LIMIT 1
RETURN elementId(p) AS elem_id, dt AS seconds, p.lat AS lat, p.lon AS lon, toFloat(p.mph) AS mph
"""

# --- NEW: generic NEAR edge creation by elementId target ---
NQ_ATTACH_NEAR_BY_ELEMID = """
MATCH (e:DashcamEmbedding {id:$eid})
MATCH (n) WHERE elementId(n) = $elem_id
MERGE (e)-[:NEAR {seconds:$seconds, source:$source}]->(n)
"""

NQ_CREATE_CONSTRAINTS = [
    """
    CREATE CONSTRAINT dashcam_clip_key IF NOT EXISTS
    FOR (c:DashcamClip) REQUIRE c.key IS UNIQUE
    """,
    """
    CREATE CONSTRAINT dashcam_embedding_id IF NOT EXISTS
    FOR (e:DashcamEmbedding) REQUIRE e.id IS UNIQUE
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
    """
]

NQ_UPSERT_CLIP = """
MERGE (c:DashcamClip {key: $key})
ON CREATE SET c.path=$path, c.view=$view, c.width=$width, c.height=$height,
              c.fps=$fps, c.duration_s=$duration_s, c.created_at=timestamp()
ON MATCH  SET c.path=$path, c.width=$width, c.height=$height,
              c.fps=$fps, c.duration_s=$duration_s, c.updated_at=timestamp()
RETURN c
"""

NQ_UPSERT_EMBED = """
MERGE (e:DashcamEmbedding {id: $id})
ON CREATE SET e.level=$level, e.key=$key, e.view=$view,
              e.t0=$t0, e.t1=$t1, e.dim=$dim,
              e.grids_str=$grids_str,
              e.l2=$l2, e.model=$model, e.concat_views=$concat_views,
              e.vec=$vec, e.loc_vec=$loc_vec, e.loc_dim=$loc_dim,
              e.lat=$lat, e.lon=$lon, e.mph=$mph, e.loc_source=$loc_source,
              e.created_at=timestamp()
ON MATCH  SET e.vec=$vec, e.loc_vec=$loc_vec, e.lat=$lat, e.lon=$lon, e.mph=$mph,
              e.loc_dim=$loc_dim, e.updated_at=timestamp(), e.loc_source=$loc_source,
              e.dim=$dim, e.grids_str=$grids_str, e.concat_views=$concat_views
WITH e
MATCH (c:DashcamClip {key:$key})
MERGE (c)-[:HAS_EMBEDDING {level:$level}]->(e)
RETURN e
"""

NQ_LINK_NEXT = "MATCH (a:DashcamEmbedding {id:$a_id}),(b:DashcamEmbedding {id:$b_id}) MERGE (a)-[:NEXT]->(b)"

NQ_ATTACH_NEAREST_PHONELOG = """
MATCH (e:DashcamEmbedding {id:$eid}), (c:DashcamClip {key:$key})
WITH e, c
WITH e, datetime({epochSeconds: e.t0}) AS et0
MATCH (p:PhoneLog)
WHERE p.timestamp >= (et0 - duration({minutes:$win_mins}))
  AND p.timestamp <= (et0 + duration({minutes:$win_mins}))
WITH e, p, abs(duration.between(p.timestamp, et0).seconds) AS dt
ORDER BY dt ASC
LIMIT 1
MERGE (e)-[:NEAR {seconds: dt, source:'PhoneLog'}]->(p)
RETURN dt, p
"""

NQ_ATTACH_NEAREST_LOCEVENT = """
MATCH (e:DashcamEmbedding {id:$eid})
WITH e, datetime({epochSeconds: e.t0}) AS et0
MATCH (l:LocationEvent)-[:BELONGS_TO]->(:Trip)
WHERE l.eventTime >= (et0 - duration({minutes:$win_mins}))
  AND l.eventTime <= (et0 + duration({minutes:$win_mins}))
WITH e, l, abs(duration.between(l.eventTime, et0).seconds) AS dt
ORDER BY dt ASC
LIMIT 1
MERGE (e)-[:NEAR {seconds: dt, source:'LocationEvent'}]->(l)
RETURN dt, l
"""

NQ_ENRICH_FRAME = """
MERGE (f:Frame {key:$base_key, frame:$frame})
ON CREATE SET f.created_at = timestamp()
SET f.lat = coalesce(f.lat, $lat),
    f.lon = coalesce(f.lon, $lon),
    f.long = coalesce(f.long, $lon),   // keep both spellings filled
    f.mph = coalesce(f.mph, $mph)
WITH f
MATCH (c:DashcamClip {key:$dc_key})
MERGE (f)-[:BELONGS_TO]->(c)
RETURN f
"""

NQ_GET_EXISTING_SECONDS = """
MATCH (e:DashcamEmbedding {key:$key, view:$view, level:'second'})
RETURN collect(e.t0) AS secs
"""

NQ_MINUTE_EXISTS = """
MATCH (e:DashcamEmbedding {key:$key, view:$view, level:'minute'})
RETURN count(e) AS c
"""

NQ_GET_CLIP_META = """
MATCH (c:DashcamClip {key:$key})
RETURN c.duration_s AS duration_s, c.fps AS fps, c.width AS width, c.height AS height, c.path AS path
"""

NQ_REBUILD_NEXT = """
MATCH (e:DashcamEmbedding {key:$key, view:$view, level:'second'})
WITH e ORDER BY e.t0 ASC
WITH collect(e) AS es
UNWIND range(0, size(es)-2) AS i
WITH es[i] AS a, es[i+1] AS b
MERGE (a)-[:NEXT]->(b)
"""


def neo4j_session(uri: str, user: str, pwd: str):
    return GraphDatabase.driver(uri, auth=(user, pwd))

def neo4j_create_constraints(sess):
    for stmt in NQ_CREATE_CONSTRAINTS:
        try:
            sess.run(stmt)
        except Exception as e:
            logging.warning(f"Constraint/index creation failed or already exists: {e}")


def neo4j_upsert_clip(sess, key: str, path: str, view: str, width: int, height: int, fps: float, dur: float):
    return sess.run(NQ_UPSERT_CLIP, key=key, path=path, view=view, width=width, height=height, fps=fps, duration_s=int(dur)).single()

def neo4j_upsert_embed(sess, *, emb_id: str, level: str, key: str, view: str,
                       t0: int, t1: int, vec: np.ndarray, loc_vec: np.ndarray,
                       loc_scalars: Dict[str,Optional[float]], loc_source: str,
                       spec: EmbeddingSpec, grids: List[GridSpec], model: str="yolov8n"):
    v = vec.astype(np.float32)

    # Append location (do not L2 the scene when density_heatmap=True)
    if spec.append_location and loc_vec is not None:
        v = np.concatenate([v, loc_vec.astype(np.float32)], axis=0)

    # Only L2 if we're NOT in density mode
    if spec.l2_normalize and not spec.density_heatmap:
        v = ensure_unit_l2(v)

    params = {
        "id": emb_id, "level": level, "key": key, "view": view,
        "t0": int(t0), "t1": int(t1),
        "dim": int(v.shape[0]), "vec": v.tolist(),
        "loc_vec": (loc_vec.tolist() if loc_vec is not None else None),
        "loc_dim": (int(loc_vec.shape[0]) if loc_vec is not None else 0),
        "l2": bool(spec.l2_normalize and not spec.density_heatmap),
        "grids_str": [f"{g.gw}x{g.gh}" for g in grids],
        "model": model, "concat_views": bool(spec.concat_views),
        "lat": loc_scalars.get("lat"), "lon": loc_scalars.get("lon"), "mph": loc_scalars.get("mph"),
        "loc_source": loc_source
    }
    return sess.run(NQ_UPSERT_EMBED, **params).single()



def neo4j_link_next(sess, a_id: str, b_id: str):
    sess.run(NQ_LINK_NEXT, a_id=a_id, b_id=b_id)

def neo4j_attach_nearest_phonelog(sess, eid: str, key: str, win_mins: int) -> Optional[int]:
    rec = sess.run(NQ_ATTACH_NEAREST_PHONELOG, eid=eid, key=key, win_mins=int(win_mins)).single()
    return rec["dt"] if rec else None

def neo4j_attach_nearest_locevent(sess, eid: str, win_mins: int) -> Optional[int]:
    rec = sess.run(NQ_ATTACH_NEAREST_LOCEVENT, eid=eid, win_mins=int(win_mins)).single()
    return rec["dt"] if rec else None


def neo4j_enrich_frame(sess, key: str, frame_idx: int,
                       lat: Optional[float], lon: Optional[float], mph: Optional[float]):
    base_key = clip_base_key(key)
    sess.run(
        NQ_ENRICH_FRAME,
        base_key=base_key,
        dc_key=key,
        frame=int(frame_idx),
        lat=lat, lon=lon, mph=mph
    )

def neo4j_get_existing_seconds(sess, key: str, view: str) -> set:
    rec = sess.run(NQ_GET_EXISTING_SECONDS, key=key, view=view).single()
    return set(rec["secs"] or []) if rec else set()

def neo4j_minute_exists(sess, key: str, view: str) -> bool:
    rec = sess.run(NQ_MINUTE_EXISTS, key=key, view=view).single()
    return (rec and rec["c"] and int(rec["c"]) > 0)

def neo4j_get_clip_meta(sess, key: str) -> Tuple[Optional[int], Optional[float], Optional[int], Optional[int], Optional[str]]:
    rec = sess.run(NQ_GET_CLIP_META, key=key).single()
    if not rec:
        return None, None, None, None, None
    return (rec["duration_s"], rec["fps"], rec["width"], rec["height"], rec["path"])

def neo4j_rebuild_next(sess, key: str, view: str):
    sess.run(NQ_REBUILD_NEXT, key=key, view=view)

def neo4j_find_nearest_frame(sess, base_key: str, frame_idx: int, max_delta: int = 15,
                             dc_key: Optional[str] = None) -> Optional[dict]:
    """
    Try progressively larger windows and different attachment styles,
    returning only frames that actually have lat/lon.
    """
    windows = [max_delta, int(3*max(1, max_delta)), int(30*max(1, max_delta))]  # ~0.5s, ~1.5s, ~15s if fps≈30

    for win in windows:
        # 1) Frame.key
        rec = sess.run(
            NQ_FIND_NEAREST_FRAME_BY_KEY,
            base_key=base_key, frame=int(frame_idx), max_delta=int(win)
        ).single()
        if rec:
            d = dict(rec)
            logging.info(f"[GPS] Frame-by-key hit base_key={base_key} frame≈{frame_idx} d={d.get('dframe')} lat={d.get('lat')} lon={d.get('lon')}")
            return d

        # 2) DashcamClip relation (full key with _F/_R)
        if dc_key:
            rec = sess.run(
                NQ_FIND_NEAREST_FRAME_BY_DASHCAM,
                dc_key=dc_key, frame=int(frame_idx), max_delta=int(win)
            ).single()
            if rec:
                d = dict(rec)
                logging.info(f"[GPS] Frame-by-DashcamClip hit dc_key={dc_key} frame≈{frame_idx} d={d.get('dframe')} lat={d.get('lat')} lon={d.get('lon')}")
                return d

        # 3) Clip.id (base key)
        rec = sess.run(
            NQ_FIND_NEAREST_FRAME_BY_CLIPID,
            base_key=base_key, frame=int(frame_idx), max_delta=int(win)
        ).single()
        if rec:
            d = dict(rec)
            logging.info(f"[GPS] Frame-by-ClipID hit base_key={base_key} frame≈{frame_idx} d={d.get('dframe')} lat={d.get('lat')} lon={d.get('lon')}")
            return d

    logging.info(f"[GPS] No Frame WITH GPS found for base_key={base_key} frame≈{frame_idx} (windows={windows})")
    return None


# =========================
# Robust video probing & repair
# =========================
def _parse_rate(rate_str: Optional[str]) -> Optional[float]:
    if not rate_str:
        return None
    try:
        if "/" in rate_str:
            a, b = rate_str.split("/", 1)
            a = float(a); b = float(b)
            return a / b if b else a
        return float(rate_str)
    except Exception:
        return None

def ffprobe_video_meta(path: str) -> Tuple[int, int, float, float]:
    """
    Returns (w, h, fps, dur). Uses ffprobe JSON and derives sensibly if pieces are missing.
    Raises on unrecoverable failure.
    """
    cmd = (
        'ffprobe -v error '
        '-select_streams v:0 '
        '-show_entries stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration '
        '-show_entries format=duration '
        '-of json '
        + shlex.quote(path)
    )
    p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffprobe failed ({p.returncode}): {p.stderr.strip()}")

    data = json.loads(p.stdout or "{}")
    streams = data.get("streams") or []
    fmt = data.get("format") or {}
    if not streams:
        raise RuntimeError("No video stream found")

    s0 = streams[0]
    w = int(s0.get("width") or 0)
    h = int(s0.get("height") or 0)
    fps = _parse_rate(s0.get("avg_frame_rate")) or _parse_rate(s0.get("r_frame_rate")) or 30.0

    # duration candidates
    dur_candidates = [s0.get("duration"), fmt.get("duration")]
    dur = None
    for d in dur_candidates:
        try:
            if d is not None:
                dur = float(d)
                break
        except Exception:
            pass

    if dur is None:
        nb_frames = s0.get("nb_frames")
        if nb_frames is not None:
            try:
                dur = float(nb_frames) / float(max(fps, 1e-6))
            except Exception:
                pass

    if not w or not h:
        raise RuntimeError("Width/height unavailable")
    if dur is None or not np.isfinite(dur) or dur <= 0:
        raise RuntimeError("Duration unavailable")

    return int(w), int(h), float(fps), float(dur)

def opencv_video_meta(path: str) -> Optional[Tuple[int,int,float,float]]:
    try:
        import cv2  # optional fallback
    except Exception:
        return None
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 30.0
    n  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if not w or not h:
        return None
    dur = (n / fps) if n and fps else 0.0
    if dur <= 0:
        return None
    return w, h, fps, dur

def infer_meta_from_csv(csv_path: str, default_fps: float = 30.0) -> Optional[Tuple[int,int,float,float]]:
    """
    As a last resort when the video is unreadable:
    - derive duration from max frame index found in YOLO CSV.
    Returns (0,0,fps,dur) since width/height are unknown; caller must decide to skip.
    """
    if not os.path.exists(csv_path):
        return None
    try:
        df = parse_yolo_csv(csv_path)
        if "frame" in df.columns and df["frame"].notna().any():
            max_fr = int(pd.to_numeric(df["frame"], errors="coerce").max() or 0)
            fps = float(default_fps)
            dur = float(max(1, int(math.ceil((max_fr + 1) / max(fps, 1e-6)))))
            return 0, 0, fps, dur
    except Exception:
        pass
    return None

def try_fix_missing_moov(src: str, dst: str) -> bool:
    """
    OPTIONAL: copy atoms to 'faststart' layout and synthesize PTS.
    Returns True on success. No re-encode.
    """
    cmd = f'ffmpeg -y -v error -fflags +genpts -i {shlex.quote(src)} -c copy -movflags +faststart {shlex.quote(dst)}'
    p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.returncode == 0

def get_video_meta(mp4_path: str, csv_path_hint: Optional[str] = None, *, allow_csv_fallback: bool = True) -> Tuple[int, int, float, float]:
    """
    Returns (img_w, img_h, fps, dur_seconds).
    Order: ffprobe → OpenCV → (optional) YOLO CSV inference → MoviePy (last resort).
    Raises RuntimeError if we truly cannot provide sensible meta.
    """
    # 1) ffprobe
    try:
        return ffprobe_video_meta(mp4_path)
    except Exception as e:
        logging.warning(f"[probe] ffprobe failed on {mp4_path}: {e}")

    # 2) OpenCV fallback
    ocv = opencv_video_meta(mp4_path)
    if ocv is not None:
        return ocv

    # 3) YOLO CSV heuristic duration (no width/height)
    if allow_csv_fallback and csv_path_hint:
        inferred = infer_meta_from_csv(csv_path_hint)
        if inferred is not None:
            w, h, fps, dur = inferred
            if w == 0 or h == 0:
                # Can't build grids without image size → let caller decide to skip.
                raise RuntimeError("Video unreadable; CSV inferred duration but no width/height")
            return inferred

    # 4) MoviePy (slowest; also fails on missing moov)
    try:
        with VideoFileClip(mp4_path) as clip:
            fps = float(clip.fps or 30.0)
            dur = float(clip.duration or 60.0)
            img_w, img_h = int(clip.w), int(clip.h)
        return img_w, img_h, fps, dur
    except Exception as e:
        raise RuntimeError(f"MoviePy could not open {mp4_path}: {e}")


# =========================
# Embedding builders (scene)
# =========================
def build_grid_embedding_for_interval(
    detections: Iterable[pd.Series],
    img_w: int,
    img_h: int,
    grids: List[GridSpec],
    include_conf: bool,
    density: bool = False,
) -> np.ndarray:
    parts = []
    for gs in grids:
        grid = np.zeros((gs.gh, gs.gw), dtype=np.float32)
        for row in detections:
            if hasattr(row, "_asdict"):
                row_like = row._asdict()
            elif isinstance(row, pd.Series):
                row_like = row.to_dict()
            else:
                row_like = row

            bbox = to_xyxy_abs(row_like, img_w, img_h)
            if bbox is None:
                continue

            conf = float(row_like.get("confidence", 1.0)) if include_conf else 1.0
            add_bbox_to_grid(grid, bbox, img_w, img_h, conf, gs, density=density)

        parts.append(grid.ravel().astype(np.float32))
    return np.concatenate(parts, axis=0)


def aggregate_seconds_to_minute(second_vecs: List[np.ndarray], *, mode: str = "sum") -> np.ndarray:
    """
    mode='sum' → accumulate activity over time (heatmap density).
    mode='mean' → legacy smoothing.
    """
    if not second_vecs:
        return None
    M = np.stack(second_vecs, axis=0)
    return np.nansum(M, axis=0) if mode == "sum" else np.nanmean(M, axis=0)


def build_vectors_for_key(
    file_dir: str,
    key: str,
    spec: EmbeddingSpec,
    keep_names: set,
    id_map: Dict[int,str],
    grids: List[GridSpec],
    include_heatmap_png: bool,
    minute_heatmap_out: Optional[str],
    keep_all: bool = False,
    *,
    seconds_whitelist: Optional[set] = None,
    compute_minute: bool = True,
    pre_meta: Optional[Tuple[int,int,float,float]] = None,  # (img_w, img_h, fps, dur)
    repair_missing_moov: bool = False,
) -> Dict[str, Dict]:
    """
    If seconds_whitelist is provided and compute_minute=False:
      - only compute those specific seconds and return maps keyed by second
      - no minute vector is produced
    If compute_minute=True:
      - we must compute vectors across all seconds (minute = sum/mean of seconds)
    """
    result = {}

    mp4_path = os.path.join(file_dir, f"{key}.MP4")
    if not os.path.exists(mp4_path) or os.path.getsize(mp4_path) == 0:
        logging.warning(f"[scan] Missing or empty MP4 for {key}, skipping")
        return result

    view = "F" if key.endswith("_F") or key.endswith("_FR") else ("R" if key.endswith("_R") else "?")

    csv_path = os.path.join(file_dir, f"{key}_YOLOv8n.csv")
    if not os.path.exists(csv_path):
        return result

    # Prefer preknown meta (from Neo4j) to avoid opening the file
    if pre_meta is not None and all(v is not None for v in pre_meta):
        img_w, img_h, fps, dur = pre_meta
    else:
        try:
            img_w, img_h, fps, dur = get_video_meta(mp4_path, csv_path_hint=csv_path, allow_csv_fallback=True)
        except Exception as e:
            # Try repair once if configured
            if repair_missing_moov:
                fixed = mp4_path.replace(".MP4", "_fixed.MP4")
                ok = try_fix_missing_moov(mp4_path, fixed)
                if ok and os.path.exists(fixed):
                    try:
                        img_w, img_h, fps, dur = get_video_meta(fixed, csv_path_hint=csv_path, allow_csv_fallback=True)
                        mp4_path = fixed  # from here on, use the repaired path
                    except Exception as e2:
                        logging.error(f"[probe] Repair produced unreadable video; SKIP {key}: {e2}")
                        return result
                else:
                    logging.error(f"[probe] Repair failed; SKIP {key}: {e}")
                    return result
            else:
                logging.error(f"[probe] Unreadable video; SKIP {key}: {e}")
                return result

    df = parse_yolo_csv(csv_path)
    if df.empty:
        s_count = max(1, int(math.ceil(dur)))
        empty_vec = np.zeros(sum(g.gw*g.gh for g in grids), dtype=np.float32)
        if spec.l2_normalize and not spec.density_heatmap:
            empty_vec = ensure_unit_l2(empty_vec)

        if seconds_whitelist is not None and not compute_minute:
            return {
                view: {
                    "second_vecs_by_sec": {},
                    "second_locvecs_by_sec": {},
                    "second_loc_scalars_by_sec": {},
                    "fps": fps, "dur": dur, "img_w": img_w, "img_h": img_h, "path": mp4_path,
                    "dt0_utc": parse_key_datetime(key) or datetime.datetime.fromtimestamp(0, tz=datetime.timezone.utc),
                }
            }
        else:
            return {
                view: {
                    "second_vecs": [empty_vec] * s_count,
                    "second_locvecs": [np.zeros(7, dtype=np.float32)] * s_count,
                    "second_loc_scalars": [({"lat": None, "lon": None, "mph": None}, "none")] * s_count,
                    "minute_vec": empty_vec if compute_minute else None,
                    "fps": fps, "dur": dur, "img_w": img_w, "img_h": img_h, "path": mp4_path,
                    "dt0_utc": parse_key_datetime(key) or datetime.datetime.fromtimestamp(0, tz=datetime.timezone.utc),
                }
            }

    # Filtering
    if not keep_all:
        df["__keep__"] = df.apply(lambda r: keep_detection(r, keep_names, id_map), axis=1)
        df = df[df["__keep__"] == True].copy()

    # Assign seconds
    df = assign_seconds(df, fps=fps, dur=dur, key=key)

    # Optional metadata CSV → seconds
    meta_df = read_clip_metadata_csv(file_dir, key)
    meta_by_sec: Dict[int, Dict[str, Optional[float]]] = {}
    if meta_df is not None and "frame" in meta_df.columns:
        sec_col = meta_df["frame"].apply(lambda fr: second_from_frame(fr, fps))
        tmp = meta_df.copy(); tmp["__sec__"] = sec_col
        g = tmp.dropna(subset=["__sec__"]).groupby("__sec__", as_index=False).agg(
            lat=("lat","mean"),
            lon=("lon","mean"),
            mph=("mph","mean") if "mph" in tmp.columns else ("frame","count")
        )
        for r in g.itertuples(index=False):
            meta_by_sec[int(getattr(r, "__sec__"))] = {
                "lat": getattr(r, "lat", None),
                "lon": getattr(r, "lon", None),
                "mph": getattr(r, "mph", None) if "mph" in g.columns else None
            }

    s_count = max(1, int(math.ceil(dur)))
    dt0_utc = parse_key_datetime(key) or datetime.datetime.fromtimestamp(0, tz=datetime.timezone.utc)
    cells_total = sum(g.gw * g.gh for g in grids)

    # Decide which seconds we’re actually computing
    if (seconds_whitelist is not None) and (not compute_minute):
        seconds_iter = sorted(int(s) for s in seconds_whitelist if 0 <= s < s_count)
        # sparse per-second maps
        second_vecs_by_sec = {}
        second_locvecs_by_sec = {}
        second_loc_scalars_by_sec = {}

        for s in seconds_iter:
            sec_df = df[df["__sec__"] == s]
            bbox_ok = 0
            rows = []
            if not sec_df.empty:
                rows = list(sec_df.itertuples(index=False))
                for r in rows:
                    row_like = r._asdict() if hasattr(r, "_asdict") else r
                    if to_xyxy_abs(row_like, img_w, img_h) is not None:
                        bbox_ok += 1

            if bbox_ok > 0:
                vec = build_grid_embedding_for_interval(
                    rows, img_w, img_h, grids, spec.include_conf, density=spec.density_heatmap
                )
            else:
                vec = np.zeros(cells_total, dtype=np.float32)

            vec_final = vec if spec.density_heatmap else (ensure_unit_l2(vec) if spec.l2_normalize else vec)

            lat = lon = mph = None
            source = "metadata_csv" if s in meta_by_sec else "none"
            if s in meta_by_sec:
                lat = meta_by_sec[s].get("lat")
                lon = meta_by_sec[s].get("lon")
                mph = meta_by_sec[s].get("mph")

            t_utc = dt0_utc + datetime.timedelta(seconds=s)
            loc_vec, scalars = location_feature(lat, lon, mph, t_utc, include_time=spec.time_features)

            second_vecs_by_sec[s] = vec_final
            second_locvecs_by_sec[s] = loc_vec
            second_loc_scalars_by_sec[s] = (scalars, source)

        result[view] = {
            "second_vecs_by_sec": second_vecs_by_sec,
            "second_locvecs_by_sec": second_locvecs_by_sec,
            "second_loc_scalars_by_sec": second_loc_scalars_by_sec,
            "fps": fps, "dur": dur, "img_w": img_w, "img_h": img_h, "path": mp4_path,
            "dt0_utc": dt0_utc,
        }
        return result

    # Otherwise, compute full arrays (needed for minute or when whitelist is None)
    second_vecs: List[np.ndarray] = []
    second_locvecs: List[np.ndarray] = []
    second_loc_scalars: List[Tuple[Dict[str, Optional[float]], str]] = []

    for s in range(s_count):
        sec_df = df[df["__sec__"] == s]
        rows = []
        bbox_ok = 0
        if not sec_df.empty:
            rows = list(sec_df.itertuples(index=False))
            for r in rows:
                row_like = r._asdict() if hasattr(r, "_asdict") else r
                if to_xyxy_abs(row_like, img_w, img_h) is not None:
                    bbox_ok += 1

        if bbox_ok > 0:
            vec = build_grid_embedding_for_interval(
                rows, img_w, img_h, grids, spec.include_conf, density=spec.density_heatmap
            )
        else:
            vec = np.zeros(cells_total, dtype=np.float32)

        vec_final = vec if spec.density_heatmap else (ensure_unit_l2(vec) if spec.l2_normalize else vec)

        lat = lon = mph = None
        source = "metadata_csv" if s in meta_by_sec else "none"
        if s in meta_by_sec:
            lat = meta_by_sec[s].get("lat")
            lon = meta_by_sec[s].get("lon")
            mph = meta_by_sec[s].get("mph")

        t_utc = dt0_utc + datetime.timedelta(seconds=s)
        loc_vec, scalars = location_feature(lat, lon, mph, t_utc, include_time=spec.time_features)

        second_vecs.append(vec_final)
        second_locvecs.append(loc_vec)
        second_loc_scalars.append((scalars, source))

    minute_vec = None
    if compute_minute:
        minute_vec = aggregate_seconds_to_minute(second_vecs, mode=("sum" if spec.density_heatmap else "mean"))
        if minute_vec is None:
            minute_vec = np.zeros_like(second_vecs[0])
        if not spec.density_heatmap and spec.l2_normalize:
            minute_vec = ensure_unit_l2(minute_vec)

        if include_heatmap_png and minute_heatmap_out:
            finest = grids[-1]; grid_sz = finest.gh * finest.gw
            start = sum(g.gw*g.gh for g in grids[:-1])
            last = minute_vec[start:start + grid_sz]
            try:
                import matplotlib.pyplot as plt
                plt.imshow(last.reshape(finest.gh, finest.gw), cmap='hot', interpolation='nearest')
                plt.colorbar(); plt.title(f"Minute Heatmap: {key}")
                plt.savefig(minute_heatmap_out); plt.clf()
            except Exception as e:
                print(f"[DEBUG] {key}: failed to save minute heatmap: {e}")

    result[view] = {
        "second_vecs": second_vecs,
        "second_locvecs": second_locvecs,
        "second_loc_scalars": second_loc_scalars,
        "minute_vec": minute_vec,
        "fps": fps, "dur": dur, "img_w": img_w, "img_h": img_h, "path": mp4_path,
        "dt0_utc": dt0_utc
    }
    return result


# =========================
# NEW: read per-clip metadata CSV (frame→lat/lon/mph)
# =========================
def read_clip_metadata_csv(file_dir: str, key: str) -> Optional[pd.DataFrame]:
    p = os.path.join(file_dir, f"{key}_metadata.csv")
    if not os.path.exists(p): return None
    try:
        df = pd.read_csv(p)
        # normalize column names
        cols = {c.lower():c for c in df.columns}
        # expected: frame, lat/latitude, long/longitude, mph/speed
        def pick(*names):
            for n in names:
                if n in cols: return cols[n]
            return None
        fcol = pick("frame")
        latc = pick("lat","latitude")
        lonc = pick("long","longitude","lon")
        mphc = pick("mph","speed")
        if not fcol or not (latc and lonc):
            return None
        out = df[[fcol] + [x for x in [latc,lonc,mphc] if x]].copy()
        out.columns = ["frame","lat","lon"] + (["mph"] if mphc else [])
        out["frame"] = pd.to_numeric(out["frame"], errors="coerce").astype("Int64")
        if "mph" in out.columns:
            out["mph"] = pd.to_numeric(out["mph"], errors="coerce")
        return out
    except Exception as e:
        logging.warning(f"Failed to read metadata CSV for {key}: {e}")
        return None


# =========================
# Directory processing (UPDATED to attach location links / fallbacks)
# =========================
def process_directory(
    date_dir: str,
    spec: EmbeddingSpec,
    grids: List[GridSpec],
    keep_names: set,
    id_map: Dict[int,str],
    include_heatmap_png: bool,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_pwd: str,
    win_mins: int,
    resume: bool,
    force: bool,
    force_minute: bool,
    keep_all: bool = False,
    repair_missing_moov: bool = False,
):
    keys = find_file_keys(date_dir)
    if not keys:
        return

    driver = neo4j_session(neo4j_uri, neo4j_user, neo4j_pwd)
    with driver.session() as sess:
        neo4j_create_constraints(sess)

        for key in keys:
            logging.info(f"Processing {key}")
            view_suffix = key.split("_")[-1]

            # ---------- Preflight: ask DB what's already done ----------
            dur_db, fps_db, width_db, height_db, path_db = neo4j_get_clip_meta(sess, key)
            minute_done = neo4j_minute_exists(sess, key, view_suffix)
            secs_done = neo4j_get_existing_seconds(sess, key, view_suffix)

            total_secs_db = max(1, int(math.ceil(dur_db))) if dur_db is not None else None

            # If not forcing and DB knows duration and everything is present → skip clip
            if not force and (total_secs_db is not None) and minute_done and (len(secs_done) >= total_secs_db):
                logging.info(f"  SKIP: fully ingested {key} ({len(secs_done)}/{total_secs_db} secs + minute)")
                continue

            # Decide whether we need to recompute minute
            need_minute = (spec.per_minute and (force or force_minute or not minute_done))

            # We’ll compute per-second only for missing seconds when possible
            missing_secs: Optional[set] = None
            if spec.per_second and not force:
                if total_secs_db is not None:
                    all_secs = set(range(total_secs_db))
                    missing_secs = all_secs - set(int(s) for s in secs_done)
                    if not need_minute and not missing_secs:
                        logging.info(f"  SKIP: all seconds already present and minute exists for {key}")
                        continue
                else:
                    # we don't know duration yet → we’ll compute full once to learn it
                    missing_secs = None

            # Preknown video meta if DB has it (avoid opening video)
            pre_meta = None
            if (width_db is not None) and (height_db is not None) and (fps_db is not None) and (dur_db is not None):
                pre_meta = (int(width_db), int(height_db), float(fps_db), float(dur_db))

            out_png = os.path.join(date_dir, f"{key}_heatmap.png") if include_heatmap_png else None

            # ---------- Build vectors ----------
            if need_minute:
                # minute recompute requires all seconds
                views_data = build_vectors_for_key(
                    date_dir, key, spec, keep_names, id_map, grids,
                    include_heatmap_png, out_png, keep_all=keep_all,
                    seconds_whitelist=None, compute_minute=True,
                    pre_meta=pre_meta, repair_missing_moov=repair_missing_moov
                )
            else:
                # compute only missing seconds (fast path)
                views_data = build_vectors_for_key(
                    date_dir, key, spec, keep_names, id_map, grids,
                    include_heatmap_png, out_png, keep_all=keep_all,
                    seconds_whitelist=missing_secs, compute_minute=False,
                    pre_meta=pre_meta, repair_missing_moov=repair_missing_moov
                )

            if not views_data:
                continue

            # Upsert clip meta (once)
            any_view = next(iter(views_data.values()), None)
            if any_view:
                neo4j_upsert_clip(
                    sess, key=key, path=any_view.get("path", path_db) or path_db or os.path.join(date_dir, f"{key}.MP4"),
                    view=view_suffix,
                    width=any_view.get("img_w") or width_db or 0,
                    height=any_view.get("img_h") or height_db or 0,
                    fps=any_view.get("fps") or fps_db or 30.0,
                    dur=any_view.get("dur") or dur_db or 0.0
                )

            for vview, data in views_data.items():
                fps = float(data["fps"])
                dur = float(data["dur"])
                total_secs = max(1, int(math.ceil(dur)))

                # Figure out which seconds we actually have vectors for in this build
                if "second_vecs_by_sec" in data:
                    # sparse mode (we built only missing seconds)
                    sec_map_vec = data["second_vecs_by_sec"]
                    sec_map_locv = data["second_locvecs_by_sec"]
                    sec_map_scal = data["second_loc_scalars_by_sec"]

                    if spec.per_second:
                        prev_fix = None
                        prev_fix_sec = None
                        carry_secs = 10
                        resolved_locvecs_for_minute = []
                        resolved_scalars_for_minute = []

                        for s_idx in sorted(sec_map_vec.keys()):
                            vec = sec_map_vec[s_idx]
                            _loc_init = sec_map_locv.get(s_idx)
                            (meta_scalars, _src_ignored) = sec_map_scal.get(s_idx, ({"lat":None,"lon":None,"mph":None},"none"))

                            emb_id = f"{key}|{vview}|sec|{s_idx}"
                            t_utc = data["dt0_utc"] + datetime.timedelta(seconds=s_idx)

                            scalars_resolved, source, near_elem, seconds_delta = resolve_location_for_second(
                                sess=sess,
                                key=key,
                                fps=fps,
                                second=int(s_idx),
                                dt0_utc=data["dt0_utc"],
                                win_mins=win_mins,
                                meta_scalars=meta_scalars,
                                prev_fix=prev_fix,
                            )
                            if source == "none" and prev_fix is not None and prev_fix_sec is not None:
                                if (s_idx - prev_fix_sec) <= carry_secs:
                                    scalars_resolved = {"lat": prev_fix[0], "lon": prev_fix[1], "mph": prev_fix[2]}
                                    source = "carryforward"

                            loc_vec, scalars = location_feature(
                                scalars_resolved.get("lat"),
                                scalars_resolved.get("lon"),
                                scalars_resolved.get("mph"),
                                t_utc,
                                include_time=spec.time_features
                            )

                            if near_elem:
                                sess.run(
                                    NQ_ATTACH_NEAR_BY_ELEMID,
                                    eid=emb_id,
                                    elem_id=near_elem,
                                    seconds=int(seconds_delta or 0),
                                    source=source
                                )

                            neo4j_upsert_embed(
                                sess,
                                emb_id=emb_id, level="second", key=key, view=vview,
                                t0=int(s_idx), t1=int(min(s_idx+1, total_secs)),
                                vec=vec, loc_vec=loc_vec, loc_scalars=scalars, loc_source=source,
                                spec=spec, grids=grids
                            )

                            neo4j_enrich_frame(
                                sess,
                                key=key,
                                frame_idx=int(s_idx * round(fps)),
                                lat=scalars.get("lat"),
                                lon=scalars.get("lon"),
                                mph=scalars.get("mph")
                            )

                            if scalars.get("lat") is not None and scalars.get("lon") is not None:
                                prev_fix = (scalars["lat"], scalars["lon"], scalars.get("mph"))
                                prev_fix_sec = int(s_idx)

                            logging.info(f"[GPS] {key} sec={s_idx} source={source} lat={scalars.get('lat')} lon={scalars.get('lon')}")

                            resolved_locvecs_for_minute.append(loc_vec)
                            resolved_scalars_for_minute.append(scalars)

                    if spec.per_second:
                        neo4j_rebuild_next(sess, key=key, view=vview)

                    # In sparse mode we did not compute minute; do nothing here.
                    continue

                # Full arrays mode (used when minute recompute is requested or whitelist not used)
                if spec.per_second:
                    existing_secs = neo4j_get_existing_seconds(sess, key, vview) if not force else set()

                    prev_fix = None
                    prev_fix_sec = None
                    carry_secs = 10
                    resolved_locvecs_for_minute = []
                    resolved_scalars_for_minute = []

                    for s_idx, (vec, _locv_init, (meta_scalars, _src_ignored)) in enumerate(
                        zip(data["second_vecs"], data["second_locvecs"], data["second_loc_scalars"])
                    ):
                        if not force and (s_idx in existing_secs):
                            continue

                        emb_id = f"{key}|{vview}|sec|{s_idx}"
                        t_utc = data["dt0_utc"] + datetime.timedelta(seconds=s_idx)

                        scalars_resolved, source, near_elem, seconds_delta = resolve_location_for_second(
                            sess=sess,
                            key=key,
                            fps=fps,
                            second=s_idx,
                            dt0_utc=data["dt0_utc"],
                            win_mins=win_mins,
                            meta_scalars=meta_scalars,
                            prev_fix=prev_fix,
                        )

                        if source == "none" and prev_fix is not None and prev_fix_sec is not None:
                            if (s_idx - prev_fix_sec) <= carry_secs:
                                scalars_resolved = {"lat": prev_fix[0], "lon": prev_fix[1], "mph": prev_fix[2]}
                                source = "carryforward"

                        loc_vec, scalars = location_feature(
                            scalars_resolved.get("lat"),
                            scalars_resolved.get("lon"),
                            scalars_resolved.get("mph"),
                            t_utc,
                            include_time=spec.time_features
                        )

                        if near_elem:
                            sess.run(
                                NQ_ATTACH_NEAR_BY_ELEMID,
                                eid=emb_id,
                                elem_id=near_elem,
                                seconds=int(seconds_delta or 0),
                                source=source
                            )

                        neo4j_upsert_embed(
                            sess,
                            emb_id=emb_id, level="second", key=key, view=vview,
                            t0=int(s_idx), t1=int(min(s_idx+1, total_secs)),
                            vec=vec, loc_vec=loc_vec, loc_scalars=scalars, loc_source=source,
                            spec=spec, grids=grids
                        )

                        neo4j_enrich_frame(
                            sess,
                            key=key,
                            frame_idx=int(s_idx * round(fps)),
                            lat=scalars.get("lat"),
                            lon=scalars.get("lon"),
                            mph=scalars.get("mph")
                        )

                        if scalars.get("lat") is not None and scalars.get("lon") is not None:
                            prev_fix = (scalars["lat"], scalars["lon"], scalars.get("mph"))
                            prev_fix_sec = s_idx

                        logging.info(f"[GPS] {key} sec={s_idx} source={source} lat={scalars.get('lat')} lon={scalars.get('lon')}")
                        resolved_locvecs_for_minute.append(loc_vec)
                        resolved_scalars_for_minute.append(scalars)

                if spec.per_second:
                    neo4j_rebuild_next(sess, key=key, view=vview)

                # Minute embedding if requested in this pass
                if spec.per_minute and data.get("minute_vec") is not None:
                    mvec = data["minute_vec"]
                    if spec.l2_normalize and not spec.density_heatmap:
                        mvec = ensure_unit_l2(mvec)

                    if 'resolved_locvecs_for_minute' in locals() and resolved_locvecs_for_minute:
                        mloc = ensure_unit_l2(np.nanmean(np.stack(resolved_locvecs_for_minute, 0), axis=0))
                        scalars_agg = next((sc for sc in resolved_scalars_for_minute
                                            if sc.get("lat") is not None and sc.get("lon") is not None),
                                           {"lat": None, "lon": None, "mph": None})
                    else:
                        mloc = np.zeros((7,), dtype=np.float32)
                        scalars_agg = {"lat": None, "lon": None, "mph": None}

                    neo4j_upsert_embed(
                        sess,
                        emb_id=f"{key}|{vview}|minute|0", level="minute",
                        key=key, view=vview, t0=0, t1=int(total_secs),
                        vec=mvec, loc_vec=mloc, loc_scalars=scalars_agg, loc_source="mixed",
                        spec=spec, grids=grids
                    )

    driver.close()


# =========================
# File discovery (hardened)
# =========================
def find_file_keys(directory: str) -> List[str]:
    keys = set()
    for fn in os.listdir(directory):
        if re.search(r"_YOLOv8n\.csv$", fn):
            k = fn.rsplit("_YOLOv8n", 1)[0]
            mp4 = os.path.join(directory, f"{k}.MP4")
            if os.path.exists(mp4) and os.path.getsize(mp4) > 0:
                keys.add(k)
            else:
                logging.warning(f"[scan] Missing or empty MP4 for {k}, skipping")
    return sorted(keys)

def is_yyyymmdd_dir(path: str) -> bool:
    parts = os.path.normpath(path).split(os.sep)
    if len(parts) < 3: return False
    y,m,d = parts[-3], parts[-2], parts[-1]
    if not (y.isdigit() and m.isdigit() and d.isdigit() and len(y)==4 and len(m)==2 and len(d)==2): return False
    try: pd.Timestamp(f"{y}-{m}-{d}"); return True
    except Exception: return False

def walk_date_dirs(base: str) -> List[str]:
    targets=[]
    for root, dirs, files in os.walk(base):
        if is_yyyymmdd_dir(root): targets.append(root)
    return sorted(targets)


# =========================
# CLI
# =========================
def main():
    parser = argparse.ArgumentParser(description="YOLO structure embeddings with location augmentation → Neo4j")
    parser.add_argument("--bases", nargs="+", default=[
        "/mnt/8TB_2025/fileserver/dashcam/",
        "/mnt/8TBHDD/fileserver/dashcam/"
    ])
    parser.add_argument("--grid", default="16x9")
    parser.add_argument("--pyramid", action="store_true")
    parser.add_argument("--no-conf", action="store_true")
    parser.add_argument("--no-l2", action="store_true")
    parser.add_argument("--no-second", action="store_true")
    parser.add_argument("--no-minute", action="store_true")
    parser.add_argument("--heatmap", action="store_true")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-pass", default="password")
    parser.add_argument("--classes", nargs="*", default=list(DEFAULT_VEHICLE_CLASSES))
    parser.add_argument("--no-append-location", action="store_true", help="Do not append loc_vec to scene vector")
    parser.add_argument("--no-time-features", action="store_true", help="Do not add time-of-day sin/cos to loc_vec")
    parser.add_argument("--win-mins", type=int, default=10, help="± minutes window for PhoneLog/LocationEvent fallback")
    parser.add_argument("--resume", action="store_true", help="Skip already ingested seconds/minute for each clip/view; fill only missing parts")
    parser.add_argument("--force", action="store_true", help="Recompute and overwrite everything (disables --resume)")
    parser.add_argument("--force-minute", action="store_true", help="Recompute minute embeddings even if they already exist")
    parser.add_argument("--keep-all", action="store_true", help="Keep ALL detections regardless of class/name (disables filtering)")
    parser.add_argument("--no-density-heatmap", action="store_true", help="Use mean + L2 style embeddings instead of density accumulation")
    parser.add_argument("--repair-missing-moov", action="store_true",
                        help="Attempt to faststart/fix MP4s with missing moov atom (no re-encode)")

    args = parser.parse_args()
    gw, gh = map(int, args.grid.lower().split("x"))
    grids = [GridSpec(gw, gh)]
    if args.pyramid: grids = [GridSpec(8,4), GridSpec(gw,gh), GridSpec(32,18)]

    spec = EmbeddingSpec(
        grids=grids,
        include_conf=not args.no_conf,
        l2_normalize=not args.no_l2,
        concat_views=True,
        per_second=not args.no_second,
        per_minute=not args.no_minute,
        append_location=not args.no_append_location,
        time_features=not args.no_time_features,
        density_heatmap=not args.no_density_heatmap,
    )

    keep_names = set([s.lower() for s in args.classes]); id_map = DEFAULT_CLASS_ID_MAP
    base_dirs = [os.path.abspath(b) for b in args.bases]
    date_dirs=[]
    for b in base_dirs:
        if not os.path.isdir(b):
            logging.warning(f"Base not found: {b}"); continue
        date_dirs.extend(walk_date_dirs(b))
    logging.info(f"Found {len(date_dirs)} date folders")

    for dd in date_dirs:
        logging.info(f"Processing date dir: {dd}")
        process_directory(
            dd, spec, grids, keep_names, id_map,
            include_heatmap_png=bool(args.heatmap),
            neo4j_uri=args.neo4j_uri, neo4j_user=args.neo4j_user, neo4j_pwd=args.neo4j_pass,
            win_mins=args.win_mins, resume=args.resume, force=args.force,
            force_minute=args.force_minute, keep_all=args.keep_all,
            repair_missing_moov=args.repair_missing_moov
        )

if __name__ == "__main__":
    main()
