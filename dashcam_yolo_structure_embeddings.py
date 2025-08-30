#!/usr/bin/env python3
import os, re, ast, json, math, logging, argparse
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
# Vehicle-ish classes to keep (names or ids—script will accept either and map to keep set)
DEFAULT_VEHICLE_CLASSES = {
    "car","truck","bus","motorbike","motorcycle","bicycle","van","pickup","suv","trailer"
}
# If your CSV has class *ids*, map them here (example only; adjust to your YOLO model)
# e.g., COCO-style: car=2, motorcycle=3, bus=5, truck=7, bicycle=1 …
DEFAULT_CLASS_ID_MAP = {  # id -> name
    1: "bicycle", 2: "car", 3: "motorbike", 5: "bus", 7: "truck"
}

@dataclass
class GridSpec:
    gw: int  # grid width (columns)
    gh: int  # grid height (rows)

@dataclass
class EmbeddingSpec:
    grids: List[GridSpec]   # one or more grids for spatial pyramid
    include_conf: bool = True     # multiply weight by confidence
    l2_normalize: bool = True     # normalize every vector to unit L2
    concat_views: bool = True     # concat F and R vectors (recommended)
    per_second: bool = True       # produce per-second embeddings
    per_minute: bool = True       # produce per-minute embeddings

# =========================
# Utility
# =========================
def safe_literal_eval(s: str):
    try:
        return ast.literal_eval(s)
    except Exception:
        return None

def ensure_unit_l2(vec: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec)
    if n == 0.0 or not np.isfinite(n):
        return vec
    return vec / n

def flatten_grid(grid: np.ndarray) -> np.ndarray:
    return grid.astype(np.float32).ravel()

def bbox_overlap_area(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2) -> float:
    """Overlap area of two axis-aligned rectangles."""
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return float(ix2 - ix1) * float(iy2 - iy1)

def add_bbox_to_grid(grid: np.ndarray, bbox: Tuple[float,float,float,float],
                     img_w: int, img_h: int, weight: float, gs: GridSpec):
    """
    Distribute 'weight' into the grid cells proportional to the overlap area
    between bbox and each cell. bbox is in absolute pixel xyxy.
    """
    gw, gh = gs.gw, gs.gh
    x1, y1, x2, y2 = bbox
    # clamp
    x1 = max(0, min(img_w, x1)); x2 = max(0, min(img_w, x2))
    y1 = max(0, min(img_h, y1)); y2 = max(0, min(img_h, y2))
    if x2 <= x1 or y2 <= y1:
        return

    cell_w = img_w / gw
    cell_h = img_h / gh

    # cell index range overlapped by bbox
    cx1 = int(np.floor(x1 / cell_w))
    cx2 = int(np.floor((x2 - 1e-6) / cell_w))
    cy1 = int(np.floor(y1 / cell_h))
    cy2 = int(np.floor((y2 - 1e-6) / cell_h))

    cx1 = max(0, min(gw - 1, cx1))
    cx2 = max(0, min(gw - 1, cx2))
    cy1 = max(0, min(gh - 1, cy1))
    cy2 = max(0, min(gh - 1, cy2))

    bbox_area = (x2 - x1) * (y2 - y1)
    if bbox_area <= 0:
        return

    for cy in range(cy1, cy2 + 1):
        cell_y1 = cy * cell_h
        cell_y2 = (cy + 1) * cell_h
        for cx in range(cx1, cx2 + 1):
            cell_x1 = cx * cell_w
            cell_x2 = (cx + 1) * cell_w
            overlap = bbox_overlap_area(x1, y1, x2, y2, cell_x1, cell_y1, cell_x2, cell_y2)
            if overlap > 0:
                grid[cy, cx] += weight * (overlap / bbox_area)

def parse_yolo_csv(csv_path: str) -> pd.DataFrame:
    """
    Robustly parse a YOLO CSV that may contain xyxy or xyxyn or xywh columns, plus frame, class, confidence.
    Expected header columns (any subset): frame, class, name, confidence, xyxy, xyxyn, xywh, t, vehicle_id
    """
    rows = []
    with open(csv_path, "r") as f:
        header = f.readline().strip().split(",")
        for line in f:
            parts = []
            temp = ""
            in_brackets = False
            for ch in line:
                if ch == "[":
                    in_brackets = True
                if ch == "]":
                    in_brackets = False
                if ch == "," and not in_brackets:
                    parts.append(temp.strip()); temp = ""
                else:
                    temp += ch
            parts.append(temp.strip())
            # convert bracketed parts to lists
            parts = [safe_literal_eval(p) if ("[" in p and "]" in p) else p for p in parts]
            rows.append(dict(zip(header, parts)))
    df = pd.DataFrame(rows)

    # Normalize column names and types
    if "confidence" in df.columns:
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(1.0).clip(0, 1)
    else:
        df["confidence"] = 1.0

    # class name/ids
    if "name" not in df.columns:
        df["name"] = None
    if "class" in df.columns:
        df["class"] = pd.to_numeric(df["class"], errors="coerce")

    # xyxy absolute preferred
    if "xyxy" not in df.columns or df["xyxy"].isna().all():
        # fallback: xyxyn (normalized) -> convert using video size later
        if "xyxyn" not in df.columns and "xywh" not in df.columns:
            raise ValueError(f"{csv_path}: no bbox columns found (xyxy/xyxyn/xywh)")
    return df

def to_xyxy_abs(row: pd.Series, img_w: int, img_h: int) -> Optional[Tuple[int,int,int,int]]:
    """
    Return absolute pixel xyxy bbox for a row using available columns.
    """
    if "xyxy" in row and isinstance(row["xyxy"], list) and len(row["xyxy"]) == 4:
        x1,y1,x2,y2 = row["xyxy"]
        return int(x1), int(y1), int(x2), int(y2)
    if "xyxyn" in row and isinstance(row["xyxyn"], list) and len(row["xyxyn"]) == 4:
        x1n,y1n,x2n,y2n = row["xyxyn"]
        return int(x1n * img_w), int(y1n * img_h), int(x2n * img_w), int(y2n * img_h)
    if "xywh" in row and isinstance(row["xywh"], list) and len(row["xywh"]) == 4:
        # xywh absolute or normalized? try to infer: if values <= 1.0 assume normalized
        cx, cy, w, h = row["xywh"]
        if max(cx, cy, w, h) <= 1.0001:
            cx *= img_w; cy *= img_h; w *= img_w; h *= img_h
        x1 = int(cx - w/2); y1 = int(cy - h/2); x2 = int(cx + w/2); y2 = int(cy + h/2)
        return x1, y1, x2, y2
    return None

def keep_detection(row: pd.Series,
                   keep_names: set,
                   id_map: Dict[int,str]) -> bool:
    name = row.get("name")
    cid  = row.get("class")
    if isinstance(name, str) and name.lower() in keep_names:
        return True
    if pd.notna(cid) and int(cid) in id_map and id_map[int(cid)] in keep_names:
        return True
    return False

def second_from_frame(frame: Optional[float], fps: float) -> Optional[int]:
    if frame is None or not np.isfinite(frame):
        return None
    try:
        return int(frame // fps)
    except Exception:
        return None

def element_id_safe(s: str) -> str:
    # A stable id you can use for uniqueness (also okay as a property with a unique constraint)
    return s

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
    # Helper to extract
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
        # else: allow as last resort if still plausible
        if plausible_fix(prev_fix, mlat, mlon, mmph):
            return {"lat":mlat, "lon":mlon, "mph":mmph}, "metadata_csv"

    return {"lat":None, "lon":None, "mph":None}, "none"


# =========================
# Embedding builders
# =========================
def build_grid_embedding_for_interval(
    detections: Iterable[pd.Series],
    img_w: int, img_h: int,
    grids: List[GridSpec],
    include_conf: bool,
) -> np.ndarray:
    """
    Build a spatial pyramid vector by accumulating fractional-overlap weights for all bboxes in 'detections'
    """
    parts: List[np.ndarray] = []
    for gs in grids:
        grid = np.zeros((gs.gh, gs.gw), dtype=np.float32)
        for row in detections:
            conf = float(row.get("confidence", 1.0)) if include_conf else 1.0
            bbox = to_xyxy_abs(row, img_w, img_h)
            if bbox is None:
                continue
            add_bbox_to_grid(grid, bbox, img_w, img_h, weight=conf, gs=gs)
        parts.append(flatten_grid(grid))
    vec = np.concatenate(parts, axis=0)
    return vec

def aggregate_seconds_to_minute(second_vecs: List[np.ndarray]) -> np.ndarray:
    """
    Mean-pool second vectors (zero-fill missing seconds) → minute-level vector.
    """
    if not second_vecs:
        return None
    M = np.stack(second_vecs, axis=0)  # [S, D]
    # mean over available seconds (if any seconds are missing, mean over the present ones)
    return np.nanmean(M, axis=0)

# =========================
# Neo4j I/O
# =========================
NQ_CREATE_CONSTRAINTS = """
CREATE CONSTRAINT dashcam_clip_key IF NOT EXISTS
FOR (c:DashcamClip) REQUIRE c.key IS UNIQUE;

CREATE CONSTRAINT dashcam_embedding_id IF NOT EXISTS
FOR (e:DashcamEmbedding) REQUIRE e.id IS UNIQUE;
"""

NQ_UPSERT_CLIP = """
MERGE (c:DashcamClip {key: $key})
ON CREATE SET c.path = $path, c.view = $view, c.width = $width, c.height = $height,
              c.fps = $fps, c.duration_s = $duration_s, c.created_at = timestamp()
ON MATCH  SET c.path = $path, c.width = $width, c.height = $height,
              c.fps = $fps, c.duration_s = $duration_s, c.updated_at = timestamp()
RETURN c
"""

NQ_UPSERT_EMBED = """
MERGE (e:DashcamEmbedding {id: $id})
ON CREATE SET e.level = $level, e.key = $key, e.view = $view,
              e.t0 = $t0, e.t1 = $t1, e.dim = $dim, e.grids = $grids,
              e.l2 = $l2, e.model = $model, e.concat_views = $concat_views,
              e.created_at = timestamp(), e.vec = $vec
ON MATCH  SET e.vec = $vec, e.l2 = $l2, e.updated_at = timestamp(),
              e.dim = $dim, e.grids = $grids, e.concat_views = $concat_views
WITH e
MATCH (c:DashcamClip {key: $key})
MERGE (c)-[:HAS_EMBEDDING {level:$level}]->(e)
RETURN e
"""

NQ_LINK_NEXT = """
MATCH (a:DashcamEmbedding {id:$a_id}), (b:DashcamEmbedding {id:$b_id})
MERGE (a)-[:NEXT]->(b)
"""

# --- NEW: nearest lookups that RETURN node elementId + coords (no edge yet) ---
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

def neo4j_session(uri: str, user: str, pwd: str):
    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    return driver

def neo4j_create_constraints(sess):
    sess.run(NQ_CREATE_CONSTRAINTS)

def neo4j_upsert_clip(sess, key: str, path: str, view: str, width: int, height: int, fps: float, dur: float):
    return sess.run(NQ_UPSERT_CLIP, key=key, path=path, view=view, width=width,
                    height=height, fps=fps, duration_s=int(dur)).single()

def neo4j_upsert_embed(sess, *, emb_id: str, level: str, key: str, view: str,
                       t0: int, t1: int, vec: np.ndarray, spec: EmbeddingSpec, grids: List[GridSpec],
                       model: str = "yolov8n"):

    v = vec.astype(np.float32)
    v = ensure_unit_l2(v) if spec.l2_normalize else v
    params = {
        "id": emb_id,
        "level": level,         # "second" or "minute"
        "key": key,
        "view": view,
        "t0": int(t0),
        "t1": int(t1),
        "dim": int(v.shape[0]),
        "vec": v.tolist(),
        "l2": True if spec.l2_normalize else False,
        "grids": [{"gw": g.gw, "gh": g.gh} for g in grids],
        "model": model,
        "concat_views": bool(spec.concat_views),
    }
    return sess.run(NQ_UPSERT_EMBED, **params).single()

def neo4j_link_next(sess, a_id: str, b_id: str):
    sess.run(NQ_LINK_NEXT, a_id=a_id, b_id=b_id)

# =========================
# Core processing
# =========================
def find_file_keys(directory: str) -> List[str]:
    keys = set()
    for fn in os.listdir(directory):
        if re.search(r"_YOLOv8n\.csv$", fn):
            keys.add(fn.rsplit("_YOLOv8n", 1)[0])  # e.g., 2025_0101_173933_F
    # keep keys only if MP4 exists
    valids = []
    for k in sorted(keys):
        if os.path.exists(os.path.join(directory, f"{k}.MP4")):
            valids.append(k)
    return valids

def is_yyyymmdd_dir(path: str) -> bool:
    # expects .../YYYY/MM/DD
    parts = os.path.normpath(path).split(os.sep)
    if len(parts) < 3: return False
    y,m,d = parts[-3], parts[-2], parts[-1]
    if not (y.isdigit() and m.isdigit() and d.isdigit() and len(y)==4 and len(m)==2 and len(d)==2):
        return False
    try:
        pd.Timestamp(f"{y}-{m}-{d}")
        return True
    except Exception:
        return False

def walk_date_dirs(base: str) -> List[str]:
    targets = []
    for root, dirs, files in os.walk(base):
        if is_yyyymmdd_dir(root):
            targets.append(root)
    return sorted(targets)

def build_vectors_for_key(
    file_dir: str,
    key: str,
    spec: EmbeddingSpec,
    keep_names: set,
    id_map: Dict[int,str],
    grids: List[GridSpec],
    include_heatmap_png: bool,
    minute_heatmap_out: Optional[str] = None
) -> Dict[str, Dict]:
    """
    Returns:
      {
        "F": {"second_vecs": [np.ndarray...], "minute_vec": np.ndarray, "fps": float, "dur": float, "img_w": int, "img_h": int, "path": str},
        "R": {...}  # if present
      }
    """
    result: Dict[str, Dict] = {}
    mp4_path = os.path.join(file_dir, f"{key}.MP4")
    if not os.path.exists(mp4_path):
        return result

    # Derive view ("F" or "R") from key suffix
    view = "F" if key.endswith("_F") or key.endswith("_FR") or key.endswith("_F.MP4") else ("R" if key.endswith("_R") else "?")

    # Load video meta
    with VideoFileClip(mp4_path) as clip:
        fps = float(clip.fps or 30.0)
        dur = float(clip.duration or 60.0)
        img_w, img_h = int(clip.w), int(clip.h)

    csv_path = os.path.join(file_dir, f"{key}_YOLOv8n.csv")
    if not os.path.exists(csv_path):
        return result

    df = parse_yolo_csv(csv_path)

    # Filter by class set
    df["__keep__"] = df.apply(lambda r: keep_detection(r, keep_names, id_map), axis=1)
    df = df[df["__keep__"] == True].copy()
    if df.empty:
        # Still produce zero vectors to keep indexing aligned
        s_count = max(1, int(math.ceil(dur)))
        dim = sum(g.gw*g.gh for g in grids)
        z = np.zeros((s_count, dim), dtype=np.float32)
        minute_vec = np.zeros((dim,), dtype=np.float32)
        result[view] = {"second_vecs": list(z), "minute_vec": minute_vec,
                        "fps": fps, "dur": dur, "img_w": img_w, "img_h": img_h, "path": mp4_path}
        return result

    # Second assignment
    frame_col = "frame" if "frame" in df.columns else None
    if frame_col:
        df["__sec__"] = df[frame_col].apply(lambda fr: second_from_frame(pd.to_numeric(fr, errors="coerce"), fps))
    elif "t" in df.columns:
        df["__sec__"] = pd.to_numeric(df["t"], errors="coerce").apply(lambda t: int(np.floor(t)))
    else:
        # best effort: assume ordered rows across the minute
        s_count = max(1, int(math.ceil(dur)))
        df["__sec__"] = np.random.randint(0, s_count, size=len(df))  # fallback (rare)

    # Build per-second vectors
    s_count = max(1, int(math.ceil(dur)))
    second_vecs: List[np.ndarray] = []
    for s in range(s_count):
        rows = (df[df["__sec__"] == s]).itertuples(index=False)
        vec = build_grid_embedding_for_interval(
            detections=rows, img_w=img_w, img_h=img_h,
            grids=grids, include_conf=spec.include_conf
        )
        if spec.l2_normalize:
            vec = ensure_unit_l2(vec)
        second_vecs.append(vec)

    # Minute vector = mean of seconds (then L2 normalize to be comparable)
    minute_vec = aggregate_seconds_to_minute(second_vecs)
    if minute_vec is None:
        minute_vec = np.zeros_like(second_vecs[0])
    if spec.l2_normalize:
        minute_vec = ensure_unit_l2(minute_vec)

    # Optional: output a PNG heatmap (minute)
    if include_heatmap_png and minute_heatmap_out:
        # render simple 1-grid heatmap using the finest grid
        finest = grids[-1]
        grid_sz = finest.gh * finest.gw
        start = sum(g.gw*g.gh for g in grids[:-1])
        last = minute_vec[start:start+grid_sz]  # take last level
        grid = last.reshape(finest.gh, finest.gw)
        import matplotlib.pyplot as plt
        plt.imshow(grid, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title(f"Minute Heatmap: {key}")
        plt.savefig(minute_heatmap_out)
        plt.clf()

    result[view] = {
        "second_vecs": second_vecs,
        "minute_vec": minute_vec,
        "fps": fps, "dur": dur, "img_w": img_w, "img_h": img_h,
        "path": mp4_path
    }
    return result

def process_directory(
    date_dir: str,
    spec: EmbeddingSpec,
    grids: List[GridSpec],
    keep_names: set,
    id_map: Dict[int,str],
    include_heatmap_png: bool,
    neo4j_uri: str, neo4j_user: str, neo4j_pwd: str,
    concat_views: bool
):
    keys = find_file_keys(date_dir)
    if not keys:
        return

    driver = neo4j_session(neo4j_uri, neo4j_user, neo4j_pwd)
    with driver.session() as sess:
        neo4j_create_constraints(sess)
        for key in keys:
            logging.info(f"Processing {key}")
            out_png = os.path.join(date_dir, f"{key}_heatmap.png") if include_heatmap_png else None
            views_data = build_vectors_for_key(
                date_dir, key, spec, keep_names, id_map, grids,
                include_heatmap_png=include_heatmap_png,
                minute_heatmap_out=out_png
            )
            # derive view label in key (ensure _F/_R)
            view_suffix = key.split("_")[-1]
            base_key = key  # keep as-is, your naming already encodes view

            # If concat_views=True AND the sibling view exists for same timestamp, handle later.
            # Upsert clip node first
            # Choose one view's meta (they share fps/dur/size typically)
            any_view = next(iter(views_data.values()), None)
            if any_view:
                neo4j_upsert_clip(sess,
                                  key=base_key,
                                  path=any_view["path"],
                                  view=view_suffix,
                                  width=any_view["img_w"],
                                  height=any_view["img_h"],
                                  fps=any_view["fps"],
                                  dur=any_view["dur"])

            # Ingest embeddings (per-second + minute)
            for vview, data in views_data.items():
                                # per-second
                prev_fix = None  # (lat, lon, mph) of last accepted fix
                if spec.per_second:
                    for s_idx, (vec, locv_init, (meta_scalars, _src_ignored)) in enumerate(zip(
                        data["second_vecs"], data["second_locvecs"], data["second_loc_scalars"]
                    )):
                        emb_id = f"{key}|{vview}|sec|{s_idx}"
                        # time for this second
                        t_utc = data["dt0_utc"] + datetime.timedelta(seconds=s_idx)

                        # Query DB for LocationEvent (±1h) then PhoneLog; then check metadata as last resort
                        primary, fallback = find_best_locevent_then_phonelog(sess, t_utc, win_mins)
                        chosen, source = choose_location(primary, fallback, meta_scalars, prev_fix)

                        # Build loc_vec from chosen (overrides the initial zero/default)
                        lat = chosen.get("lat"); lon = chosen.get("lon"); mph = chosen.get("mph")
                        loc_vec, scalars = location_feature(lat, lon, mph, t_utc, include_time=spec.time_features)

                        # Upsert embedding FIRST (so the node exists)
                        neo4j_upsert_embed(
                            sess,
                            emb_id=emb_id, level="second", key=key, view=vview,
                            t0=int(s_idx), t1=int(min(s_idx+1, math.ceil(dur))),
                            vec=vec, loc_vec=loc_vec, loc_scalars=scalars, loc_source=source,
                            spec=spec, grids=grids
                        )

                        # Then attach NEAR to whichever source we actually used (if it has elem_id)
                        if source in ("LocationEvent","PhoneLog") and "elem_id" in chosen and chosen["elem_id"]:
                            sess.run(
                                NQ_ATTACH_NEAR_BY_ELEMID,
                                eid=emb_id, elem_id=chosen["elem_id"],
                                seconds=int(chosen.get("seconds", 0)),
                                source=source
                            )

                        # Enrich Frame with the chosen fix
                        neo4j_enrich_frame(sess, key=key, frame_idx=s_idx*int(round(fps)),
                                           lat=scalars.get("lat"), lon=scalars.get("lon"),
                                           mph=scalars.get("mph"))

                        # Maintain NEXT chain
                        if prev_id: neo4j_link_next(sess, prev_id, emb_id)
                        prev_id = emb_id

                        # Update prev_fix if we accepted a plausible fix
                        if scalars.get("lat") is not None and scalars.get("lon") is not None:
                            prev_fix = (scalars["lat"], scalars["lon"], scalars.get("mph"))


                # per-minute
                if spec.per_minute:
                    mvec = data["minute_vec"]
                    emb_id = element_id_safe(f"{base_key}|{vview}|minute|0")
                    neo4j_upsert_embed(
                        sess,
                        emb_id=emb_id,
                        level="minute",
                        key=base_key,
                        view=vview,
                        t0=0,
                        t1=int(math.ceil(data['dur'])),
                        vec=mvec,
                        spec=spec,
                        grids=grids
                    )
    driver.close()

# =========================
# CLI
# =========================
def main():
    parser = argparse.ArgumentParser(description="Build structure-preserving YOLO embeddings (per-second & per-minute) and ingest into Neo4j.")
    parser.add_argument("--bases", nargs="+", default=[
        "/mnt/8TBHDD/fileserver/dashcam/",
        "/mnt/8TB_2025/fileserver/dashcam/"
    ], help="One or more base directories to walk (expects YYYY/MM/DD subdirs)")
    parser.add_argument("--grid", default="16x9", help="Primary grid WxH (e.g., 16x9)")
    parser.add_argument("--pyramid", action="store_true", help="Use spatial pyramid (adds 8x4 and 32x18 to the base grid)")
    parser.add_argument("--no-conf", action="store_true", help="Ignore detection confidence (uniform weight)")
    parser.add_argument("--no-l2", action="store_true", help="Disable L2 normalization")
    parser.add_argument("--no-second", action="store_true", help="Skip per-second embeddings")
    parser.add_argument("--no-minute", action="store_true", help="Skip per-minute embeddings")
    parser.add_argument("--heatmap", action="store_true", help="Also render/update a minute-level heatmap PNG (like before)")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-pass", default="password")
    parser.add_argument("--classes", nargs="*", default=list(DEFAULT_VEHICLE_CLASSES),
                        help="Class names to keep (case-insensitive). Default = vehicle-ish")
    parser.add_argument("--dry-run", action="store_true", help="Parse & build but do not ingest into Neo4j")
    args = parser.parse_args()

    # build grid spec(s)
    gw, gh = map(int, args.grid.lower().split("x"))
    grids = [GridSpec(gw=gw, gh=gh)]
    if args.pyramid:
        grids = [GridSpec(8,4), GridSpec(gw,gh), GridSpec(32,18)]

    spec = EmbeddingSpec(
        grids=grids,
        include_conf=not args.no_conf,
        l2_normalize=not args.no_l2,
        concat_views=True,
        per_second=not args.no_second,
        per_minute=not args.no_minute,
    )

    keep_names = set([s.lower() for s in args.classes])
    id_map = DEFAULT_CLASS_ID_MAP

    base_dirs = [os.path.abspath(b) for b in args.bases]
    date_dirs = []
    for b in base_dirs:
        if not os.path.isdir(b):
            logging.warning(f"Base not found: {b}")
            continue
        date_dirs.extend(walk_date_dirs(b))
    logging.info(f"Found {len(date_dirs)} date folders")

    if args.dry_run:
        for dd in date_dirs:
            keys = find_file_keys(dd)
            logging.info(f"[dry-run] {dd}: {len(keys)} candidate clips")
        return

    for dd in date_dirs:
        logging.info(f"Processing date dir: {dd}")
        process_directory(
            dd, spec, grids, keep_names, id_map,
            include_heatmap_png=bool(args.heatmap),
            neo4j_uri=args.neo4j-uri,  # noqa
            neo4j_user=args.neo4j_user,
            neo4j_pwd=args.neo4j_pass,
            concat_views=True
        )

if __name__ == "__main__":
    main()
