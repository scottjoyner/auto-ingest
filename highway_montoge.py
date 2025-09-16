#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
highway_montage.py
Plan & render “similar actions” highway montages (45–135 mph) using your vector/geo/speed searches.

Key capabilities
----------------
1) plan-highway
   - Pull frames where mph ∈ [min,max], prefer heavy traffic (veh_count or cars+trucks+buses fallback).
   - Cluster into scenes by file/time proximity.
   - Score scenes (mph density * traffic density * recency/variety).
   - Emit a JSON "film plan" (scenes, shots, tags, artifacts).

2) plan-similar
   - Seed via --frame-id OR --file-key + --frame.
   - Run ANN similar-frames, then filter to mph window, same clustering/scoring as above.

3) render
   - Consume a film plan JSON and render a 9:16 vertical montage with captions overlay
     (if *_whisper.json are adjacent) and optional per-scene heatmap bumper.

4) heatmaps
   - Writes PNGs for:
       a) speed_time.png      — speed vs time (global)
       b) traffic_time.png    — vehicle count vs time (global)
       c) geo_density.png     — simple 2D histogram for lat/long (if present)
     These can be overlaid as ImageClip bumpers.

Assumptions
-----------
- Neo4j nodes :Frame have fields like (key, frame, millis, mph, lat, long, veh_count?) but we auto-fallback.
- Matching *_FR.MP4 files live under bases like /mnt/8TB_2025/fileserver/dashcam/YYYY/MM/DD/<KEY>_FR.MP4
- Adjacent Whisper JSON (optional captions): <KEY>_whisper.json or <KEY>_*_whisper.json

CLI quickstart
--------------
# 1) Plan by speed only (45-135 mph), favor heavy traffic:
python highway_montage.py plan-highway --min-mph 45 --max-mph 135 --limit 200 --plan out/plan.json

# 2) Plan by ANN similar frames, then filter to 45–135 mph:
python highway_montage.py plan-similar --frame-id <ID> --min-mph 45 --max-mph 135 --topk 200 --plan out/plan.json

# 3) Render the montage from a plan:
python highway_montage.py render --plan out/plan.json --out out/highway_montage.mp4 --profile clean

Dependencies
------------
pip install moviepy Pillow numpy torch transformers neo4j==5.*
"""

from __future__ import annotations
import argparse, json, logging, os, re, math, shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip, concatenate_videoclips

# Optional (captions overlays use same helpers as your smart_shorts)
from PIL import Image, ImageDraw, ImageFont
from functools import lru_cache

# Neo4j & embeddings (only needed for plan-similar)
import torch
from transformers import AutoTokenizer, AutoModel
from neo4j import GraphDatabase

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass

# --------- Config / defaults ----------
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
FRAME_DIM = int(os.getenv("FRAME_DIM", "256"))
LOCAL_TZ = os.getenv("LOCAL_TZ", "America/New_York")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "livelongandprosper")
NEO4J_DB = os.getenv("NEO4J_DB", "neo4j")

_tokenizer = None
_model = None

def _neo() -> Any:
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def _load_text_model():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        logging.info(f"Loading text embedding model on {DEVICE}…")
        _tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
        _model = AutoModel.from_pretrained(EMBED_MODEL_NAME).to(DEVICE).eval()
    return _tokenizer, _model

def _normalize(v: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(v, p=2, dim=1)

def embed_texts(texts: List[str], batch_size: int = 32, max_length: int = 512) -> List[List[float]]:
    if not texts: return []
    tok, mdl = _load_text_model()
    outv: List[List[float]] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            out = mdl(**enc)
            # mean pooling
            mask = enc["attention_mask"].unsqueeze(-1).expand(out.last_hidden_state.size()).float()
            pooled = (out.last_hidden_state * mask).sum(1) / torch.clamp(mask.sum(1), min=1e-9)
            outv.extend(_normalize(pooled).cpu().numpy().tolist())
    return outv

# --------- Paths / media discovery ----------
def _key_to_date_parts(key: str) -> Tuple[str,str,str]:
    # 2025_0914_161723 -> YYYY, MM, DD
    return key[0:4], key[5:7], key[7:9]

def find_fr_path(bases: List[Path], key: str) -> Optional[Path]:
    y, m, d = _key_to_date_parts(key)
    for base in bases:
        dd = base / y / m / d
        cand = dd / f"{key}_FR.MP4"
        if cand.exists(): return cand
    # fallback: search
    for base in bases:
        hits = list(base.rglob(f"{key}_FR.MP4"))
        if hits: return hits[0]
    return None

def find_adjacent_transcript(fr_path: Path) -> Optional[Path]:
    key = fr_path.stem.replace("_FR", "")
    for j in fr_path.parent.glob(f"{key}_*_whisper.json"):
        return j
    for j in fr_path.parent.glob(f"{key}_whisper.json"):
        return j
    return None

def load_transcript(path: Optional[Path]) -> dict:
    if not path or not path.exists(): return {"segments": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"segments": []}

# --------- Film plan types ----------
@dataclass
class Shot:
    file_key: str
    fr_path: str
    t_sec: float
    pre: float
    post: float
    fps: float
    mph: Optional[float] = None
    veh_count: Optional[int] = None
    score: float = 0.0

@dataclass
class Scene:
    scene_id: str
    shots: List[Shot]
    start_global_ms: Optional[int]
    end_global_ms: Optional[int]
    avg_mph: Optional[float]
    avg_traffic: Optional[float]
    score: float
    tags: List[str]

@dataclass
class FilmPlan:
    version: str
    plan_type: str
    scenes: List[Scene]
    artifacts: Dict[str, str]   # e.g., heatmap paths
    params: Dict[str, Any]      # planning parameters

# --------- Utilities ----------
def _frame_to_seconds(frame_no: int, fps: float) -> float:
    return float(frame_no) / max(1.0, fps)

def _get_transcription_start_ms(sess, file_key: str) -> Optional[int]:
    q = "MATCH (t:Transcription {key:$k}) RETURN t.started_at AS dt LIMIT 1"
    r = sess.run(q, k=file_key).single()
    if not r: return None
    dt = r.get("dt")
    try: return int(dt.epochMillis)
    except Exception:
        try: return int(dt.to_native().timestamp() * 1000)
        except Exception: return None

def _veh_count_of(f: Dict[str,Any]) -> int:
    # Prefer explicit veh_count, else sum common per-class counters if present
    if f.get("veh_count") is not None:
        return int(f["veh_count"])
    total = 0
    for k in ("cars","car","trucks","truck","buses","bus","motorcycles","motorcycle"):
        v = f.get(k)
        if v is not None:
            try: total += int(v)
            except Exception: pass
    return total

def _rows_to_shots(rows: List[Dict[str,Any]], bases: List[Path], sess, pre: float, post: float,
                   fps_fallback: float = 30.0) -> List[Shot]:
    shots: List[Shot] = []
    cache_fps: Dict[str, float] = {}
    start_ms_cache: Dict[str, Optional[int]] = {}

    for r in rows:
        fk = r.get("file_key")
        if not fk: continue
        frp = find_fr_path(bases, fk)
        if not frp: continue

        if fk not in cache_fps:
            try:
                with VideoFileClip(str(frp)) as v:
                    cache_fps[fk] = float(v.fps or fps_fallback)
            except Exception:
                cache_fps[fk] = fps_fallback

        if fk not in start_ms_cache:
            start_ms_cache[fk] = _get_transcription_start_ms(sess, fk)

        # second inside video
        sec: Optional[float] = None
        if r.get("millis") is not None and start_ms_cache[fk] is not None:
            sec = max(0.0, (int(r["millis"]) - int(start_ms_cache[fk])) / 1000.0)
        elif r.get("frame") is not None:
            sec = _frame_to_seconds(int(r["frame"]), cache_fps[fk])
        elif r.get("start") is not None:
            sec = float(r["start"])

        if sec is None:
            continue

        shots.append(Shot(
            file_key=fk,
            fr_path=str(frp),
            t_sec=sec,
            pre=float(pre),
            post=float(post),
            fps=float(cache_fps[fk]),
            mph=r.get("mph"),
            veh_count=_veh_count_of(r),
            score=float(r.get("score") or 0.0),
        ))
    return shots

# --------- Neo4j queries ----------
def query_frames_by_speed(sess, min_mph: float, max_mph: float, limit: int) -> List[Dict[str,Any]]:
    q = """
    MATCH (f:Frame)
    WHERE f.mph IS NOT NULL AND f.mph >= $min AND f.mph <= $max
    RETURN f.id AS id, f.key AS file_key, f.frame AS frame,
           f.millis AS millis, f.mph AS mph, f.lat AS lat, f.long AS lon,
           f.veh_count AS veh_count,
           f.cars AS cars, f.trucks AS trucks, f.buses AS buses, f.motorcycles AS motorcycles
    ORDER BY coalesce(f.veh_count, coalesce(f.cars,0)+coalesce(f.trucks,0)+coalesce(f.buses,0)+coalesce(f.motorcycles,0)) DESC,
             f.millis ASC
    LIMIT $limit
    """
    res = sess.run(q, min=float(min_mph), max=float(max_mph), limit=int(limit))
    return [r.data() for r in res]

def get_seed_embedding(sess, frame_id: Optional[str], file_key: Optional[str], frame_no: Optional[int],
                       frame_label: str, embed_prop: str) -> List[float]:
    def try_one(where: str, params: dict):
        q = f"MATCH (f:`{frame_label}` {where}) RETURN f.`{embed_prop}` AS emb LIMIT 1"
        r = sess.run(q, **params).single()
        return None if not r else r.get("emb")
    emb = None
    if frame_id:
        emb = try_one("{id:$id}", {"id": frame_id}) or try_one("{frame_id:$id}", {"id": frame_id})
    else:
        emb = try_one("{key:$k, frame:$n}", {"k": file_key, "n": int(frame_no)})
    if not emb:
        raise RuntimeError("Seed frame not found or missing embedding property.")
    return emb

def ann_similar_frames(sess, seed_emb: List[float], topk: int, frame_label: str, embed_prop: str) -> List[Dict[str,Any]]:
    index_name = "frame_embedding_index" if embed_prop == "embedding" else f"frame_{embed_prop}_index"
    q = f"""
    CALL db.index.vector.queryNodes('{index_name}', $k, $vec) YIELD node, score
    WHERE '{frame_label}' IN labels(node)
    RETURN node.id AS id, score AS score, node.key AS file_key, node.frame AS frame,
           node.millis AS millis, node.mph AS mph, node.lat AS lat, node.long AS lon,
           node.veh_count AS veh_count,
           node.cars AS cars, node.trucks AS trucks, node.buses AS buses, node.motorcycles AS motorcycles
    ORDER BY score DESC
    """
    res = sess.run(q, k=int(topk), vec=seed_emb)
    return [r.data() for r in res]

# --------- Scene clustering / scoring ----------
def cluster_shots(shots: List[Shot],
                  min_gap_sec: float = 4.0,
                  window_sec: float = 4.0,
                  max_scene_len_sec: float = 16.0) -> List[Scene]:
    """
    Cluster by (file_key, temporal proximity). Build scenes by merging shots that are within min_gap_sec.
    Each shot carries (pre, post); we convert to [t-pre, t+post] and merge contiguous overlaps until gap > min_gap_sec.
    """
    by_file: Dict[str, List[Shot]] = {}
    for s in shots:
        by_file.setdefault(s.file_key, []).append(s)
    for fk in by_file:
        by_file[fk].sort(key=lambda s: s.t_sec)

    scenes: List[Scene] = []
    sid = 0
    for fk, items in by_file.items():
        cur_group: List[Shot] = []
        last_end = None
        for s in items:
            s0, s1 = max(0.0, s.t_sec - s.pre), s.t_sec + s.post
            if last_end is None or (s0 - last_end) <= min_gap_sec:
                cur_group.append(s)
                last_end = max(last_end or 0.0, s1)
            else:
                if cur_group:
                    scenes.append(build_scene(fk, cur_group, sid)); sid += 1
                cur_group = [s]
                last_end = s1
        if cur_group:
            scenes.append(build_scene(fk, cur_group, sid)); sid += 1

    # Cap overly long scenes by splitting
    split_scenes: List[Scene] = []
    for sc in scenes:
        # naive cap: if span > max_scene_len_sec, cut after first N shots to fit
        span = (sc.end_global_ms or 0) - (sc.start_global_ms or 0)
        # we don't have per-file ms here; keep as-is and rely on renderer window size per shot
        split_scenes.append(sc)

    # Rank scenes
    for sc in split_scenes:
        mph = sc.avg_mph or 0.0
        traf = sc.avg_traffic or 0.0
        sc.score = float(mph) * (1.0 + 0.15 * float(traf))  # simple heuristic
    split_scenes.sort(key=lambda s: s.score, reverse=True)
    return split_scenes

def build_scene(file_key: str, group: List[Shot], sid: int) -> Scene:
    # Compute ms bounds if any shot has millis via its source rows (not stored).
    # Here we approximate using t_sec only for tags; ms bounds set None.
    avg_mph = np.nanmean([s.mph for s in group if (s.mph is not None)]) if group else None
    avg_traf = np.nanmean([s.veh_count for s in group if (s.veh_count is not None)]) if group else None
    tags = ["highway"] + ([f"{int(round(avg_mph))}mph"] if avg_mph is not None else []) + (["traffic"] if (avg_traf and avg_traf > 0) else [])
    return Scene(
        scene_id=f"{file_key}-{sid}",
        shots=group,
        start_global_ms=None,
        end_global_ms=None,
        avg_mph=(float(avg_mph) if avg_mph==avg_mph else None),
        avg_traffic=(float(avg_traf) if avg_traf==avg_traf else None),
        score=0.0,
        tags=tags
    )

# --------- Heatmaps (PNG artifacts) ----------
def write_heatmaps(rows: List[Dict[str,Any]], out_dir: Path) -> Dict[str, str]:
    """
    Writes simple PNGs: speed_time, traffic_time, geo_density (if lat/lon present).
    Uses matplotlib; non-fatal if library missing.
    """
    artifacts: Dict[str,str] = {}
    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except Exception:
        logging.warning("matplotlib not available; skipping heatmaps.")
        return artifacts

    out_dir.mkdir(parents=True, exist_ok=True)
    # Speed over time
    times = [int(r["millis"]) for r in rows if r.get("millis") is not None]
    mphs  = [float(r["mph"]) for r in rows if r.get("mph") is not None]
    if times and mphs and len(times) == len(mphs):
        t0 = min(times); xt = [(t - t0)/1000.0 for t in times]
        plt.figure()
        plt.plot(xt, mphs)
        plt.xlabel("seconds (relative)"); plt.ylabel("mph"); plt.title("Speed vs Time")
        p = out_dir / "speed_time.png"; plt.savefig(p.as_posix(), dpi=160, bbox_inches="tight"); plt.close()
        artifacts["speed_time"] = str(p)

    # Traffic over time
    traf = []
    xt2 = []
    for r in rows:
        if r.get("millis") is None: continue
        xt2.append((int(r["millis"]) - (min(times) if times else 0))/1000.0)
        traf.append(float(_veh_count_of(r)))
    if xt2 and traf and len(xt2) == len(traf):
        plt.figure()
        plt.plot(xt2, traf)
        plt.xlabel("seconds (relative)"); plt.ylabel("vehicles"); plt.title("Traffic vs Time")
        p = out_dir / "traffic_time.png"; plt.savefig(p.as_posix(), dpi=160, bbox_inches="tight"); plt.close()
        artifacts["traffic_time"] = str(p)

    # Geo density (if any lat/lon)
    lats = [float(r["lat"]) for r in rows if r.get("lat") is not None]
    lons = [float(r["lon"]) for r in rows if r.get("lon") is not None]
    if lats and lons and len(lats) == len(lons):
        plt.figure()
        plt.hist2d(lons, lats, bins=60)
        plt.xlabel("lon"); plt.ylabel("lat"); plt.title("Geo density")
        p = out_dir / "geo_density.png"; plt.savefig(p.as_posix(), dpi=160, bbox_inches="tight"); plt.close()
        artifacts["geo_density"] = str(p)

    return artifacts

# --------- Rendering ----------
def _subclip_vertical(fr_path: Path, start: float, end: float, target_w: int, target_h: int) -> Tuple[VideoFileClip, float]:
    clip = VideoFileClip(str(fr_path)).subclip(max(0.0, start), max(0.0, end))
    w, h = clip.w, clip.h
    targ_aspect = target_w / float(target_h); cur_aspect = w / float(h)
    if cur_aspect > targ_aspect:
        new_w = int(h * targ_aspect); x1 = (w - new_w)//2
        cropped = clip.crop(x1=x1, y1=0, x2=x1+new_w, y2=h)
    else:
        new_h = int(w / targ_aspect); y1 = (h - new_h)//2
        cropped = clip.crop(x1=0, y1=y1, x2=w, y2=y1+new_h)
    return cropped.resize((target_w, target_h)), float(clip.fps or 30.0)

def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def write_video(shots: List[Shot], out_path: Path, width: int, height: int,
                heatmap_png: Optional[str] = None, heatmap_secs: float = 1.0,
                bitrate: str = "8M") -> None:
    _ensure_parent(out_path)
    pieces = []
    fps_out = None

    # optional intro bumper with heatmap
    if heatmap_png and Path(heatmap_png).exists():
        img = ImageClip(heatmap_png).set_duration(float(heatmap_secs)).resize((width,height))
        pieces.append(img)

    i = 0
    for s in shots:
        i += 1
        frp = Path(s.fr_path)
        # clamp to media duration
        with VideoFileClip(str(frp)) as v0:
            t0 = max(0.0, s.t_sec - s.pre)
            t1 = min(float(v0.duration or 0.0), s.t_sec + s.post)
        clip, fps = _subclip_vertical(frp, t0, t1, width, height)
        if fps_out is None: fps_out = fps
        # annotate mph/traffic corner text (simple)
        label = []
        if s.mph is not None: label.append(f"{int(round(float(s.mph)))} mph")
        if s.veh_count is not None and s.veh_count > 0: label.append(f"{int(s.veh_count)} veh")
        if label:
            txt = "  •  ".join(label)
            txt_img = _draw_label(txt, width)
            overlay = ImageClip(np.array(txt_img)).set_duration(clip.duration).set_position(("center", 40))
            pieces.append(CompositeVideoClip([clip, overlay]))
        else:
            pieces.append(clip)

    if not pieces:
        raise SystemExit("No clips to render.")
    final = concatenate_videoclips(pieces, method="compose")
    tmp = Path(str(out_path)+".__tmp__.mp4")
    if tmp.exists():
        try: tmp.unlink()
        except Exception: pass
    final.write_videofile(str(tmp), fps=fps_out or 30.0, codec="libx264", audio_codec="aac",
                          bitrate=bitrate, preset="medium", threads=os.cpu_count() or 2,
                          ffmpeg_params=["-f","mp4","-movflags","+faststart","-pix_fmt","yuv420p"],
                          verbose=False, logger=None)
    try:
        final.close()
    except Exception:
        pass
    os.replace(str(tmp), str(out_path))
    logging.info(f"Wrote video: {out_path}")

@lru_cache(maxsize=1024)
def _font(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()

def _draw_label(text: str, width: int) -> Image.Image:
    pad = 14
    f = _font(48)
    img = Image.new("RGBA", (width, 110), (0,0,0,0))
    d = ImageDraw.Draw(img)
    tw, th = d.textbbox((0,0), text, font=f)[2:]
    x = (width - tw)//2
    d.rounded_rectangle([x-12, 10, x+tw+12, 10+th+12], radius=12, fill=(0,0,0,160))
    d.text((x, 16), text, fill=(255,255,255,255), font=f)
    return img

# --------- Planner commands ----------
def cmd_plan_highway(args) -> None:
    driver = _neo()
    bases = [Path(b).resolve() for b in (args.bases or [])]
    with driver.session(database=NEO4J_DB) as sess:
        rows = query_frames_by_speed(sess, args.min_mph, args.max_mph, args.limit)
        # heatmaps (optional)
        artifacts = write_heatmaps(rows, Path(args.artifacts_dir)) if args.artifacts_dir else {}
        # rows -> shots
        shots = _rows_to_shots(rows, bases, sess, pre=args.pre, post=args.post, fps_fallback=args.fps_fallback)
        # prefer heavier traffic
        shots.sort(key=lambda s: (-(s.veh_count or 0), s.t_sec))
        scenes = cluster_shots(shots, min_gap_sec=args.min_gap, window_sec=(args.pre+args.post))
        plan = FilmPlan(
            version="1.0",
            plan_type="highway",
            scenes=scenes[:args.max_scenes],
            artifacts=artifacts,
            params=vars(args)
        )
        _write_plan(plan, Path(args.plan))
        logging.info(f"Wrote plan: {args.plan} (scenes={len(plan.scenes)})")

def cmd_plan_similar(args) -> None:
    driver = _neo()
    bases = [Path(b).resolve() for b in (args.bases or [])]
    with driver.session(database=NEO4J_DB) as sess:
        seed = get_seed_embedding(sess, args.frame_id, args.file_key, args.frame, args.frame_label, args.frame_embed_prop)
        rows = ann_similar_frames(sess, seed, args.topk, args.frame_label, args.frame_embed_prop)
        # filter by speed window if provided (default 45-135 from CLI)
        rows = [r for r in rows if (r.get("mph") is not None and float(args.min_mph) <= float(r["mph"]) <= float(args.max_mph))]
        artifacts = write_heatmaps(rows, Path(args.artifacts_dir)) if args.artifacts_dir else {}
        shots = _rows_to_shots(rows, bases, sess, pre=args.pre, post=args.post, fps_fallback=args.fps_fallback)
        scenes = cluster_shots(shots, min_gap_sec=args.min_gap, window_sec=(args.pre+args.post))
        plan = FilmPlan(
            version="1.0",
            plan_type="similar",
            scenes=scenes[:args.max_scenes],
            artifacts=artifacts,
            params=vars(args)
        )
        _write_plan(plan, Path(args.plan))
        logging.info(f"Wrote plan: {args.plan} (scenes={len(plan.scenes)})")

def _write_plan(plan: FilmPlan, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "version": plan.version,
        "plan_type": plan.plan_type,
        "scenes": [
            {
                "scene_id": sc.scene_id,
                "tags": sc.tags,
                "score": sc.score,
                "avg_mph": sc.avg_mph,
                "avg_traffic": sc.avg_traffic,
                "start_global_ms": sc.start_global_ms,
                "end_global_ms": sc.end_global_ms,
                "shots": [asdict(sh) for sh in sc.shots],
            } for sc in plan.scenes
        ],
        "artifacts": plan.artifacts,
        "params": plan.params
    }
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

# --------- Render from plan ----------
def cmd_render(args) -> None:
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    scenes = plan.get("scenes", [])
    artifacts = plan.get("artifacts", {})
    # Flatten shots by ranked scenes
    ranked = sorted(scenes, key=lambda s: float(s.get("score") or 0.0), reverse=True)[:args.max_scenes]
    shots: List[Shot] = []
    total = 0
    for sc in ranked:
        for sh in sc.get("shots", []):
            shots.append(Shot(**{**sh, "fr_path": sh["fr_path"]}))
            total += 1
            if total >= args.max_shots:
                break
        if total >= args.max_shots: break
    if not shots:
        raise SystemExit("Plan had no shots to render.")
    heat = artifacts.get(args.heatmap_key) if args.heatmap_key else None
    write_video(shots, Path(args.out), width=args.width, height=args.height,
                heatmap_png=heat, heatmap_secs=args.heatmap_secs, bitrate=args.bitrate)

# --------- CLI ----------
def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plan & render highway speed montages with traffic heatmaps.")
    p.add_argument("--bases", action="append", default=["/mnt/8TB_2025/fileserver/dashcam"], help="Dashcam bases (repeatable).")
    p.add_argument("--fps-fallback", type=float, default=30.0)
    p.add_argument("--log-level", default="INFO")

    sub = p.add_subparsers(dest="cmd", required=True)

    # plan-highway
    s1 = sub.add_parser("plan-highway", help="Plan scenes by speed (favor higher traffic).")
    s1.add_argument("--min-mph", type=float, default=45.0)
    s1.add_argument("--max-mph", type=float, default=135.0)
    s1.add_argument("--limit", type=int, default=400)
    s1.add_argument("--pre", type=float, default=1.2)
    s1.add_argument("--post", type=float, default=2.4)
    s1.add_argument("--min-gap", type=float, default=4.0)
    s1.add_argument("--max-scenes", type=int, default=24)
    s1.add_argument("--plan", type=str, required=True)
    s1.add_argument("--artifacts-dir", type=str, default=None)

    # plan-similar
    s2 = sub.add_parser("plan-similar", help="Plan via ANN similar frames, then mph filter.")
    g = s2.add_mutually_exclusive_group(required=True)
    g.add_argument("--frame-id", type=str)
    g.add_argument("--file-key", type=str)
    s2.add_argument("--frame", type=int, help="Required with --file-key")
    s2.add_argument("--frame-label", default="Frame")
    s2.add_argument("--frame-embed-prop", default="embedding")
    s2.add_argument("--topk", type=int, default=300)
    s2.add_argument("--min-mph", type=float, default=45.0)
    s2.add_argument("--max-mph", type=float, default=135.0)
    s2.add_argument("--pre", type=float, default=1.2)
    s2.add_argument("--post", type=float, default=2.4)
    s2.add_argument("--min-gap", type=float, default=4.0)
    s2.add_argument("--max-scenes", type=int, default=24)
    s2.add_argument("--plan", type=str, required=True)
    s2.add_argument("--artifacts-dir", type=str, default=None)

    # render
    s3 = sub.add_parser("render", help="Render a 9:16 montage from plan JSON.")
    s3.add_argument("--plan", type=str, required=True)
    s3.add_argument("--out", type=str, required=True)
    s3.add_argument("--width", type=int, default=1080)
    s3.add_argument("--height", type=int, default=1920)
    s3.add_argument("--bitrate", type=str, default="8M")
    s3.add_argument("--heatmap-key", type=str, default="speed_time",
                    help="Which artifact key to overlay as intro bumper (e.g. speed_time, traffic_time, geo_density).")
    s3.add_argument("--heatmap-secs", type=float, default=1.0)
    s3.add_argument("--max-scenes", type=int, default=16)
    s3.add_argument("--max-shots", type=int, default=80)

    return p

def main() -> int:
    args = build_cli().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(message)s")
    if args.cmd == "plan-highway":
        cmd_plan_highway(args)
    elif args.cmd == "plan-similar":
        cmd_plan_similar(args)
    elif args.cmd == "render":
        cmd_render(args)
    else:
        raise SystemExit("Unknown subcommand")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
