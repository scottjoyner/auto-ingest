#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
smart_shorts.py
Build a single 9:16 short from *selected* micro-scenes chosen by vector/geo/speed queries.

Sources:
- Combines your 'shorts_builder.py' (captions, profiles, crop, safe-write) and
  'vector_search.py' (text/ANN/geo) into one tool.
- Writes ONE final compilation video built from handpicked windows (clips)
  instead of captioning the entire source video.

Key ideas:
- Use Neo4j to pick anchors (frames or text segments) by semantic/visual/geo/speed logic.
- For each anchor -> compute timestamp inside file_key's *_FR.MP4*
- Subclip [t - pre_sec, t + post_sec], crop to 9:16, overlay captions only for that window.
- Concatenate clips, write final mp4.

Requirements:
  pip install moviepy Pillow numpy torch transformers neo4j==5.*  # (plus whisper if you transcribe)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import re
import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import (
    CompositeVideoClip,
    ImageClip,
    VideoFileClip,
    concatenate_videoclips,
)

# ---------- Neo4j + embedding ----------
import torch
from transformers import AutoTokenizer, AutoModel
from neo4j import GraphDatabase

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
FRAME_DIM = int(os.getenv("FRAME_DIM", "256"))  # YOLO/frame vector dim
LOCAL_TZ = os.getenv("LOCAL_TZ", "America/New_York")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "livelongandprosper")
NEO4J_DB = os.getenv("NEO4J_DB", "neo4j")

_tokenizer = None
_model = None

def load_text_model():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        logging.info(f"Loading text embedding model on {DEVICE} …")
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

def embed_texts(texts: List[str], batch_size: int = 32, max_length: int = 512) -> List[List[float]]:
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
            vecs.extend(_normalize(pooled).cpu().numpy().tolist())
    return vecs

def neo4j_driver(uri: str, user: str, password: str):
    return GraphDatabase.driver(uri, auth=(user, password))

def _create_vector_index(sess, label: str, prop: str, name: str, dim: int):
    q = f"""
    CREATE VECTOR INDEX {name} IF NOT EXISTS
    FOR (n:{label}) ON (n.{prop})
    OPTIONS {{
      indexConfig: {{
        `vector.dimensions`: {dim},
        `vector.similarity_function`: 'cosine'
      }}
    }}"""
    sess.run(q)

SCHEMA_QUERIES = [
    "CREATE CONSTRAINT transcription_id IF NOT EXISTS FOR (t:Transcription) REQUIRE t.id IS UNIQUE",
    "CREATE CONSTRAINT segment_id IF NOT EXISTS FOR (s:Segment) REQUIRE s.id IS UNIQUE",
    "CREATE CONSTRAINT utterance_id IF NOT EXISTS FOR (u:Utterance) REQUIRE u.id IS UNIQUE",
    "CREATE CONSTRAINT frame_id IF NOT EXISTS FOR (f:Frame) REQUIRE f.id IS UNIQUE",
    "CREATE INDEX transcription_key IF NOT EXISTS FOR (t:Transcription) ON (t.key)",
    "CREATE INDEX frame_key IF NOT EXISTS FOR (f:Frame) ON (f.key)",
    "CREATE INDEX frame_key_frame IF NOT EXISTS FOR (f:Frame) ON (f.key, f.frame)"
]

def ensure_indexes(driver, frame_label: str, embed_prop: str):
    with driver.session(database=NEO4J_DB) as sess:
        for q in SCHEMA_QUERIES:
            sess.run(q)
        _create_vector_index(sess, "Segment", "embedding", "segment_embedding_index", EMBED_DIM)
        _create_vector_index(sess, "Transcription", "embedding", "transcription_embedding_index", EMBED_DIM)
        _create_vector_index(sess, "Utterance", "embedding", "utterance_embedding_index", EMBED_DIM)
        name = "frame_embedding_index" if embed_prop == "embedding" else f"frame_{embed_prop}_index"
        _create_vector_index(sess, frame_label, embed_prop, name, FRAME_DIM)

# ---------- Profiles (same defaults as your shorts_builder) ----------
DEFAULT_PROFILES: Dict[str, Dict[str, Any]] = {
    "clean": {
        "font_path": "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "font_size": 64, "y_pos_ratio": 0.82, "line_height_px": 90, "stroke_px": 4,
        "colors": {"text": (255,255,255,255), "stroke": (0,0,0,255), "highlight_text": (0,0,0,255)},
        "max_width_px": 1000, "wordgrid": False, "karaoke": True, "show_speaker_tag": True,
    },
    "karaoke": {
        "font_path": "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "font_size": 72, "y_pos_ratio": 0.80, "line_height_px": 100, "stroke_px": 6,
        "colors": {"text": (255,255,255,255), "stroke": (0,0,0,255), "highlight_text": (0,0,0,255)},
        "max_width_px": 1080 - 160, "wordgrid": False, "karaoke": True, "show_speaker_tag": True,
    },
    "wordgrid": {
        "font_path": "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "font_size": 60, "y_pos_ratio": 0.75, "line_height_px": 90, "stroke_px": 2,
        "colors": {"text": (255,255,255,255), "stroke": (0,0,0,255), "highlight_text": (255,255,255,255)},
        "max_width_px": 980, "wordgrid": True, "karaoke": False, "show_speaker_tag": True,
    },
}

def load_profiles_from_json(path: Path) -> Dict[str, Dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    for prof in data.values():
        if "colors" in prof:
            prof["colors"] = {k: tuple(v) for k, v in prof["colors"].items()}
    return data

# ---------- Speaker mapping (kept minimal) ----------
@dataclass
class SpeakerMapEntry:
    local_label: str
    global_id: Optional[str] = None
    name: Optional[str] = None

def load_speaker_map(path: Optional[Path]) -> Dict[str, SpeakerMapEntry]:
    mapping: Dict[str, SpeakerMapEntry] = {}
    if not path or not path.exists():
        return mapping
    data = json.loads(path.read_text(encoding="utf-8"))
    for k, v in data.items():
        mapping[k] = SpeakerMapEntry(local_label=k, global_id=v.get("global_id"), name=v.get("name"))
    return mapping

# ---------- Colors / fonts / rendering ----------
def _hash_color(key: str) -> Tuple[int,int,int,int]:
    h = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16)
    hue = (h % 360) / 360.0
    s, v = 0.65, 0.95
    i = int(hue * 6); f = hue * 6 - i
    p = int(255 * v * (1 - s)); q = int(255 * v * (1 - f * s)); t = int(255 * v * (1 - (1 - f) * s)); V = int(255 * v); i %= 6
    if   i==0: r,g,b = V, t, p
    elif i==1: r,g,b = q, V, p
    elif i==2: r,g,b = p, V, t
    elif i==3: r,g,b = p, q, V
    elif i==4: r,g,b = t, p, V
    else:      r,g,b = V, p, q
    return (r,g,b,230)

@lru_cache(maxsize=4096)
def _load_font(font_path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(font_path, size)
    except Exception:
        return ImageFont.load_default()

def _normalize_colors_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        out[k] = tuple(v) if isinstance(v, list) else v
    return out

def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> List[str]:
    words = text.split()
    lines: List[str] = []; cur: List[str] = []
    for w in words:
        test = " ".join(cur + [w])
        wbox = draw.textbbox((0, 0), test, font=font)
        if (wbox[2] - wbox[0]) <= max_width or not cur:
            cur.append(w)
        else:
            lines.append(" ".join(cur))
            cur = [w]
    if cur: lines.append(" ".join(cur))
    return lines

def _speaker_color(local_label: Optional[str]) -> Tuple[int,int,int,int]:
    if not local_label:
        return (255, 235, 59, 230)
    return _hash_color(local_label)

def _render_tag(text: str, font: ImageFont.ImageFont, color_rgba: Tuple[int,int,int,int]) -> Image.Image:
    dimg = Image.new("RGBA", (10,10), (0,0,0,0)); d = ImageDraw.Draw(dimg)
    bbox = d.textbbox((0,0), text, font=font); w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]; pad = 10
    img = Image.new("RGBA", (w+2*pad, h+2*pad), (0,0,0,0)); di = ImageDraw.Draw(img); r = 12
    di.rounded_rectangle([0,0,img.width,img.height], radius=r, fill=color_rgba)
    di.text((pad,pad), text, font=font, fill=(0,0,0,255)); return img

def _draw_sentence(sentence: str, width: int, profile: Dict, speaker_label: Optional[str]) -> Image.Image:
    font = _load_font(profile["font_path"], profile["font_size"])
    stroke_px = int(profile["stroke_px"]); colors = _normalize_colors_dict(profile["colors"])
    line_h = int(profile["line_height_px"]); max_w = min(int(profile["max_width_px"]), width - 80)
    img = Image.new("RGBA", (width, line_h*3 + 60), (0,0,0,0)); d = ImageDraw.Draw(img)
    lines = _wrap_text(d, sentence.strip(), font, max_w); y = 0
    if profile.get("show_speaker_tag") and speaker_label:
        tag_font = _load_font(profile["font_path"], int(profile["font_size"]*0.55))
        tag_img = _render_tag(f"[{speaker_label}]", tag_font, _speaker_color(speaker_label))
        img.paste(tag_img, ((width - tag_img.width)//2, y), tag_img); y += tag_img.height + 8
    for line in lines:
        bbox = d.textbbox((0,0), line, font=font, stroke_width=stroke_px)
        tw = bbox[2] - bbox[0]; x = (width - tw)//2
        d.text((x,y), line, font=font, fill=colors.get("text",(255,255,255,255)),
               stroke_width=stroke_px, stroke_fill=colors.get("stroke",(0,0,0,255)))
        y += line_h
    return img.crop((0,0,width, y or line_h))

def _draw_sentence_highlight(sentence: str, word: str, width: int, profile: Dict, speaker_label: Optional[str]) -> Image.Image:
    font = _load_font(profile["font_path"], profile["font_size"])
    stroke_px = int(profile["stroke_px"]); colors = _normalize_colors_dict(profile["colors"])
    line_h = int(profile["line_height_px"]); max_w = min(int(profile["max_width_px"]), width - 80)
    hi_bg = _speaker_color(speaker_label); img = Image.new("RGBA",(width,line_h*3),(0,0,0,0)); d = ImageDraw.Draw(img)
    lines = _wrap_text(d, sentence.strip(), font, max_w); y = 0
    for line in lines:
        x = (width - d.textbbox((0,0), line, font=font, stroke_width=stroke_px)[2]) // 2
        for token in line.split():
            wbox = d.textbbox((0,0), token, font=font, stroke_width=stroke_px)
            w_w, w_h = wbox[2]-wbox[0], wbox[3]-wbox[1]
            if token.strip() == word.strip():
                pad = 6
                d.rectangle([x-pad,y-pad,x+w_w+pad,y+w_h+pad], fill=hi_bg)
                d.text((x,y), token, font=font, fill=colors.get("highlight_text",(0,0,0,255)))
            else:
                d.text((x,y), token, font=font, fill=colors.get("text",(255,255,255,255)),
                       stroke_width=stroke_px, stroke_fill=colors.get("stroke",(0,0,0,255)))
            x += w_w + font.size//3
        y += line_h
    return img.crop((0,0,width, y or line_h))

@lru_cache(maxsize=8192)
def _render_cached(key: str) -> np.ndarray:
    spec = json.loads(key)
    kind = spec["kind"]; prof = spec["profile"]
    if kind == "sentence":
        img = _draw_sentence(spec["sentence"], spec["width"], prof, spec.get("speaker_label"))
    elif kind == "karaoke":
        img = _draw_sentence_highlight(spec["sentence"], spec["word"], spec["width"], prof, spec.get("speaker_label"))
    else:
        raise ValueError("Unknown render kind")
    return np.array(img)

def _cap_key(kind: str, **kwargs) -> str:
    return json.dumps({"kind": kind, **kwargs}, sort_keys=True)

# ---------- IO helpers ----------
def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def _temp_path_with_same_ext(final_path: Path, token: str = ".__tmp__") -> Path:
    root, ext = os.path.splitext(str(final_path)); ext = ext or ".mp4"
    return Path(f"{root}{token}{ext}")

def _atomic_replace(src: Path, dst: Path) -> None:
    _ensure_parent_dir(dst)
    try:
        os.replace(str(src), str(dst))
    except Exception:
        tmp_dst = Path(str(dst) + ".__swap__")
        shutil.copy2(str(src), str(tmp_dst))
        os.replace(str(tmp_dst), str(dst))
        try: os.remove(str(src))
        except FileNotFoundError: pass

def _write_videofile_safely(
    clip, final_path: Path, *, fps: float, bitrate: str = "8M",
    codec: str = "libx264", audio_codec: Optional[str] = "aac",
    threads: Optional[int] = None, extra_ffmpeg_params: Optional[List[str]] = None,
) -> None:
    _ensure_parent_dir(final_path)
    tmp_path = _temp_path_with_same_ext(final_path)
    if tmp_path.exists():
        try: tmp_path.unlink()
        except FileNotFoundError: pass
    params = extra_ffmpeg_params or []
    merged = ["-f","mp4","-movflags","+faststart","-pix_fmt","yuv420p"] + params
    clip.write_videofile(
        str(tmp_path), fps=fps, codec=codec, audio_codec=audio_codec,
        threads=threads or (os.cpu_count() or 2), preset="medium",
        bitrate=bitrate, ffmpeg_params=merged, verbose=False, logger=None,
    )
    _atomic_replace(tmp_path, final_path)

# ---------- Dashcam path + transcript discovery ----------
DATE_DIR_RE = re.compile(r"^\d{4}/\d{2}/\d{2}$")

def _key_to_date_parts(key: str) -> Tuple[str,str,str]:
    # keys like 2024_0209_163231 -> year=2024, month=02, day=09
    year = key[0:4]; month = key[5:7]; day = key[7:9]
    return year, month, day

def _resolve_fr_path(bases: List[Path], key: str) -> Optional[Path]:
    y, m, d = _key_to_date_parts(key)
    for base in bases:
        dd = base / y / m / d
        cand = dd / f"{key}_FR.MP4"
        if cand.exists(): return cand
    # fallback: glob search (slower)
    for base in bases:
        hits = list(base.rglob(f"{key}_FR.MP4"))
        if hits: return hits[0]
    return None

def _find_transcript_json(fr_path: Path) -> Optional[Path]:
    # any *_whisper.json adjacent to *_FR.MP4
    key = fr_path.stem.replace("_FR", "")
    for j in fr_path.parent.glob(f"{key}_*_whisper.json"):
        return j
    for j in fr_path.parent.glob(f"{key}_whisper.json"):
        return j
    return None

def _load_transcript(json_path: Optional[Path]) -> dict:
    if not json_path or not json_path.exists():
        return {"segments": []}
    return json.loads(json_path.read_text(encoding="utf-8"))

# ---------- Caption scheduling for a time window ----------
def _clips_for_window(transcript: dict, window_start: float, window_end: float,
                      W: int, H: int, profile: Dict, y_ratio: float) -> List[ImageClip]:
    base_y = int(H * y_ratio)
    out: List[ImageClip] = []
    for seg in transcript.get("segments", []):
        s_start, s_end = seg.get("start"), seg.get("end")
        if s_start is None or s_end is None: continue
        # include if any overlap with [window_start, window_end]
        if s_end < window_start or s_start > window_end: continue
        # Clamp to window and shift to 0-based
        s0 = max(s_start, window_start) - window_start
        s1 = min(s_end, window_end) - window_start
        if s1 <= s0: continue
        sentence = (seg.get("text") or "").strip()
        seg_speaker = seg.get("speaker")
        if not sentence:
            continue

        # Base sentence panel
        base_key = _cap_key("sentence", sentence=sentence, width=W, profile=profile, speaker_label=seg_speaker)
        base_arr = _render_cached(base_key)
        out.append(ImageClip(base_arr).set_duration(s1 - s0).set_start(s0).set_position(("center", base_y)))

        # Karaoke per-word highlights if enabled
        if profile.get("karaoke", True):
            for w in seg.get("words", []):
                token = (w.get("word") or "").strip()
                st, en = w.get("start"), w.get("end")
                if not token or st is None or en is None:
                    continue
                # skip words outside window
                if en < window_start or st > window_end:
                    continue
                ws = max(st, window_start) - window_start
                we = min(en, window_end) - window_start
                if we <= ws: continue
                kara_key = _cap_key("karaoke", sentence=sentence, word=token, width=W, profile=profile, speaker_label=w.get("speaker", seg_speaker))
                kara_arr = _render_cached(kara_key)
                out.append(ImageClip(kara_arr).set_duration(we - ws).set_start(ws).set_position(("center", base_y)))
    return out

# ---------- Cropping to 9:16 and subclipping ----------
def _subclip_vertical(fr_path: Path, start: float, end: float, target_w: int, target_h: int) -> Tuple[VideoFileClip, float]:
    clip = VideoFileClip(str(fr_path)).subclip(max(0.0, start), end)
    w, h = clip.w, clip.h
    targ_aspect = target_w / float(target_h); cur_aspect = w / float(h)
    if cur_aspect > targ_aspect:
        new_w = int(h * targ_aspect); x1 = (w - new_w)//2
        cropped = clip.crop(x1=x1, y1=0, x2=x1+new_w, y2=h)
    else:
        new_h = int(w / targ_aspect); y1 = (h - new_h)//2
        cropped = clip.crop(x1=0, y1=y1, x2=w, y2=y1+new_h)
    final = cropped.resize((target_w, target_h))
    fps = float(clip.fps or 30.0)
    return final, fps

# ---------- Search modes (text / similar-frames / geo / speed) ----------
def search_text(driver, query: str, target: str, top_k: int, win_minutes: int) -> List[Dict[str,Any]]:
    qvec = embed_texts([query])[0]
    if target == "utterance":
        index_name = "utterance_embedding_index"
        cypher = f"""
        CALL db.index.vector.queryNodes('{index_name}', $k, $qvec) YIELD node, score
        WHERE 'Utterance' IN labels(node)
        MATCH (t:Transcription)-[:HAS_UTTERANCE]->(u:Utterance) WHERE u = node
        WITH node, score, t, u
        RETURN node.id AS id, score, u.start AS start, u.end AS end,
               t.id AS transcription_id, t.key AS file_key, t.started_at AS started_at
        ORDER BY score DESC
        """
    elif target == "segment":
        index_name = "segment_embedding_index"
        cypher = f"""
        CALL db.index.vector.queryNodes('{index_name}', $k, $qvec) YIELD node, score
        WHERE 'Segment' IN labels(node)
        MATCH (t:Transcription)-[:HAS_SEGMENT]->(s:Segment) WHERE s = node
        RETURN node.id AS id, score, s.start AS start, s.end AS end,
               t.id AS transcription_id, t.key AS file_key, t.started_at AS started_at
        ORDER BY score DESC
        """
    else:
        index_name = "transcription_embedding_index"
        cypher = f"""
        CALL db.index.vector.queryNodes('{index_name}', $k, $qvec) YIELD node, score
        WHERE 'Transcription' IN labels(node)
        RETURN node.id AS id, score, null AS start, null AS end,
               node.id AS transcription_id, node.key AS file_key, node.started_at AS started_at
        ORDER BY score DESC
        """
    with driver.session(database=NEO4J_DB) as sess:
        res = sess.run(cypher, k=top_k, qvec=qvec)
        return [r.data() for r in res]

def get_frame_seed(driver, frame_id: Optional[str], file_key: Optional[str], frame_no: Optional[int],
                   frame_label: str, embed_prop: str) -> Tuple[List[float], Dict[str,Any]]:
    # Try to fetch a frame's vector by ID or (key, frame)
    with driver.session(database=NEO4J_DB) as sess:
        def by(where: str, params: dict):
            q = f"MATCH (f:`{frame_label}` {where}) RETURN f.id AS id, f.key AS key, f.frame AS frame, f.`{embed_prop}` AS emb LIMIT 1"
            return sess.run(q, **params).single()
        rec = None
        if frame_id:
            rec = by("{id:$id}", {"id": frame_id}) or sess.run(
                f"MATCH (f:`{frame_label}` {{frame_id:$id}}) RETURN f.id AS id, f.key AS key, f.frame AS frame, f.`{embed_prop}` AS emb LIMIT 1",
                id=frame_id).single()
        else:
            rec = by("{key:$key, frame:$frame}", {"key": file_key, "frame": int(frame_no)})
        if not rec or rec.get("emb") is None:
            raise RuntimeError("Seed frame not found or has no embedding on requested property.")
        return rec["emb"], {"id": rec["id"], "key": rec["key"], "frame": rec["frame"]}

def similar_frames(driver, seed_emb: List[float], top_k: int, frame_label: str, embed_prop: str) -> List[Dict[str,Any]]:
    index_name = "frame_embedding_index" if embed_prop == "embedding" else f"frame_{embed_prop}_index"
    cypher = f"""
    CALL db.index.vector.queryNodes('{index_name}', $k, $qvec) YIELD node, score
    WHERE '{frame_label}' IN labels(node)
    RETURN node.id AS id, score AS score, node.key AS file_key, node.frame AS frame,
           node.millis AS millis, node.mph AS mph
    ORDER BY score DESC
    """
    with driver.session(database=NEO4J_DB) as sess:
        res = sess.run(cypher, k=top_k, qvec=seed_emb)
        return [r.data() for r in res]

def geo_frames(driver, lat: float, lon: float, radius_m: float,
               start_ms: Optional[int], end_ms: Optional[int],
               min_mph: Optional[float], max_mph: Optional[float], limit: int) -> List[Dict[str,Any]]:
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
      AND ($start_ms IS NULL OR (f.millis IS NOT NULL AND f.millis >= $start_ms))
      AND ($end_ms   IS NULL OR (f.millis IS NOT NULL AND f.millis <= $end_ms))
      AND ($min_mph IS NULL OR (f.mph IS NOT NULL AND f.mph >= $min_mph))
      AND ($max_mph IS NULL OR (f.mph IS NOT NULL AND f.mph <= $max_mph))
    RETURN f.id AS id, f.key AS file_key, f.frame AS frame, f.millis AS millis, f.mph AS mph, dist AS meters
    ORDER BY dist ASC
    LIMIT $limit
    """
    with driver.session(database=NEO4J_DB) as sess:
        res = sess.run(cypher, lat=float(lat), lon=float(lon), radius=float(radius_m),
                       start_ms=start_ms, end_ms=end_ms, min_mph=min_mph, max_mph=max_mph, limit=int(limit))
        return [r.data() for r in res]

def transcription_start_ms(driver, file_key: str) -> Optional[int]:
    cypher = "MATCH (t:Transcription {key:$k}) RETURN t.started_at AS started_at LIMIT 1"
    with driver.session(database=NEO4J_DB) as sess:
        r = sess.run(cypher, k=file_key).single()
        if not r: return None
        dt = r.get("started_at")
        if dt is None: return None
        # Neo4j returns DateTime; the python driver exposes .to_native() or .epochMillis; handle both:
        try:
            return int(dt.epochMillis)  # neo4j DateTime
        except Exception:
            try:
                native = dt.to_native()
                return int(native.timestamp()*1000)
            except Exception:
                return None

# ---------- Anchor → seconds inside video ----------
def _frame_to_seconds(frame_no: int, fps: float) -> float:
    return float(frame_no) / max(1.0, fps)

def _anchor_second(row: Dict[str,Any], start_ms: Optional[int], fps_fallback: float) -> Optional[float]:
    """
    Prefer millis relative to epoch + transcription.start_ms.
    Fallback to frame/fps mapping if millis or start_ms missing.
    """
    if row.get("millis") is not None and start_ms is not None:
        return max(0.0, (int(row["millis"]) - int(start_ms)) / 1000.0)
    if row.get("start") is not None:  # segment/utterance
        return float(row["start"])
    if row.get("frame") is not None:
        return _frame_to_seconds(int(row["frame"]), fps_fallback)
    return None

# ---------- Shot selection / dedupe ----------
def build_shot_list(rows: List[Dict[str,Any]], driver, bases: List[Path],
                    pre_sec: float, post_sec: float, min_gap_sec: float,
                    fps_fallback: float = 30.0, max_clips: int = 12) -> List[Dict[str,Any]]:
    out: List[Dict[str,Any]] = []
    grouped: Dict[str, List[Dict[str,Any]]] = {}
    # group by file_key to dedupe within same video
    for r in rows:
        fk = r.get("file_key")
        if not fk: continue
        grouped.setdefault(fk, []).append(r)

    for file_key, items in grouped.items():
        start_ms = transcription_start_ms(driver, file_key)
        # sort by score desc if exists, else by millis/frame asc
        def keyf(x):
            return (-float(x.get("score", 0.0)), x.get("millis") or 0, x.get("frame") or 0)
        items.sort(key=keyf)
        chosen_secs: List[float] = []
        for r in items:
            fr_path = _resolve_fr_path(bases, file_key)
            if not fr_path: continue
            # get fps from file once
            try:
                with VideoFileClip(str(fr_path)) as v:
                    fps = float(v.fps or fps_fallback)
                    # derive second inside video
                    sec = _anchor_second(r, start_ms, fps)
                    if sec is None: continue
                    # dedupe: keep if far enough from existing picks
                    if any(abs(sec - s) < min_gap_sec for s in chosen_secs):
                        continue
                    chosen_secs.append(sec)
                    out.append({
                        "file_key": file_key,
                        "fr_path": str(fr_path),
                        "t_sec": sec,
                        "pre": pre_sec,
                        "post": post_sec,
                        "fps": fps,
                        "score": float(r.get("score", 0.0)),
                        "mph": r.get("mph"),
                    })
                    if len(chosen_secs) >= max_clips:
                        break
            except Exception:
                continue
    # Global cap:
    return out[:max_clips]

# ---------- Build compilation ----------
def build_compilation(
    shots: List[Dict[str,Any]],
    *, width: int, height: int, profile: Dict, y_ratio: float,
    out_path: Path, bitrate: str = "8M"
) -> None:
    pieces = []
    fps_out = None

    for i, sh in enumerate(shots, 1):
        frp = Path(sh["fr_path"]); t0 = max(0.0, sh["t_sec"] - float(sh["pre"])); t1 = sh["t_sec"] + float(sh["post"])
        # Open once to clamp end:
        with VideoFileClip(str(frp)) as v:
            t1 = min(float(v.duration or 0.0), t1)

        clip, fps = _subclip_vertical(frp, t0, t1, width, height)
        if fps_out is None: fps_out = fps

        # Find transcript for captions (optional)
        transcript = _load_transcript(_find_transcript_json(frp))
        # Overlay only words/segments in [t0, t1]
        caption_clips = _clips_for_window(transcript, t0, t1, width, height, profile, y_ratio)
        if caption_clips:
            comp = CompositeVideoClip([clip] + caption_clips)
        else:
            comp = clip

        # Add a tiny fade between clips for cohesion
        comp = comp.crossfadein(0.08) if i > 1 else comp
        pieces.append(comp)

    if not pieces:
        raise SystemExit("No clips to compile (shot list empty).")

    final = concatenate_videoclips(pieces, method="compose")
    _write_videofile_safely(final, out_path, fps=fps_out or 30.0, bitrate=bitrate)
    try:
        final.close()
    except Exception:
        pass
    logging.info(f"Wrote compilation: {out_path}")

# ---------- CLI ----------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build one 9:16 short from vector/geo/speed-selected scenes.")
    # Data roots / output
    p.add_argument("--base", dest="bases", action="append", default=["/mnt/8TB_2025/fileserver/dashcam"], help="Dashcam root (repeatable).")
    p.add_argument("--out", type=Path, default=Path("./short_compilation.mp4"))
    # Geometry / style
    p.add_argument("--width", type=int, default=1080)
    p.add_argument("--height", type=int, default=1920)
    p.add_argument("--profile", choices=["clean","karaoke","wordgrid"], default="clean")
    p.add_argument("--profiles-file", type=Path, default=None, help="Optional profile JSON to merge/override.")
    # Clip selection parameters
    p.add_argument("--pre", type=float, default=1.2, help="Seconds before anchor.")
    p.add_argument("--post", type=float, default=2.4, help="Seconds after anchor.")
    p.add_argument("--min-gap", type=float, default=4.0, help="Min separation between anchors in same file.")
    p.add_argument("--max-clips", type=int, default=10)
    p.add_argument("--bitrate", type=str, default="8M")
    # Neo4j / frame config
    p.add_argument("--neo4j-uri", default=NEO4J_URI)
    p.add_argument("--neo4j-user", default=NEO4J_USER)
    p.add_argument("--neo4j-pass", default=NEO4J_PASSWORD)
    p.add_argument("--neo4j-db", default=NEO4J_DB)
    p.add_argument("--frame-label", default="Frame")
    p.add_argument("--frame-embed-prop", default="embedding", help="Frame vector property (embedding/vec/vector/features/feat/...).")
    p.add_argument("--no-index-check", action="store_true")
    # Speaker map (optional tag coloring)
    p.add_argument("--speaker-map", type=Path, default=None)
    # Subcommands
    sub = p.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("from-text", help="Pick scenes by text similarity.")
    s1.add_argument("--q", required=True)
    s1.add_argument("--target", choices=["utterance","segment","transcription"], default="utterance")
    s1.add_argument("--topk", type=int, default=30)
    s1.add_argument("--win-mins", type=int, default=10)

    s2 = sub.add_parser("from-frame", help="Pick scenes by visually similar frames.")
    g = s2.add_mutually_exclusive_group(required=True)
    g.add_argument("--frame-id", type=str)
    g.add_argument("--file-key", type=str)
    s2.add_argument("--frame", type=int, help="Required with --file-key")
    s2.add_argument("--topk", type=int, default=40)

    s3 = sub.add_parser("from-geo", help="Pick scenes by geo proximity.")
    s3.add_argument("--lat", type=float, required=True)
    s3.add_argument("--lon", type=float, required=True)
    s3.add_argument("--radius-m", type=float, default=200.0)
    s3.add_argument("--start-ms", type=int, default=None)
    s3.add_argument("--end-ms", type=int, default=None)
    s3.add_argument("--min-mph", type=float, default=None)
    s3.add_argument("--max-mph", type=float, default=None)
    s3.add_argument("--limit", type=int, default=120)

    s4 = sub.add_parser("from-speed", help="Pick scenes by speed range (mph).")
    s4.add_argument("--min-mph", type=float, required=True)
    s4.add_argument("--max-mph", type=float, required=True)
    s4.add_argument("--limit", type=int, default=120)

    p.add_argument("--log-level", default="INFO")
    return p.parse_args()

def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(message)s")

    # Neo4j config
    global NEO4J_DB
    NEO4J_DB = args.neo4j_db
    driver = neo4j_driver(args.neo4j_uri, args.neo4j_user, args.neo4j_pass)
    if not args.no_index_check:
        ensure_indexes(driver, args.frame_label, args.frame_embed_prop)

    # Profiles
    profiles = dict(DEFAULT_PROFILES)
    if args.profiles_file:
        if not args.profiles_file.exists():
            logging.error(f"--profiles-file not found: {args.profiles_file}")
            return 2
        profiles.update(load_profiles_from_json(args.profiles_file))
    prof = profiles[args.profile]
    y_ratio = float(prof.get("y_pos_ratio", 0.82))

    bases = [Path(b).resolve() for b in (args.bases or [])]
    smap = load_speaker_map(args.speaker_map)  # currently only used for colors/tags if you wire in diar labels later

    # Gather rows according to mode
    if args.cmd == "from-text":
        rows = search_text(driver, args.q, args.target, args.topk, args.win_mins)

    elif args.cmd == "from-frame":
        if not args.frame_id and (not args.file_key or args.frame is None):
            logging.error("--frame-id OR (--file-key and --frame) required.")
            return 2
        seed, meta = get_frame_seed(driver, args.frame_id, args.file_key, args.frame,
                                    args.frame_label, args.frame_embed_prop)
        rows = similar_frames(driver, seed, args.topk, args.frame_label, args.frame_embed_prop)
        # annotate score already provided; rows include file_key, frame, millis?, mph

    elif args.cmd == "from-geo":
        rows = geo_frames(driver, args.lat, args.lon, args.radius_m,
                          args.start_ms, args.end_ms, args.min_mph, args.max_mph, args.limit)

    elif args.cmd == "from-speed":
        # speed-only pull = geo_frames without geo, filter via mph index (scan frames with mph)
        cypher = """
        MATCH (f:Frame)
        WHERE f.mph IS NOT NULL AND f.mph >= $min AND f.mph <= $max
        RETURN f.id AS id, f.key AS file_key, f.frame AS frame, f.millis AS millis, f.mph AS mph
        ORDER BY f.millis ASC
        LIMIT $limit
        """
        with driver.session(database=NEO4J_DB) as sess:
            res = sess.run(cypher, min=float(args.min_mph), max=float(args.max_mph), limit=int(args.limit))
            rows = [r.data() for r in res]
    else:
        logging.error("Unknown subcommand")
        return 2

    if not rows:
        logging.error("No search results found.")
        return 3

    # Build shot list (dedupe & clamp)
    shots = build_shot_list(rows, driver, bases, pre_sec=args.pre, post_sec=args.post,
                            min_gap_sec=args.min_gap, max_clips=args.max_clips)

    if not shots:
        logging.error("No viable shots after dedupe/path resolution.")
        return 4

    # Build final compilation
    out_path = args.out.resolve()
    build_compilation(shots, width=args.width, height=args.height, profile=prof,
                      y_ratio=y_ratio, out_path=out_path, bitrate=args.bitrate)
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
