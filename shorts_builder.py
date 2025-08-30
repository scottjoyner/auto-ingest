#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build vertical 9:16 'shorts' with dynamic, speaker-aware captions from *_FR.MP4.

Adds:
- Optional diarization (RTTM) -> per-word speaker labels
- Optional local->global speaker mapping (JSON)
- Optional Neo4j manifest upsert for Video/Short/Segments/Speakers

Usage examples:
  python shorts_builder.py --file /mnt/.../2025_0101_173933_FR.MP4 \
    --profiles clean karaoke wordgrid \
    --whisper-model base --device cuda \
    --diarization /mnt/.../2025_0101_173933_diarization.rttm \
    --speaker-map speakers_map.json \
    --neo4j-uri bolt://localhost:7687 --neo4j-user neo4j --neo4j-pass 'pass'

"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional, Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip

# Lazy import whisper only if needed
_whisper = None
try:
    from neo4j import GraphDatabase
except Exception:
    GraphDatabase = None  # Only needed if --neo4j-uri is used

DATE_DIR_RE = re.compile(r"^\d{4}/\d{2}/\d{2}$")

# ---------------------------
# Profiles (add as many as you like)
# ---------------------------
DEFAULT_PROFILES: Dict[str, Dict[str, Any]] = {
    "clean": {
        "font_path": "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "font_size": 64,
        "y_pos_ratio": 0.82,
        "line_height_px": 90,
        "stroke_px": 4,
        "colors": {
            "text": (255, 255, 255, 255),
            "stroke": (0, 0, 0, 255),
            "highlight_text": (0, 0, 0, 255),
        },
        "max_width_px": 1000,
        "wordgrid": False,
        "karaoke": True,
        "show_speaker_tag": True,   # show [Speaker] prefix
    },
    "karaoke": {
        "font_path": "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "font_size": 72,
        "y_pos_ratio": 0.80,
        "line_height_px": 100,
        "stroke_px": 6,
        "colors": {
            "text": (255, 255, 255, 255),
            "stroke": (0, 0, 0, 255),
            "highlight_text": (0, 0, 0, 255),
        },
        "max_width_px": 1080 - 160,
        "wordgrid": False,
        "karaoke": True,
        "show_speaker_tag": True,
    },
    "wordgrid": {
        "font_path": "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "font_size": 60,
        "y_pos_ratio": 0.75,
        "line_height_px": 90,
        "stroke_px": 2,
        "colors": {
            "text": (255, 255, 255, 255),
            "stroke": (0, 0, 0, 255),
            "highlight_text": (255, 255, 255, 255),
        },
        "max_width_px": 980,
        "wordgrid": True,
        "karaoke": False,
        "show_speaker_tag": True,
    },
}

def load_profiles_from_json(path: Path) -> Dict[str, Dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    # Convert lists back to tuples for colors
    for prof in data.values():
        if "colors" in prof:
            prof["colors"] = {k: tuple(v) for k, v in prof["colors"].items()}
    return data

# ---------------------------
# Speaker helpers
# ---------------------------

def _hash_color(key: str) -> Tuple[int, int, int, int]:
    """
    Deterministic RGBA color based on key (global_id or local label).
    Nice-ish evenly spaced hues.
    """
    h = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16)
    # Simple hash -> HSV -> RGB (manual quick & dirty)
    hue = (h % 360) / 360.0
    s, v = 0.65, 0.95
    i = int(hue * 6)
    f = hue * 6 - i
    p = int(255 * v * (1 - s))
    q = int(255 * v * (1 - f * s))
    t = int(255 * v * (1 - (1 - f) * s))
    V = int(255 * v)
    i %= 6
    if i == 0: r, g, b = V, t, p
    elif i == 1: r, g, b = q, V, p
    elif i == 2: r, g, b = p, V, t
    elif i == 3: r, g, b = p, q, V
    elif i == 4: r, g, b = t, p, V
    else: r, g, b = V, p, q
    return (r, g, b, 230)

@dataclass
class SpeakerMapEntry:
    local_label: str
    global_id: Optional[str] = None
    name: Optional[str] = None

def load_speaker_map(path: Optional[Path]) -> Dict[str, SpeakerMapEntry]:
    """
    JSON format:
    {
      "SPEAKER_00": {"global_id":"gs_abc","name":"Scott"},
      "SPEAKER_01": {"global_id":"gs_xyz","name":"Madison"}
    }
    """
    mapping: Dict[str, SpeakerMapEntry] = {}
    if not path or not path.exists():
        return mapping
    data = json.loads(path.read_text(encoding="utf-8"))
    for k, v in data.items():
        mapping[k] = SpeakerMapEntry(local_label=k, global_id=v.get("global_id"), name=v.get("name"))
    return mapping

# ---------------------------
# Diarization (RTTM) parsing
# ---------------------------

@dataclass
class DiarSegment:
    start: float
    duration: float
    end: float
    speaker: str

def parse_rttm(rttm_path: Path) -> List[DiarSegment]:
    """
    Parse a simple RTTM file where lines contain:
      SPEAKER <file> 1 <onset> <dur> <ortho> <stype> <name> <conf> <speaker>
    We'll read onset, dur, and speaker label.
    """
    segs: List[DiarSegment] = []
    for line in rttm_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts[0] != "SPEAKER":
            continue
        try:
            onset = float(parts[3]); dur = float(parts[4])
            spk = parts[7] if len(parts) > 7 else parts[-1]
            segs.append(DiarSegment(onset, dur, onset + dur, spk))
        except Exception:
            continue
    segs.sort(key=lambda s: s.start)
    return segs

def align_words_to_speakers(whisper_result: dict, diar: List[DiarSegment]) -> None:
    """
    Mutates whisper_result["segments"][i]["words"][j] to include "speaker" where possible.
    Strategy: assign by word midpoint ∈ diarization segment.
    """
    if not diar:
        return
    for seg in whisper_result.get("segments", []):
        for w in seg.get("words", []):
            st = w.get("start"); en = w.get("end")
            if st is None or en is None: 
                continue
            mid = 0.5 * (st + en)
            # binary search could be added; linear is fine for small lists
            label = None
            for ds in diar:
                if ds.start <= mid <= ds.end:
                    label = ds.speaker
                    break
            if label:
                w["speaker"] = label
        # If no word-level labels, fallback to segment-level majority
        labels = [w.get("speaker") for w in seg.get("words", []) if w.get("speaker")]
        if labels:
            seg["speaker"] = max(set(labels), key=labels.count)

def _apply_speaker_override(profile: Dict, speaker_label: Optional[str], smap: Dict[str, SpeakerMapEntry]) -> Dict:
    """
    Merge per-speaker overrides into the active profile when a GlobalSpeaker mapping exists.
    - Looks up the local speaker_label in smap.
    - If that entry has a global_id and the profile contains speaker_overrides[global_id],
      merge its fields (e.g., colors, font_path, font_size) into a shallow copy of profile.
    """
    if not speaker_label:
        return profile
    entry = smap.get(speaker_label)
    if not entry or not entry.global_id:
        return profile

    overrides = profile.get("speaker_overrides", {}).get(entry.global_id)
    if not overrides:
        return profile

    merged = {**profile}  # shallow copy
    # merge nested dicts like colors
    if "colors" in overrides:
        merged["colors"] = {**profile.get("colors", {}), **overrides["colors"]}
    if "font_path" in overrides:
        merged["font_path"] = overrides["font_path"]
    if "font_size" in overrides:
        merged["font_size"] = overrides["font_size"]
    if "line_height_px" in overrides:
        merged["line_height_px"] = overrides["line_height_px"]
    if "stroke_px" in overrides:
        merged["stroke_px"] = overrides["stroke_px"]
    return merged


# ---------------------------
# Transcription I/O
# ---------------------------

def _default_transcript_paths(fr_file: Path, model: str) -> Tuple[Path, Path]:
    base = fr_file.with_suffix("")
    json_path = base.with_name(f"{base.name}_{model}_whisper.json")
    csv_path  = base.with_name(f"{base.name}_transcription.csv")
    return json_path, csv_path

def save_whisper_json(result: dict, path: Path) -> None:
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")

def save_segments_csv(result: dict, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["SegmentIndex","Start","End","Speaker","Text"])
        for i, seg in enumerate(result.get("segments", [])):
            w.writerow([i, seg.get("start"), seg.get("end"), seg.get("speaker",""), seg.get("text","").strip()])

def load_or_transcribe(audio_path: Path, json_path: Path, *, model_size: str, device: str, language: Optional[str], overwrite: bool) -> dict:
    if json_path.exists() and not overwrite:
        logging.info(f"Using cached transcript: {json_path.name}")
        return json.loads(json_path.read_text(encoding="utf-8"))
    global _whisper
    if _whisper is None:
        import whisper as _w
        _whisper = _w
    logging.info(f"Transcribing {audio_path.name} with Whisper({model_size}, device={device})")
    model = _whisper.load_model(model_size, device=device if device != "auto" else None)
    kwargs = {"word_timestamps": True}
    if language:
        kwargs["language"] = language
    result = model.transcribe(str(audio_path), **kwargs)
    save_whisper_json(result, json_path)
    return result

# ---------------------------
# Video: crop to 9:16
# ---------------------------

def crop_to_vertical(src_video: Path, dst_video: Path, target_w: int, target_h: int) -> Tuple[float, float]:
    tmp = dst_video.with_suffix(".mp4.tmp")
    with VideoFileClip(str(src_video)) as clip:
        w, h = clip.size
        current_aspect = w / h
        target_aspect = target_w / target_h
        if current_aspect > target_aspect:
            new_w = int(h * target_aspect)
            x1 = (w - new_w) // 2
            cropped = clip.crop(x1=x1, y1=0, x2=x1 + new_w, y2=h)
        else:
            new_h = int(w / target_aspect)
            y1 = (h - new_h) // 2
            cropped = clip.crop(x1=0, y1=y1, x2=w, y2=y1 + new_h)
        final = cropped.resize((target_w, target_h))
        fps = clip.fps or 30
        logging.info(f"Writing vertical temp: {tmp.name} (fps={fps})")
        final.write_videofile(
            str(tmp),
            fps=fps,
            codec="libx264",
            audio_codec="aac",
            threads=max(1, (os.cpu_count() or 2) - 1),
            verbose=False,
            logger=None,
        )
        duration = final.duration
    tmp.replace(dst_video)
    return fps, duration

# ---------------------------
# Caption rendering (speaker-aware)
# ---------------------------

@lru_cache(maxsize=4096)
def _load_font(font_path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(font_path, size)

def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> List[str]:
    words = text.split()
    lines: List[str] = []
    cur: List[str] = []
    for w in words:
        test = " ".join(cur + [w])
        wbox = draw.textbbox((0,0), test, font=font)
        if (wbox[2]-wbox[0]) <= max_width or not cur:
            cur.append(w)
        else:
            lines.append(" ".join(cur)); cur = [w]
    if cur:
        lines.append(" ".join(cur))
    return lines

def _speaker_display_name(local_label: Optional[str], smap: Dict[str, SpeakerMapEntry]) -> Optional[str]:
    if not local_label: return None
    entry = smap.get(local_label)
    return entry.name if (entry and entry.name) else local_label

def _speaker_color(local_label: Optional[str], smap: Dict[str, SpeakerMapEntry]) -> Tuple[int,int,int,int]:
    if not local_label:
        return (255, 235, 59, 230)  # default amber
    entry = smap.get(local_label)
    key = entry.global_id or local_label if entry else local_label
    return _hash_color(key)

def _render_tag(text: str, font: ImageFont.ImageFont, color_rgba: Tuple[int,int,int,int]) -> Image.Image:
    dimg = Image.new("RGBA", (10,10), (0,0,0,0))
    d = ImageDraw.Draw(dimg)
    bbox = d.textbbox((0,0), text, font=font)
    w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
    pad = 10
    img = Image.new("RGBA", (w + 2*pad, h + 2*pad), (0,0,0,0))
    di = ImageDraw.Draw(img)
    # tag pill
    r = 12
    di.rounded_rectangle([0,0,img.width,img.height], radius=r, fill=color_rgba)
    di.text((pad, pad), text, font=font, fill=(0,0,0,255))
    return img

def _draw_sentence(
    sentence: str,
    width: int,
    profile: Dict,
    speaker_label: Optional[str],
    smap: Dict[str, SpeakerMapEntry],
) -> Image.Image:
    """Base sentence image (no per-word highlight), speaker-aware via overrides."""
    # 🔹 apply per-speaker override first
    profile = _apply_speaker_override(profile, speaker_label, smap)

    font = _load_font(profile["font_path"], profile["font_size"])
    stroke_px = profile["stroke_px"]
    colors = profile["colors"]
    line_h = profile["line_height_px"]
    max_w = min(profile["max_width_px"], width - 80)

    img = Image.new("RGBA", (width, line_h*3 + 60), (0,0,0,0))
    d = ImageDraw.Draw(img)
    lines = _wrap_text(d, sentence.strip(), font, max_w)

    y = 0
    # optional speaker tag on first row
    if profile.get("show_speaker_tag") and speaker_label:
        tag_font = _load_font(profile["font_path"], int(profile["font_size"]*0.55))
        tag_text = f"[{_speaker_display_name(speaker_label, smap)}]"
        tag_img = _render_tag(tag_text, tag_font, _speaker_color(speaker_label, smap))
        img.paste(tag_img, ((width - tag_img.width)//2, y), tag_img)
        y += tag_img.height + 8

    for line in lines:
        bbox = d.textbbox((0,0), line, font=font, stroke_width=stroke_px)
        tw = bbox[2]-bbox[0]
        x = (width - tw)//2
        d.text((x, y),
               line,
               font=font,
               fill=colors["text"],
               stroke_width=stroke_px,
               stroke_fill=colors["stroke"])
        y += line_h

    return img.crop((0,0,width,y or line_h))


def _draw_sentence_highlight(
    sentence: str,
    word: str,
    width: int,
    profile: Dict,
    speaker_label: Optional[str],
    smap: Dict[str, SpeakerMapEntry],
) -> Image.Image:
    """
    Render sentence with current 'word' highlighted (karaoke style), speaker-aware via overrides.
    """
    # 🔹 apply per-speaker override first
    profile = _apply_speaker_override(profile, speaker_label, smap)

    font = _load_font(profile["font_path"], profile["font_size"])
    stroke_px = profile["stroke_px"]
    colors = profile["colors"]
    line_h = profile["line_height_px"]
    max_w = min(profile["max_width_px"], width - 80)
    hi_bg = _speaker_color(speaker_label, smap)  # highlight box uses per-speaker color

    img = Image.new("RGBA", (width, line_h*3), (0,0,0,0))
    d = ImageDraw.Draw(img)
    lines = _wrap_text(d, sentence.strip(), font, max_w)

    y = 0
    for line in lines:
        x = (width - d.textbbox((0,0), line, font=font, stroke_width=stroke_px)[2])//2
        for token in line.split():
            wbox = d.textbbox((0,0), token, font=font, stroke_width=stroke_px)
            w_w, w_h = wbox[2]-wbox[0], wbox[3]-wbox[1]
            if token.strip() == word.strip():
                pad = 6
                d.rectangle([x-pad, y-pad, x+w_w+pad, y+w_h+pad], fill=hi_bg)
                d.text((x, y), token, font=font, fill=colors["highlight_text"])
            else:
                d.text((x, y),
                       token,
                       font=font,
                       fill=colors["text"],
                       stroke_width=stroke_px,
                       stroke_fill=colors["stroke"])
            x += w_w + font.size//3
        y += line_h

    return img.crop((0,0,width,y or line_h))


def _draw_wordgrid(
    words: List[dict],
    active_idx: int,
    width: int,
    profile: Dict,
    smap: Dict[str, SpeakerMapEntry],
) -> Image.Image:
    """
    Draw words in a grid-like wrapped flow; active word has colored bg & bigger font.
    Uses the active word's speaker for per-speaker overrides (identity styling).
    """
    # figure out the active word's speaker (fallback None)
    active_speaker = None
    if 0 <= active_idx < len(words):
        active_speaker = words[active_idx].get("speaker")

    # 🔹 apply per-speaker override based on active speaker
    profile = _apply_speaker_override(profile, active_speaker, smap)

    base_font = _load_font(profile["font_path"], profile["font_size"])
    active_font = _load_font(profile["font_path"], int(profile["font_size"]*1.18))
    colors = profile["colors"]
    stroke_px = profile["stroke_px"]
    line_h = profile["line_height_px"]
    max_w = min(profile["max_width_px"], width - 80)

    img = Image.new("RGBA", (width, line_h*3), (0,0,0,0))
    d = ImageDraw.Draw(img)

    x = (width - max_w)//2
    y = 0
    for i, w in enumerate(words):
        token = (w.get("word") or "").strip()
        spk = w.get("speaker")  # may be None
        color_bg = _speaker_color(spk, smap)  # per-word bg tint by speaker
        f = active_font if i == active_idx else base_font
        bbox = d.textbbox((0,0), token, font=f, stroke_width=stroke_px)
        w_w, w_h = bbox[2]-bbox[0], bbox[3]-bbox[1]

        if x + w_w > (width + max_w)//2:
            x = (width - max_w)//2
            y += line_h

        if i == active_idx:
            pad = 6
            d.rectangle([x-pad, y-pad, x+w_w+pad, y+w_h+pad], fill=color_bg)
            d.text((x, y), token, font=f, fill=colors["highlight_text"])
        else:
            d.text((x, y), token, font=f, fill=colors["text"],
                   stroke_width=stroke_px, stroke_fill=colors["stroke"])

        x += w_w + base_font.size//3

    return img.crop((0,0,width,y+line_h))


@lru_cache(maxsize=8192)
def _render_cached(key: str) -> np.ndarray:
    spec = json.loads(key)
    prof = spec["profile"]
    smap = {k: SpeakerMapEntry(**v) for k, v in spec["speaker_map"].items()}
    kind = spec["kind"]
    if kind == "sentence":
        img = _draw_sentence(spec["sentence"], spec["width"], prof, spec.get("speaker_label"), smap)
    elif kind == "karaoke":
        img = _draw_sentence_highlight(spec["sentence"], spec["word"], spec["width"], prof, spec.get("speaker_label"), smap)
    elif kind == "wordgrid":
        img = _draw_wordgrid(spec["words"], spec["active_idx"], spec["width"], prof, smap)
    else:
        raise ValueError("Unknown render kind")
    return np.array(img)

def _cap_key(kind: str, **kwargs) -> str:
    return json.dumps({"kind": kind, **kwargs}, sort_keys=True, default=lambda o: o.__dict__ if hasattr(o, "__dict__") else o)

# ---------------------------
# Compose shorts (speaker-aware)
# ---------------------------

def compose_shorts(
    vertical_video: Path,
    transcript: dict,
    out_path: Path,
    *,
    profile_name: str,
    profile: Dict,
    y_ratio: float,
    speaker_map: Dict[str, SpeakerMapEntry],
) -> None:
    tmp = out_path.with_suffix(".mp4.tmp")
    with VideoFileClip(str(vertical_video)) as vclip:
        H, W = vclip.h, vclip.w
        base_y = int(H * y_ratio)

        clips = [vclip]

        if profile.get("wordgrid", False):
            for seg in transcript.get("segments", []):
                words = seg.get("words", [])
                if not words:
                    continue
                for idx, w in enumerate(words):
                    st, en = w.get("start"), w.get("end")
                    if st is None or en is None:
                        continue
                    img_key = _cap_key(
                        "wordgrid",
                        words=words,
                        active_idx=idx,
                        width=W,
                        profile=profile,
                        speaker_map={k: v.__dict__ for k, v in speaker_map.items()},
                    )
                    arr = _render_cached(img_key)
                    ic = ImageClip(arr).set_duration(en - st).set_start(st).set_position(("center", base_y))
                    clips.append(ic)
        else:
            for seg in transcript.get("segments", []):
                s_start, s_end = seg.get("start"), seg.get("end")
                sentence = (seg.get("text") or "").strip()
                if not sentence or s_start is None or s_end is None:
                    continue
                seg_speaker = seg.get("speaker")

                # base sentence with optional [Speaker] tag
                base_key = _cap_key(
                    "sentence",
                    sentence=sentence,
                    width=W,
                    profile=profile,
                    speaker_label=seg_speaker,
                    speaker_map={k: v.__dict__ for k, v in speaker_map.items()},
                )
                base_arr = _render_cached(base_key)
                base_clip = ImageClip(base_arr).set_duration(s_end - s_start).set_start(s_start).set_position(("center", base_y))
                clips.append(base_clip)

                if profile.get("karaoke", True):
                    for w in seg.get("words", []):
                        token = (w.get("word") or "").strip()
                        st, en = w.get("start"), w.get("end")
                        if not token or st is None or en is None:
                            continue
                        w_speaker = w.get("speaker", seg_speaker)
                        kara_key = _cap_key(
                            "karaoke",
                            sentence=sentence,
                            word=token,
                            width=W,
                            profile=profile,
                            speaker_label=w_speaker,
                            speaker_map={k: v.__dict__ for k, v in speaker_map.items()},
                        )
                        kara_arr = _render_cached(kara_key)
                        kara_clip = ImageClip(kara_arr).set_duration(en - st).set_start(st).set_position(("center", base_y))
                        clips.append(kara_clip)

        logging.info(f"Writing {out_path.name}")
        CompositeVideoClip(clips).write_videofile(
            str(tmp),
            fps=vclip.fps or 30,
            codec="libx264",
            audio_codec="aac",
            threads=max(1, (os.cpu_count() or 2) - 1),
            verbose=False,
            logger=None,
        )
    tmp.replace(out_path)
    logging.info(f"Wrote {out_path}")

# ---------------------------
# Discovery & orchestration
# ---------------------------

def is_valid_date_structure(base: Path, d: Path) -> bool:
    try:
        rel = d.relative_to(base).as_posix()
    except ValueError:
        return False
    if not DATE_DIR_RE.match(rel):
        return False
    y, m, day = rel.split("/")
    import datetime as _dt
    try:
        _dt.datetime(int(y), int(m), int(day))
        return True
    except ValueError:
        return False

def iter_date_dirs(base: Path) -> Iterable[Path]:
    for y in sorted(p for p in base.iterdir() if p.is_dir() and p.name.isdigit() and len(p.name) == 4):
        for m in sorted(p for p in y.iterdir() if p.is_dir() and p.name.isdigit() and len(p.name) == 2):
            for d in sorted(p for p in m.iterdir() if p.is_dir() and p.name.isdigit() and len(p.name) == 2):
                if is_valid_date_structure(base, d):
                    yield d

def iter_fr_mp4(root: Path) -> Iterable[Path]:
    for p in root.glob("*_FR.MP4"):
        if p.is_file():
            yield p

def find_sidecar(path: Path, suffix: str) -> Optional[Path]:
    cand = path.with_name(path.stem.replace("_FR", "") + suffix)
    return cand if cand.exists() else None

# ---------------------------
# Neo4j upsert
# ---------------------------

def neo4j_upsert_manifest(
    uri: str, user: str, password: str,
    *,
    video_key: str,
    fr_path: str,
    short_profile: str,
    short_path: str,
    width: int,
    height: int,
    fps: float,
    duration: float,
    whisper_model: str,
    transcript: dict,
    speaker_map: Dict[str, SpeakerMapEntry],
) -> None:
    if GraphDatabase is None:
        logging.error("neo4j Python driver not installed; cannot upsert manifest.")
        return

    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as sess:
        # Upsert Video + Short
        sess.run("""
        MERGE (v:Video {key:$video_key})
          ON CREATE SET v.path=$fr_path, v.created_at=timestamp()
          ON MATCH  SET v.path=$fr_path
        MERGE (s:Short {key:$video_key, profile:$profile})
          ON CREATE SET s.path=$short_path, s.width=$width, s.height=$height,
                        s.fps=$fps, s.duration=$duration, s.model=$model, s.created_at=timestamp()
          ON MATCH  SET s.path=$short_path, s.width=$width, s.height=$height,
                        s.fps=$fps, s.duration=$duration, s.model=$model
        MERGE (s)-[:FROM_VIDEO]->(v)
        """, video_key=video_key, fr_path=fr_path, profile=short_profile,
             short_path=short_path, width=width, height=height, fps=fps,
             duration=duration, model=whisper_model)

        # Upsert segments + speakers
        for i, seg in enumerate(transcript.get("segments", [])):
            s_start = seg.get("start"); s_end = seg.get("end")
            s_text = (seg.get("text") or "").strip()
            s_speaker = seg.get("speaker")
            entry = speaker_map.get(s_speaker) if s_speaker else None
            global_id = entry.global_id if entry and entry.global_id else None

            sess.run("""
            MERGE (seg:Segment {key:$video_key, index:$idx, profile:$profile})
              ON CREATE SET seg.start=$start, seg.end=$end, seg.text=$text, seg.speaker_label=$speaker, seg.created_at=timestamp()
              ON MATCH  SET seg.start=$start, seg.end=$end, seg.text=$text, seg.speaker_label=$speaker
            MERGE (short:Short {key:$video_key, profile:$profile})
            MERGE (short)-[:HAS_SEGMENT]->(seg)
            """, video_key=video_key, idx=i, profile=short_profile, start=s_start, end=s_end, text=s_text, speaker=s_speaker)

            if s_speaker:
                sess.run("""
                MERGE (sp:Speaker {label:$label})
                MERGE (seg:Segment {key:$video_key, index:$idx, profile:$profile})
                MERGE (seg)-[:SPOKEN_BY]->(sp)
                """, label=s_speaker, video_key=video_key, idx=i, profile=short_profile)

                if global_id:
                    sess.run("""
                    MERGE (g:GlobalSpeaker {global_id:$gid})
                      ON CREATE SET g.name=$gname, g.created_at=timestamp()
                      ON MATCH  SET g.name=coalesce($gname, g.name)
                    MERGE (sp:Speaker {label:$label})
                    MERGE (sp)-[:IS]->(g)
                    """, gid=global_id, gname=entry.name, label=s_speaker)

    driver.close()

# ---------------------------
# File-level processing
# ---------------------------

def process_one(
    fr_path: Path,
    *,
    profiles: Dict[str, Dict],
    model_size: str,
    device: str,
    language: Optional[str],
    target_w: int,
    target_h: int,
    overwrite: bool,
    dry_run: bool,
    diarization_path: Optional[Path],
    speaker_map: Dict[str, SpeakerMapEntry],
    neo4j_cfg: Optional[dict],
) -> Tuple[int, int]:
    planned = 0
    done = 0

    base = fr_path.with_suffix("")
    key = base.name  # e.g., 2025_0101_173933
    r_audio = fr_path.with_name(f"{key}_R.MP4")
    audio_path = r_audio if r_audio.exists() else fr_path

    json_path, csv_path = _default_transcript_paths(fr_path, model_size)
    vertical_path = fr_path.with_name(f"{key}_FR_vertical.mp4")

    outputs = [fr_path.with_name(f"{key}_short-{name}.mp4") for name in profiles.keys()]
    if all(p.exists() for p in outputs) and vertical_path.exists() and json_path.exists() and csv_path.exists() and not overwrite:
        logging.info(f"Skip (all outputs exist): {fr_path.name}")
        return (0, 0)

    planned = len(outputs)

    if dry_run:
        logging.info(f"[DRY-RUN] Would process {fr_path.name} -> {', '.join(p.name for p in outputs)}")
        return (0, planned)

    # Transcribe
    transcript = load_or_transcribe(
        audio_path=audio_path,
        json_path=json_path,
        model_size=model_size,
        device=device,
        language=language,
        overwrite=overwrite,
    )

    # Diarization (optional)
    diar_file = diarization_path or find_sidecar(fr_path, "_diarization.rttm")
    diar_segments: List[DiarSegment] = parse_rttm(diar_file) if (diar_file and diar_file.exists()) else []
    if diar_segments:
        align_words_to_speakers(transcript, diar_segments)
        # If segment speaker missing, propagate from words majority already done in align_words_to_speakers

    save_segments_csv(transcript, csv_path)

    # Vertical crop/resize
    if overwrite or not vertical_path.exists():
        fps, duration = crop_to_vertical(fr_path, vertical_path, target_w, target_h)
    else:
        # read basic info
        with VideoFileClip(str(vertical_path)) as vc:
            fps, duration = vc.fps or 30, vc.duration

    # Build each profile
    for pname, prof in profiles.items():
        out = fr_path.with_name(f"{key}_short-{pname}.mp4")
        if out.exists() and not overwrite:
            logging.debug(f"Skip (exists): {out.name}")
            continue
        try:
            compose_shorts(
                vertical_video=vertical_path,
                transcript=transcript,
                out_path=out,
                profile_name=pname,
                profile=prof,
                y_ratio=prof.get("y_pos_ratio", 0.82),
                speaker_map=speaker_map,
            )
            done += 1

            # Neo4j manifest (optional)
            if neo4j_cfg:
                neo4j_upsert_manifest(
                    neo4j_cfg["uri"], neo4j_cfg["user"], neo4j_cfg["password"],
                    video_key=key,
                    fr_path=str(fr_path),
                    short_profile=pname,
                    short_path=str(out),
                    width=target_w,
                    height=target_h,
                    fps=fps,
                    duration=duration,
                    whisper_model=model_size,
                    transcript=transcript,
                    speaker_map=speaker_map,
                )
        except Exception as e:
            logging.exception(f"Failed building {out.name}: {e}")

    return (done, planned)

def process_root(
    root: Path,
    *,
    profiles: Dict[str, Dict],
    model_size: str,
    device: str,
    language: Optional[str],
    target_w: int,
    target_h: int,
    overwrite: bool,
    dry_run: bool,
    diarization_path: Optional[Path],
    speaker_map: Dict[str, SpeakerMapEntry],
    neo4j_cfg: Optional[dict],
) -> Tuple[int, int]:
    planned = 0; done = 0
    for date_dir in iter_date_dirs(root):
        for fr in iter_fr_mp4(date_dir):
            d, p = process_one(
                fr,
                profiles=profiles,
                model_size=model_size,
                device=device,
                language=language,
                target_w=target_w,
                target_h=target_h,
                overwrite=overwrite,
                dry_run=dry_run,
                diarization_path=diarization_path,  # per-file sidecar auto-detected if None
                speaker_map=speaker_map,
                neo4j_cfg=neo4j_cfg,
            )
            planned += p; done += d
    return (done, planned)

# ---------------------------
# CLI
# ---------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build speaker-aware phone shorts from *_FR.MP4 with optional Neo4j manifest.")
    # Inputs
    p.add_argument("--base", dest="bases", action="append", default=[], help="Dashcam root with YYYY/MM/DD (repeatable).")
    p.add_argument("--file", type=Path, default=None, help="Single *_FR.MP4 to process.")
    # Whisper
    p.add_argument("--whisper-model", default="base", help="tiny/base/small/medium/large")
    p.add_argument("--device", default="auto", help="auto/cpu/cuda")
    p.add_argument("--language", default=None, help="Force language (e.g., en)")
    # Geometry
    p.add_argument("--width", type=int, default=1080)
    p.add_argument("--height", type=int, default=1920)
    # Behavior
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--profiles", nargs="+", default=["clean","karaoke"], help="Subset of clean/karaoke/wordgrid")
    p.add_argument("--log-level", default="INFO")
    # Diarization / speakers
    p.add_argument("--diarization", type=Path, default=None, help="RTTM file (if omitted, auto-detect <key>_diarization.rttm).")
    p.add_argument("--speaker-map", type=Path, default=None, help="JSON mapping local speaker -> {global_id,name}.")
    # Neo4j
    p.add_argument("--neo4j-uri", type=str, default=None)
    p.add_argument("--neo4j-user", type=str, default=None)
    p.add_argument("--neo4j-pass", type=str, default=None)
    return p.parse_args()

def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")

    if args.device == "auto":
        args.device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "", "-1") else "cpu"

    profiles = {k: DEFAULT_PROFILES[k] for k in args.profiles if k in DEFAULT_PROFILES}
    if args.profiles_file:
        ext_profiles = load_profiles_from_json(args.profiles_file)
        profiles.update(ext_profiles)

    smap = load_speaker_map(args.speaker_map)
    neo4j_cfg = None
    if args.neo4j_uri and args.neo4j_user and args.neo4j_pass:
        neo4j_cfg = {"uri": args.neo4j_uri, "user": args.neo4j_user, "password": args.neo4j_pass}

    total_done = 0; total_planned = 0

    if args.file:
        fr = args.file.resolve()
        if not fr.exists() or not fr.name.endswith("_FR.MP4"):
            logging.error("--file must be an existing *_FR.MP4"); return 2
        d, p = process_one(
            fr,
            profiles=profiles,
            model_size=args.whisper_model,
            device=args.device,
            language=args.language,
            target_w=args.width,
            target_h=args.height,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            diarization_path=args.diarization,
            speaker_map=smap,
            neo4j_cfg=neo4j_cfg,
        )
        total_done += d; total_planned += p
    else:
        bases = [Path(b).resolve() for b in (args.bases or ["/mnt/8TB_2025/fileserver/dashcam/"])]
        for base in bases:
            if not base.is_dir():
                logging.warning(f"--base not found/dir: {base}"); continue
            d, p = process_root(
                base,
                profiles=profiles,
                model_size=args.whisper_model,
                device=args.device,
                language=args.language,
                target_w=args.width,
                target_h=args.height,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
                diarization_path=args.diarization,
                speaker_map=smap,
                neo4j_cfg=neo4j_cfg,
            )
            total_done += d; total_planned += p

    logging.info(f"Completed {total_done}/{total_planned} short(s).")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
