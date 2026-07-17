"""Shared caption + profile helpers for the shorts render stack.

This module is the single source of truth for speaker-aware caption rendering
previously duplicated across ``shorts_builder.py``, ``smart_shorts.py`` and
``tiktok_shorts.py``. The root-level ``shorts_builder.py`` now imports these
helpers so behavior stays identical for the legacy CLI; the modern
``auto_ingest.shorts`` package uses them via :mod:`auto_ingest.shorts.compose`.

Behavior is unchanged from the original implementations: same font loading,
text wrapping, speaker-hash colors, and cached caption bitmap rendering.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------
# Profiles
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
        "show_speaker_tag": True,
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
    "cinematic": {
        "font_path": "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "font_size": 66,
        "y_pos_ratio": 0.80,
        "line_height_px": 92,
        "stroke_px": 5,
        "colors": {
            "text": (255, 255, 255, 255),
            "stroke": (0, 0, 0, 235),
            "highlight_text": (0, 0, 0, 255),
        },
        "max_width_px": 1040,
        "wordgrid": False,
        "karaoke": True,
        "show_speaker_tag": True,
        "kenburns": True,
        "show_speed_hud": True,
        "anim_sec": 0.4,
    },
}


def load_profiles_from_json(path) -> Dict[str, Dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for prof in data.values():
        if "colors" in prof:
            prof["colors"] = {k: tuple(v) for k, v in prof["colors"].items()}
    return data


# ---------------------------
# Speaker model
# ---------------------------

@dataclass
class SpeakerMapEntry:
    local_label: str
    global_id: Optional[str] = None
    name: Optional[str] = None


def load_speaker_map(path: Optional[Path]) -> Dict[str, SpeakerMapEntry]:  # noqa: F821
    """Load a ``{local_label: {global_id, name}}`` JSON map into dataclasses.

    Accepting a bare ``Path`` for type-checker friendliness; the real
    ``shorts_builder`` passes a ``pathlib.Path``.
    """
    from pathlib import Path as _Path

    mapping: Dict[str, SpeakerMapEntry] = {}
    p = _Path(path) if path else None
    if not p or not p.exists():
        return mapping
    data = json.loads(p.read_text(encoding="utf-8"))
    for k, v in data.items():
        mapping[k] = SpeakerMapEntry(
            local_label=k, global_id=v.get("global_id"), name=v.get("name")
        )
    return mapping


# ---------------------------
# Profile normalization
# ---------------------------

def _normalize_colors_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure PIL-compatible tuples for color entries; leave others as-is."""
    if not isinstance(d, dict):
        return d
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, list):
            out[k] = tuple(v)
        elif isinstance(v, tuple) or isinstance(v, int):
            out[k] = v
        else:
            out[k] = v
    return out


def _normalize_profile_from_json(prof: Dict[str, Any]) -> Dict[str, Any]:
    """Recast JSON-serialized color lists back to tuples (preserve nested overrides)."""
    if not isinstance(prof, dict):
        return prof
    p = dict(prof)
    if "colors" in p and isinstance(p["colors"], dict):
        p["colors"] = _normalize_colors_dict(p["colors"])
    if "speaker_overrides" in p and isinstance(p["speaker_overrides"], dict):
        so: Dict[str, Any] = {}
        for gid, ov in p["speaker_overrides"].items():
            if isinstance(ov, dict) and "colors" in ov and isinstance(ov["colors"], dict):
                ov2 = dict(ov)
                ov2["colors"] = _normalize_colors_dict(ov["colors"])
                so[gid] = ov2
            else:
                so[gid] = ov
        p["speaker_overrides"] = so
    return p


def _apply_speaker_override(
    profile: Dict, speaker_label: Optional[str], smap: Dict[str, SpeakerMapEntry]
) -> Dict:
    """Merge per-speaker overrides into the active profile when a mapping exists."""
    if not speaker_label:
        return profile
    entry = smap.get(speaker_label)
    if not entry or not entry.global_id:
        return profile
    overrides = profile.get("speaker_overrides", {}).get(entry.global_id)
    if not overrides:
        return profile
    merged = {**profile}
    if "colors" in overrides:
        ov_colors = overrides["colors"]
        if isinstance(ov_colors, dict):
            ov_colors = _normalize_colors_dict(ov_colors)
        merged["colors"] = {**profile.get("colors", {}), **ov_colors}
    for k in ("font_path", "font_size", "line_height_px", "stroke_px"):
        if k in overrides:
            merged[k] = overrides[k]
    return merged


# ---------------------------
# Color / font helpers
# ---------------------------

def _hash_color(key: str) -> Tuple[int, int, int, int]:
    """Deterministic RGBA color based on key (global_id or local label)."""
    h = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16)
    hue = (h % 360) / 360.0
    s, v = 0.65, 0.95
    i = int(hue * 6)
    f = hue * 6 - i
    p = int(255 * v * (1 - s))
    q = int(255 * v * (1 - f * s))
    t = int(255 * v * (1 - (1 - f) * s))
    V = int(255 * v)
    i %= 6
    if i == 0:
        r, g, b = V, t, p
    elif i == 1:
        r, g, b = q, V, p
    elif i == 2:
        r, g, b = p, V, t
    elif i == 3:
        r, g, b = p, q, V
    elif i == 4:
        r, g, b = t, p, V
    else:
        r, g, b = V, p, q
    return (r, g, b, 230)


@lru_cache(maxsize=4096)
def _load_font(font_path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(font_path, size)
    except Exception:
        return ImageFont.load_default()


def _wrap_text(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int
) -> List[str]:
    words = text.split()
    lines: List[str] = []
    cur: List[str] = []
    for w in words:
        test = " ".join(cur + [w])
        wbox = draw.textbbox((0, 0), test, font=font)
        if (wbox[2] - wbox[0]) <= max_width or not cur:
            cur.append(w)
        else:
            lines.append(" ".join(cur))
            cur = [w]
        # Hard guard: a single word longer than the frame must be broken so it
        # never overflows the caption band.
        if cur:
            cur_test = " ".join(cur)
            cb = draw.textbbox((0, 0), cur_test, font=font)
            if (cb[2] - cb[0]) > max_width:
                overflow = cur.pop()
                if cur:
                    lines.append(" ".join(cur))
                    cur = []
                piece = ""
                for ch in overflow:
                    if (
                        draw.textbbox((0, 0), piece + ch, font=font)[2]
                        - draw.textbbox((0, 0), piece + ch, font=font)[0]
                        > max_width
                        and piece
                    ):
                        lines.append(piece)
                        piece = ch
                    else:
                        piece += ch
                if piece:
                    cur = [piece]
    if cur:
        lines.append(" ".join(cur))
    return lines


def _speaker_display_name(
    local_label: Optional[str], smap: Dict[str, SpeakerMapEntry]
) -> Optional[str]:
    if not local_label:
        return None
    entry = smap.get(local_label)
    return entry.name if (entry and entry.name) else local_label


def _speaker_color(
    local_label: Optional[str], smap: Dict[str, SpeakerMapEntry]
) -> Tuple[int, int, int, int]:
    if not local_label:
        return (255, 235, 59, 230)  # default amber
    entry = smap.get(local_label)
    key = entry.global_id or local_label if entry else local_label
    return _hash_color(key)


def _render_tag(
    text: str, font: ImageFont.ImageFont, color_rgba: Tuple[int, int, int, int]
) -> Image.Image:
    dimg = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    d = ImageDraw.Draw(dimg)
    bbox = d.textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = 10
    img = Image.new("RGBA", (w + 2 * pad, h + 2 * pad), (0, 0, 0, 0))
    di = ImageDraw.Draw(img)
    r = 12
    di.rounded_rectangle([0, 0, img.width, img.height], radius=r, fill=color_rgba)
    di.text((pad, pad), text, font=font, fill=(0, 0, 0, 255))
    return img


# Accent colors per cue kind (research-scripted shorts).
_CUE_KIND_STYLE: Dict[str, Dict[str, Any]] = {
    "hook": {"accent": (255, 196, 0, 235), "font_scale": 1.15, "tag": "Topic"},
    "point": {"accent": (0, 200, 255, 235), "font_scale": 1.0, "tag": None},
    "source": {"accent": (180, 120, 255, 235), "font_scale": 0.92, "tag": "From your words"},
    "trip": {"accent": (120, 230, 150, 235), "font_scale": 1.0, "tag": "On the road"},
    "mood": {"accent": (255, 120, 160, 235), "font_scale": 1.05, "tag": None},
    "line": {"accent": (255, 235, 59, 230), "font_scale": 1.0, "tag": None},
    "callback": {"accent": (255, 196, 0, 235), "font_scale": 1.0, "tag": "Don't miss"},
    "payoff": {"accent": (120, 230, 150, 235), "font_scale": 1.1, "tag": "Payoff"},
    "myth": {"accent": (255, 80, 80, 235), "font_scale": 1.0, "tag": "Myth"},
    "fact": {"accent": (80, 220, 120, 235), "font_scale": 1.0, "tag": "Fact"},
    "series": {"accent": (0, 200, 255, 235), "font_scale": 1.2, "tag": "Series"},
    "question": {"accent": (255, 196, 0, 235), "font_scale": 1.0, "tag": "Wait —"},
    "reveal": {"accent": (120, 230, 150, 235), "font_scale": 1.15, "tag": "Answer"},
}


def _cue_kind_style(kind: str) -> Dict[str, Any]:
    return _CUE_KIND_STYLE.get(kind, _CUE_KIND_STYLE["line"])


# ---------------------------
# Caption drawing
# ---------------------------

def _draw_sentence(
    sentence: str,
    width: int,
    profile: Dict,
    speaker_label: Optional[str],
    smap: Dict[str, SpeakerMapEntry],
) -> Image.Image:
    """Base sentence image (no per-word highlight), speaker-aware via overrides."""
    profile = _normalize_profile_from_json(profile)
    profile = _apply_speaker_override(profile, speaker_label, smap)

    font = _load_font(profile["font_path"], profile["font_size"])
    stroke_px = int(profile["stroke_px"])
    colors = _normalize_colors_dict(profile["colors"])
    line_h = int(profile["line_height_px"])
    max_w = min(int(profile["max_width_px"]), width - 80)

    img = Image.new("RGBA", (width, line_h * 3 + 60), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    lines = _wrap_text(d, sentence.strip(), font, max_w)

    y = 0
    if profile.get("show_speaker_tag") and speaker_label:
        tag_font = _load_font(profile["font_path"], int(profile["font_size"] * 0.55))
        tag_text = f"[{_speaker_display_name(speaker_label, smap)}]"
        tag_img = _render_tag(tag_text, tag_font, _speaker_color(speaker_label, smap))
        img.paste(tag_img, ((width - tag_img.width) // 2, y), tag_img)
        y += tag_img.height + 8

    for line in lines:
        bbox = d.textbbox((0, 0), line, font=font, stroke_width=stroke_px)
        tw = bbox[2] - bbox[0]
        x = (width - tw) // 2
        d.text(
            (x, y),
            line,
            font=font,
            fill=colors.get("text", (255, 255, 255, 255)),
            stroke_width=stroke_px,
            stroke_fill=colors.get("stroke", (0, 0, 0, 255)),
        )
        y += line_h

    return img.crop((0, 0, width, y or line_h))


def _draw_sentence_highlight(
    sentence: str,
    word: str,
    width: int,
    profile: Dict,
    speaker_label: Optional[str],
    smap: Dict[str, SpeakerMapEntry],
) -> Image.Image:
    """Render sentence with current ``word`` highlighted (karaoke style)."""
    profile = _normalize_profile_from_json(profile)
    profile = _apply_speaker_override(profile, speaker_label, smap)

    font = _load_font(profile["font_path"], profile["font_size"])
    stroke_px = int(profile["stroke_px"])
    colors = _normalize_colors_dict(profile["colors"])
    line_h = int(profile["line_height_px"])
    max_w = min(int(profile["max_width_px"]), width - 80)
    hi_bg = _speaker_color(speaker_label, smap)

    img = Image.new("RGBA", (width, line_h * 3), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    lines = _wrap_text(d, sentence.strip(), font, max_w)

    y = 0
    for line in lines:
        x = (width - d.textbbox((0, 0), line, font=font, stroke_width=stroke_px)[2]) // 2
        for token in line.split():
            wbox = d.textbbox((0, 0), token, font=font, stroke_width=stroke_px)
            w_w, w_h = wbox[2] - wbox[0], wbox[3] - wbox[1]
            if token.strip().casefold() == word.strip().casefold():
                pad = 6
                d.rectangle([x - pad, y - pad, x + w_w + pad, y + w_h + pad], fill=hi_bg)
                d.text((x, y), token, font=font, fill=colors.get("highlight_text", (0, 0, 0, 255)))
            else:
                d.text(
                    (x, y),
                    token,
                    font=font,
                    fill=colors.get("text", (255, 255, 255, 255)),
                    stroke_width=stroke_px,
                    stroke_fill=colors.get("stroke", (0, 0, 0, 255)),
                )
            x += w_w + font.size // 3
        y += line_h

    return img.crop((0, 0, width, y or line_h))


def _draw_wordgrid(
    words: List[dict],
    active_idx: int,
    width: int,
    profile: Dict,
    smap: Dict[str, SpeakerMapEntry],
) -> Image.Image:
    """Draw words in a wrapped flow; active word has colored bg + bigger font."""
    active_speaker = None
    if 0 <= active_idx < len(words):
        active_speaker = words[active_idx].get("speaker")

    profile = _normalize_profile_from_json(profile)
    profile = _apply_speaker_override(profile, active_speaker, smap)

    base_font = _load_font(profile["font_path"], profile["font_size"])
    active_font = _load_font(profile["font_path"], int(profile["font_size"] * 1.18))
    colors = _normalize_colors_dict(profile["colors"])
    stroke_px = int(profile["stroke_px"])
    line_h = int(profile["line_height_px"])
    max_w = min(int(profile["max_width_px"]), width - 80)

    img = Image.new("RGBA", (width, line_h * 3), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    x = (width - max_w) // 2
    y = 0
    for i, w in enumerate(words):
        token = (w.get("word") or "").strip()
        spk = w.get("speaker")
        color_bg = _speaker_color(spk, smap)
        f = active_font if i == active_idx else base_font
        bbox = d.textbbox((0, 0), token, font=f, stroke_width=stroke_px)
        w_w, w_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

        if x + w_w > (width + max_w) // 2:
            x = (width - max_w) // 2
            y += line_h

        if i == active_idx:
            pad = 6
            d.rectangle([x - pad, y - pad, x + w_w + pad, y + w_h + pad], fill=color_bg)
            d.text((x, y), token, font=f, fill=colors.get("highlight_text", (255, 255, 255, 255)))
        else:
            d.text(
                (x, y),
                token,
                font=f,
                fill=colors.get("text", (255, 255, 255, 255)),
                stroke_width=stroke_px,
                stroke_fill=colors.get("stroke", (0, 0, 0, 255)),
            )

        x += w_w + base_font.size // 3

    return img.crop((0, 0, width, y + line_h))


@lru_cache(maxsize=8192)
def _render_cached(key: str) -> np.ndarray:
    spec = json.loads(key)
    prof = _normalize_profile_from_json(spec["profile"])
    smap = {k: SpeakerMapEntry(**v) for k, v in spec["speaker_map"].items()}
    kind = spec["kind"]
    if kind == "sentence":
        img = _draw_sentence(spec["sentence"], spec["width"], prof, spec.get("speaker_label"), smap)
    elif kind == "karaoke":
        img = _draw_sentence_highlight(
            spec["sentence"], spec["word"], spec["width"], prof, spec.get("speaker_label"), smap
        )
    elif kind == "wordgrid":
        img = _draw_wordgrid(spec["words"], spec["active_idx"], spec["width"], prof, smap)
    else:
        raise ValueError("Unknown render kind")
    return np.array(img)


def _cap_key(kind: str, **kwargs) -> str:
    return json.dumps(
        {"kind": kind, **kwargs}, sort_keys=True,
        default=lambda o: o.__dict__ if hasattr(o, "__dict__") else o,
    )


def _build_caption_image(text: str, width: int, profile: Dict, kind: str) -> Image.Image:
    """Render a styled caption image for one cue (per-kind font/accent/tag)."""
    kstyle = _cue_kind_style(kind)
    prof = _normalize_profile_from_json(profile)
    prof = dict(prof)
    prof["font_size"] = int(prof.get("font_size", 64) * kstyle["font_scale"])
    accent = kstyle["accent"]
    img = _draw_sentence(text, width, prof, None, {}).convert("RGBA")
    if accent:
        ux = ImageDraw.Draw(img)
        ux.rectangle([width * 0.30, img.height - 6, width * 0.70, img.height - 2], fill=accent)
    if kstyle.get("tag"):
        tag_font = _load_font(prof["font_path"], int(prof.get("font_size", 64) * 0.5))
        tag_img = _render_tag(kstyle["tag"], tag_font, accent)
        framed = Image.new("RGBA", (width, img.height + tag_img.height + 8), (0, 0, 0, 0))
        framed.paste(tag_img, ((width - tag_img.width) // 2, 0), tag_img)
        framed.paste(img, (0, tag_img.height + 8), img)
        img = framed
    return img


def _cue_word_timings(
    text: str, start: float, end: float, explicit: Optional[List[Dict[str, Any]]]
) -> List[Tuple[str, float, float]]:
    """Return [(word, w_start, w_end), ...] for a cue (even split if no timings)."""
    words = [w for w in text.split() if w]
    if not words:
        return []
    if explicit:
        out = []
        for w in explicit:
            ws = w.get("start")
            we = w.get("end")
            if ws is None or we is None:
                continue
            out.append((str(w.get("word", "")).strip(), float(ws), float(we)))
        if out:
            return out
    span = max(end - start, 0.1)
    step = span / len(words)
    MIN_VIS = 0.18
    out = []
    for i, w in enumerate(words):
        ws = start + i * step
        we = start + (i + 1) * step
        if we - ws < MIN_VIS:
            we = ws + MIN_VIS
        out.append((w, ws, we))
    return out


# ---------------------------
# Scrim / Ken Burns / HUD / end card (shared visual primitives)
# ---------------------------

def _draw_scrim(width: int, band_h: int, color: Tuple[int, int, int, int] = (0, 0, 0, 150)) -> Image.Image:
    """Vertical gradient scrim (transparent -> color -> transparent) for legibility."""
    img = Image.new("RGBA", (width, band_h), (0, 0, 0, 0))
    d = img.load()
    r, g, b, a = color
    for y in range(band_h):
        t = 1.0 - abs((y / max(band_h - 1, 1)) - 0.5) * 2.0
        alpha = int(a * t * t)
        for x in range(width):
            d[x, y] = (r, g, b, alpha)
    return img


def _apply_kenburns(clip, *, zoom: float = 1.12, pan: Tuple[float, float] = (0.04, 0.04)):
    """Subtle slow zoom + pan ('Ken Burns') so static B-roll feels alive."""
    import numpy as _np
    from PIL import Image as _Image

    w, h = clip.w, clip.h
    dur = float(clip.duration or 0.0) or 1.0

    def _ease(t: float) -> float:
        x = max(0.0, min(1.0, t / dur))
        return x * x * (3.0 - 2.0 * x)

    def _factor(t: float) -> float:
        return 1.0 + (zoom - 1.0) * _ease(t)

    def _frame(get_frame, t):
        f = _factor(t)
        ow = int(w * f)
        oh = int(h * f)
        px_max = max(0, ow - w)
        py_max = max(0, oh - h)
        px = int(pan[0] * px_max * _ease(t))
        py = int(pan[1] * py_max * _ease(t))
        px = max(0, min(px, px_max))
        py = max(0, min(py, py_max))
        arr = get_frame(t)
        im = _Image.fromarray(arr).resize((ow, oh), _Image.LANCZOS)
        cropped = im.crop((px, py, px + w, py + h))
        return _np.array(cropped)

    return clip.fl(_frame).resize((w, h))


def _draw_speed_hud(mph: float, width: int, height: int,
                    accent: Tuple[int, int, int, int] = (0, 200, 255, 235)) -> Image.Image:
    """Small top-left speed readout: a rounded gauge + 'NN mph' label."""
    pad = 28
    box_w, box_h = 300, 96
    x0, y0 = pad, pad
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([x0, y0, x0 + box_w, y0 + box_h], radius=20, fill=(0, 0, 0, 140))
    d.rounded_rectangle([x0, y0, x0 + box_w, y0 + box_h], radius=20, outline=accent, width=3)
    font = _load_font("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 46)
    label = f"{int(round(mph))} mph"
    d.text((x0 + 18, y0 + 22), label, font=font, fill=(255, 255, 255, 255))
    for i in range(5):
        tx = x0 + 18 + i * 52
        d.line([(tx, y0 + box_h - 14), (tx, y0 + box_h - 6)], fill=accent, width=4)
    return img


def _draw_end_card(width: int, height: int, hashtag: str,
                   accent: Tuple[int, int, int, int] = (0, 200, 255, 235)) -> Image.Image:
    """Branded outro: scrim + 'your words, not a paper' + topic hashtag."""
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    scrim = _draw_scrim(width, 520)
    img.paste(scrim, (0, height - 520), scrim)
    ax = width // 2
    d.rectangle([ax - 120, height - 430, ax + 120, height - 422], fill=accent)
    font_big = _load_font("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 58)
    d.text((ax - 300, height - 400), "your words, not a paper", font=font_big,
           fill=(255, 255, 255, 255), anchor="mm")
    font_sub = _load_font("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
    sub = f"# {hashtag}" if hashtag else "subscribe"
    d.text((ax, height - 320), sub, font=font_sub, fill=accent, anchor="mm")
    return img


# ---------------------------
# Safe video writing utilities
# ---------------------------

def _ensure_parent_dir(path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _temp_path_with_same_ext(final_path, token: str = ".__tmp__"):
    root, ext = os.path.splitext(str(final_path))
    ext = ext or ".mp4"
    return Path(f"{root}{token}{ext}")


def _atomic_replace(src, dst) -> None:
    _ensure_parent_dir(dst)
    try:
        os.replace(str(src), str(dst))
    except Exception:
        tmp_dst = Path(str(dst) + ".__swap__")
        shutil.copy2(str(src), str(tmp_dst))
        os.replace(str(tmp_dst), str(dst))
        try:
            os.remove(str(src))
        except FileNotFoundError:
            pass


def _write_videofile_safely(
    clip,
    final_path,
    *,
    fps: float,
    codec: str = "libx264",
    audio_codec: Optional[str] = "aac",
    bitrate: Optional[str] = "6M",
    threads: Optional[int] = None,
    extra_ffmpeg_params: Optional[List[str]] = None,
) -> None:
    """Write a clip to a temp path with the correct extension, then atomically replace."""
    _ensure_parent_dir(final_path)
    tmp_path = _temp_path_with_same_ext(final_path)
    if tmp_path.exists():
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass

    params = extra_ffmpeg_params or []
    merged_params = ["-f", "mp4", "-movflags", "+faststart", "-pix_fmt", "yuv420p"] + params

    clip.write_videofile(
        str(tmp_path),
        fps=fps,
        codec=codec,
        audio_codec=audio_codec,
        threads=threads or (os.cpu_count() or 2),
        preset="medium",
        bitrate=bitrate,
        ffmpeg_params=merged_params,
        verbose=False,
        logger=None,
    )
    _atomic_replace(tmp_path, final_path)
