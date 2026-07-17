"""Auto-thumbnail generation for narrated shorts.

Given a rendered MP4, pick a legible, high-contrast frame and burn the hook
text + a topic-colored pill, producing a 9:16 cover (and a 16:9 variant for
YouTube). Used as the video cover on TikTok/YouTube/Instagram so the short
survives muted autoplay and reads as a promise.

Render-side only: no Neo4j, no network.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

log = logging.getLogger("shorts.thumbnail")

_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(_FONT, size)
    except Exception:
        return ImageFont.load_default()


def _frame_contrast(arr: np.ndarray) -> float:
    """Variance-based 'interestingness' proxy for frame selection."""
    gray = arr[..., :3].mean(axis=2)
    return float(gray.std())


def _best_frame(vid, *, n_samples: int = 12) -> Tuple[int, np.ndarray]:
    dur = float(vid.duration or 0.0) or 1.0
    best_t, best_arr, best_c = 0.0, None, -1.0
    for i in range(n_samples):
        t = dur * (i + 0.5) / n_samples
        try:
            arr = vid.get_frame(min(t, dur - 0.01))
        except Exception:
            continue
        c = _frame_contrast(arr)
        if c > best_c:
            best_c, best_t, best_arr = c, t, arr
    if best_arr is None:  # fallback: first frame
        best_arr = vid.get_frame(0.0)
    return int(best_t * float(getattr(vid, "fps", 30) or 30)), best_arr


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, max_w: int) -> list:
    words = text.split()
    lines, cur = [], []
    for w in words:
        test = " ".join(cur + [w])
        if draw.textlength(test, font=font) <= max_w or not cur:
            cur.append(w)
        else:
            lines.append(" ".join(cur))
            cur = [w]
    if cur:
        lines.append(" ".join(cur))
    return lines[:3]


def make_thumbnail(mp4: Path, out_path: Path, *,
                   title: str, topic: str,
                   width: int = 1080, height: int = 1920) -> Path:
    """Write a 9:16 cover (and a 16:9 YouTube variant) for ``mp4``."""
    from moviepy.editor import VideoFileClip

    mp4 = Path(mp4)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with VideoFileClip(str(mp4)) as v:
        _, arr = _best_frame(v)
    img = Image.fromarray(arr).convert("RGB")
    img = img.resize((width, height))

    d = ImageDraw.Draw(img)
    # Darken bottom third for text legibility.
    d.rectangle([0, int(height * 0.62), width, height], fill=(0, 0, 0, 150))
    # Topic pill.
    pill_font = _load_font(44)
    pill = f"#{topic.replace('_', ' ')}"
    pw = int(d.textlength(pill, font=pill_font)) + 40
    d.rounded_rectangle([40, int(height * 0.64), 40 + pw, int(height * 0.64) + 64],
                        radius=18, fill=(0, 200, 255, 220))
    d.text((60, int(height * 0.64) + 10), pill, font=pill_font, fill=(0, 0, 0, 255))
    # Hook text (big, high-contrast).
    title_font = _load_font(72)
    lines = _wrap(d, (title or "").replace("_", " "), title_font, width - 120)
    y = int(height * 0.76)
    for ln in lines:
        d.text((60, y), ln, font=title_font, fill=(255, 255, 255, 255),
               stroke_width=3, stroke_fill=(0, 0, 0, 255))
        y += 84
    img.save(out_path)

    # 16:9 YouTube variant.
    yt = out_path.with_name(out_path.stem + "_16x9" + out_path.suffix)
    Image.fromarray(arr).convert("RGB").resize((1280, 720)).save(yt)
    log.info("Thumbnail -> %s (+%s)", out_path, yt)
    return out_path


if __name__ == "__main__":
    import sys
    p = Path(sys.argv[1])
    make_thumbnail(p, p.with_suffix(".thumb.jpg"), title=p.stem, topic="research")
