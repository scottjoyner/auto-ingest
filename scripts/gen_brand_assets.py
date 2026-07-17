"""Generate brand assets (avatar + YouTube banner) in the repo.

Reproducible brand visuals for the AI/ML research-shorts accounts, matching the
video-thumbnail palette (cyan pill, ink background, white text). Run:

    python3 scripts/gen_brand_assets.py --all
    python3 scripts/gen_brand_assets.py --avatar
    python3 scripts/gen_brand_assets.py --banner

Outputs into docs/brand/. No network, no credentials.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "brand"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

INK = (11, 14, 20)
CYAN = (0, 200, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

HANDLE = "Research in the Fast Lane"
TAGLINE = "AI/ML papers, explained before you scroll."


def _font(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(FONT, size)
    except Exception:
        return ImageFont.load_default()


def make_avatar(out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    size = 1000
    img = Image.new("RGB", (size, size), INK)
    d = ImageDraw.Draw(img)
    # Cyan ring.
    d.ellipse([60, 60, size - 60, size - 60], outline=CYAN, width=24)
    # Centered monogram.
    f = _font(520)
    txt = "R"
    bbox = d.textbbox((0, 0), txt, font=f)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.text(((size - tw) / 2 - bbox[0], (size - th) / 2 - bbox[1]), txt,
           font=f, fill=WHITE, stroke_width=6, stroke_fill=BLACK)
    img.save(out)
    return out


def make_banner(out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    w, h = 2560, 1440
    img = Image.new("RGB", (w, h), INK)
    d = ImageDraw.Draw(img)
    # Subtle cyan baseline rule (evokes a "lane").
    d.line([(0, int(h * 0.72)), (w, int(h * 0.72))], fill=CYAN, width=8)
    # Safe-area centered text.
    name_f = _font(140)
    bbox = d.textbbox((0, 0), HANDLE, font=name_f)
    nw = bbox[2] - bbox[0]
    d.text(((w - nw) / 2 - bbox[0], int(h * 0.40)), HANDLE, font=name_f,
           fill=WHITE, stroke_width=4, stroke_fill=BLACK)
    tag_f = _font(64)
    bbox = d.textbbox((0, 0), TAGLINE, font=tag_f)
    tw = bbox[2] - bbox[0]
    d.text(((w - tw) / 2 - bbox[0], int(h * 0.40) + 170), TAGLINE, font=tag_f,
           fill=CYAN)
    img.save(out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--avatar", action="store_true")
    ap.add_argument("--banner", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    do_all = args.all or not (args.avatar or args.banner)
    if args.avatar or do_all:
        p = make_avatar(OUT_DIR / "avatar.png")
        print("avatar ->", p)
    if args.banner or do_all:
        p = make_banner(OUT_DIR / "banner_youtube.png")
        print("banner ->", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
