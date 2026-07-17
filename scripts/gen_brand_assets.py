"""Generate brand assets (avatar + YouTube banner) in the repo.

Reproducible brand visuals for the AI/ML research-shorts accounts, matching the
video-thumbnail palette (cyan pill, ink background, white text). Run:

    python3 scripts/gen_brand_assets.py --all
    python3 scripts/gen_brand_assets.py --avatar
    python3 scripts/gen_brand_assets.py --banner
    python3 scripts/gen_brand_assets.py --instagram   # IG profile picture
    python3 scripts/gen_brand_assets.py --tiktok      # TikTok avatar
    python3 scripts/gen_brand_assets.py --check       # verify assets vs manifest

Outputs into docs/brand/. No network, no credentials.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "brand"
MANIFEST = OUT_DIR / "brand_manifest.json"
BRAND_SPEC = OUT_DIR / "brand_spec.md"
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


def _make_square_monogram(out: Path, size: int) -> Path:
    """Square on-brand avatar: ink bg, cyan monogram + subtle cyan ring.

    Shared renderer for the Instagram profile picture and TikTok avatar so both
    platforms stay visually identical to the YouTube avatar while being sized
    for their own crops (IG circle / TikTok circle).
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (size, size), INK)
    d = ImageDraw.Draw(img)
    pad = int(size * 0.06)
    ring = max(6, int(size * 0.024))
    d.ellipse([pad, pad, size - pad, size - pad], outline=CYAN, width=ring)
    f = _font(int(size * 0.52))
    txt = "R"
    bbox = d.textbbox((0, 0), txt, font=f)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    # Cyan monogram (brand accent) with a black stroke for contrast.
    d.text(((size - tw) / 2 - bbox[0], (size - th) / 2 - bbox[1]), txt,
           font=f, fill=CYAN, stroke_width=max(3, int(size * 0.006)),
           stroke_fill=BLACK)
    img.save(out)
    return out


def make_instagram(out: Path) -> Path:
    """Instagram profile picture (square 640×640, safe for circle crop)."""
    return _make_square_monogram(out, 640)


def make_tiktok(out: Path) -> Path:
    """TikTok avatar (square 720×720, safe for circle crop)."""
    return _make_square_monogram(out, 720)


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


def _load_manifest(brand_dir: Path) -> dict:
    return json.loads((brand_dir / "brand_manifest.json").read_text(encoding="utf-8"))


def _color_present(img: "Image.Image", target: tuple, tol: int = 24,
                   min_frac: float = 0.001) -> bool:
    """True if at least ``min_frac`` of pixels are within ``tol`` of ``target``.

    Samples at moderate resolution (cap the long edge at 256px) so thin accents
    like the banner's cyan rule survive downscaling, while staying fast.
    """
    rgb = img.convert("RGB")
    w, h = rgb.size
    long_edge = max(w, h)
    if long_edge > 256:
        scale = 256 / long_edge
        rgb = rgb.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    getter = getattr(rgb, "get_flattened_data", None)
    px = list(getter()) if getter else list(rgb.getdata())
    hits = sum(1 for r, g, b in px
               if abs(r - target[0]) <= tol
               and abs(g - target[1]) <= tol
               and abs(b - target[2]) <= tol)
    return hits >= max(1, int(len(px) * min_frac))


def check_assets(brand_dir: Path) -> "tuple[bool, list]":
    """Verify generated assets use the manifest palette + handles match brand_spec.

    Returns ``(ok, issues)``. Pure: no network. For each referenced image asset
    we sample pixels and confirm the manifest's ink (background) and cyan
    (accent) colors are actually present within tolerance. Then we confirm every
    handle in the manifest also appears verbatim in ``brand_spec.md``.
    """
    brand_dir = Path(brand_dir)
    issues: list = []
    manifest_path = brand_dir / "brand_manifest.json"
    if not manifest_path.exists():
        return False, [f"missing brand_manifest.json at {manifest_path}"]
    data = _load_manifest(brand_dir)
    palette = data.get("palette", {})
    ink = tuple(palette.get("ink", INK))
    cyan = tuple(palette.get("cyan", CYAN))

    assets = data.get("assets", {})
    if not assets:
        issues.append("manifest has no assets to check")
    for key, rel in assets.items():
        p = brand_dir / rel
        if not p.exists():
            issues.append(f"asset missing: {p}")
            continue
        try:
            with Image.open(p) as im:
                im.load()
                if not _color_present(im, ink, tol=30, min_frac=0.05):
                    issues.append(f"{rel}: manifest ink {ink} not found (bg mismatch)")
                if not _color_present(im, cyan, tol=40, min_frac=0.0005):
                    issues.append(f"{rel}: manifest cyan {cyan} not found (accent mismatch)")
        except Exception as e:
            issues.append(f"{rel}: not a readable image ({e})")

    # Handles in manifest must be declared in brand_spec.md.
    spec_path = brand_dir / "brand_spec.md"
    if spec_path.exists():
        spec = spec_path.read_text(encoding="utf-8")
        for plat, handle in (data.get("handles") or {}).items():
            if handle and handle not in spec:
                issues.append(f"handle {handle!r} ({plat}) not declared in brand_spec.md")
    else:
        issues.append("brand_spec.md missing (cannot verify handles)")

    return (not issues), issues


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--avatar", action="store_true")
    ap.add_argument("--banner", action="store_true")
    ap.add_argument("--instagram", action="store_true")
    ap.add_argument("--tiktok", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="Verify generated assets use the manifest palette and "
                         "that handles match brand_spec.md")
    ap.add_argument("--brand-dir", type=Path, default=OUT_DIR,
                    help="Override the brand asset directory (for tests)")
    args = ap.parse_args()

    brand_dir = args.brand_dir
    if args.check:
        ok, issues = check_assets(brand_dir)
        if ok:
            print(f"OK: brand assets match manifest palette + handles ({brand_dir})")
            return 0
        for iss in issues:
            print(f"[mismatch] {iss}")
        print(f"{len(issues)} mismatch(es) found")
        return 1

    do_all = args.all or not (args.avatar or args.banner or args.instagram
                              or args.tiktok)
    if args.avatar or do_all:
        p = make_avatar(brand_dir / "avatar.png")
        print("avatar ->", p)
    if args.banner or do_all:
        p = make_banner(brand_dir / "banner_youtube.png")
        print("banner ->", p)
    if args.instagram or do_all:
        p = make_instagram(brand_dir / "avatar_instagram.png")
        print("instagram ->", p)
    if args.tiktok or do_all:
        p = make_tiktok(brand_dir / "avatar_tiktok.png")
        print("tiktok ->", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
