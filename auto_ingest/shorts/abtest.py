"""A/B testing of thumbnails + titles for research-scripted shorts.

Generates alternate thumbnail/title variants per short so we can later measure
which wins. Variant state is tracked in a self-contained JSONL store
(``~/.config/auto-ingest/ab_variants.jsonl``) so this module does NOT depend on
the analytics agent's ``metrics.py`` / ``feedback.py`` (owned by another agent).

No network, no credentials. Render-side only.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

from PIL import Image, ImageDraw

from auto_ingest.shorts.models import Brief, PlannedShort

log = logging.getLogger("shorts.abtest")

VARIANT_STORE = Path(
    os.environ.get(
        "AB_VARIANTS_PATH",
        Path.home() / ".config" / "auto-ingest" / "ab_variants.jsonl",
    )
)


# --------------------------------------------------------------------------- #
# Thumbnail variants
# --------------------------------------------------------------------------- #
def make_thumbnail_variants(mp4: Union[str, Path], out_dir: Union[str, Path], *,
                            title: str, topic: str, n: int = 2) -> List[Path]:
    """Produce ``n`` visually DISTINCT thumbnail covers for ``mp4``.

    Variant 0 reuses :func:`thumbnail.make_thumbnail` (cyan pill). Variant 1+
    use a different accent colour (a magenta pill) so the covers are easy to
    tell apart in an A/B test. Files are named ``<stem>.v0.jpg``,
    ``<stem>.v1.jpg`` in ``out_dir``.

    Best-effort: any failure for a given variant is swallowed and only the
    successfully produced paths are returned.
    """
    from auto_ingest.shorts import thumbnail as thumb_mod

    mp4 = Path(mp4)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = mp4.stem
    produced: List[Path] = []

    for v in range(max(1, n)):
        try:
            if v == 0:
                out_path = out_dir / f"{stem}.v0.jpg"
                thumb_mod.make_thumbnail(
                    mp4, out_path, title=title, topic=topic)
                produced.append(out_path)
            else:
                out_path = out_dir / f"{stem}.v{v}.jpg"
                _make_alt_thumbnail(
                    mp4, out_path, title=title, topic=topic, accent=(255, 0, 200))
                produced.append(out_path)
        except Exception as e:  # pragma: no cover - environment dependent
            log.warning("Thumbnail variant %d failed for %s: %s", v, mp4, e)
    return produced


def _make_alt_thumbnail(mp4: Path, out_path: Path, *, title: str,
                        topic: str, accent: Tuple[int, int, int],
                        width: int = 1080, height: int = 1920) -> Path:
    """Render a 9:16 cover with a non-default accent pill (variant 1+).

    Mirrors :func:`thumbnail.make_thumbnail`'s layout but swaps the pill colour
    and places the hook text higher (different composition) so the variant
    reads as visually distinct.
    """
    from moviepy.editor import VideoFileClip

    from auto_ingest.shorts.thumbnail import (
        _best_frame,
        _load_font,
        _wrap,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with VideoFileClip(str(mp4)) as v:
        _, arr = _best_frame(v)
    img = Image.fromarray(arr).convert("RGB")
    img = img.resize((width, height))

    d = ImageDraw.Draw(img)
    d.rectangle([0, int(height * 0.55), width, height], fill=(0, 0, 0, 150))
    pill_font = _load_font(44)
    pill = f"#{topic.replace('_', ' ')}"
    pw = int(d.textlength(pill, font=pill_font)) + 40
    d.rounded_rectangle([40, int(height * 0.57), 40 + pw, int(height * 0.57) + 64],
                        radius=18, fill=(accent[0], accent[1], accent[2], 220))
    d.text((60, int(height * 0.57) + 10), pill, font=pill_font, fill=(0, 0, 0, 255))
    title_font = _load_font(72)
    lines = _wrap(d, (title or "").replace("_", " "), title_font, width - 120)
    y = int(height * 0.70)
    for ln in lines:
        d.text((60, y), ln, font=title_font, fill=(255, 255, 255, 255),
               stroke_width=3, stroke_fill=(0, 0, 0, 255))
        y += 84
    img.save(out_path)
    log.info("Alt thumbnail -> %s", out_path)
    return out_path


# --------------------------------------------------------------------------- #
# Title variants
# --------------------------------------------------------------------------- #
def title_variants(brief_or_short: Union[Brief, PlannedShort],
                   n: int = 2) -> List[str]:
    """Return ``n`` distinct title/hook candidates.

    Uses :func:`hook_bank.hook_for` with ``variant=0..n-1`` so each candidate is
    a rotated hook. Falls back to ``brief.title + hook`` (or the short's title)
    when no cue-driven hook is available.
    """
    from auto_ingest.shorts import hook_bank

    brief: Brief
    if isinstance(brief_or_short, PlannedShort):
        # Reconstruct a minimal Brief from the short for hook_bank.
        topic = brief_or_short.brief_topic or "topic"
        title = brief_or_short.title or topic
        brief = Brief(topic=topic, title=title, hook="", tags=[])
    else:
        brief = brief_or_short

    out: List[str] = []
    for v in range(max(1, n)):
        try:
            hook = hook_bank.hook_for(brief, variant=v)
        except Exception:
            hook = ""
        if hook:
            out.append(hook)
        else:
            base = (brief.title or brief.topic or "this").strip()
            if brief.hook:
                out.append(f"{base}: {brief.hook}")
            else:
                out.append(base)
    # De-duplicate while preserving order.
    seen = set()
    uniq: List[str] = []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


# --------------------------------------------------------------------------- #
# Variant plan store (JSONL)
# --------------------------------------------------------------------------- #
@dataclass
class VariantPlan:
    short_id: str
    platform: str
    thumb_variants: List[str] = field(default_factory=list)
    title_variants: List[str] = field(default_factory=list)
    active_variant: int = 0
    chosen_at: Optional[str] = None
    winner: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "VariantPlan":
        return cls(
            short_id=d["short_id"],
            platform=d["platform"],
            thumb_variants=list(d.get("thumb_variants", [])),
            title_variants=list(d.get("title_variants", [])),
            active_variant=int(d.get("active_variant", 0)),
            chosen_at=d.get("chosen_at"),
            winner=d.get("winner"),
        )


def _store_path() -> Path:
    return VARIANT_STORE


def _load_all() -> List[VariantPlan]:
    p = _store_path()
    if not p.exists():
        return []
    plans: List[VariantPlan] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            plans.append(VariantPlan.from_dict(json.loads(line)))
        except Exception:
            continue
    return plans


def _save_all(plans: List[VariantPlan]) -> None:
    p = _store_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for pl in plans:
            fh.write(json.dumps(pl.to_dict()) + "\n")
    tmp.replace(p)


def plan_variants(short_id: str, platform: str,
                  thumbs: List[Union[str, Path]],
                  titles: List[str]) -> VariantPlan:
    """Create (or replace) a :class:`VariantPlan` and persist it."""
    plan = VariantPlan(
        short_id=short_id,
        platform=platform,
        thumb_variants=[str(t) for t in thumbs],
        title_variants=[str(t) for t in titles],
        active_variant=0,
        chosen_at=None,
        winner=None,
    )
    plans = _load_all()
    plans = [pl for pl in plans
             if not (pl.short_id == short_id and pl.platform == platform)]
    plans.append(plan)
    _save_all(plans)
    log.info("Planned %d variants for %s/%s", len(plan.thumb_variants),
             short_id, platform)
    return plan


def load_variants(short_id: Optional[str] = None,
                  platform: Optional[str] = None) -> List[VariantPlan]:
    """Return stored plans, optionally filtered by short id / platform."""
    plans = _load_all()
    if short_id is not None:
        plans = [pl for pl in plans if pl.short_id == short_id]
    if platform is not None:
        plans = [pl for pl in plans if pl.platform == platform]
    return plans


def choose_winner(short_id: str, platform: str, variant: int) -> Optional[VariantPlan]:
    """Record ``variant`` as the winning variant for ``short_id``/``platform``.

    Returns the updated plan, or ``None`` if no matching plan exists.
    """
    plans = _load_all()
    found = None
    for pl in plans:
        if pl.short_id == short_id and pl.platform == platform:
            pl.winner = int(variant)
            pl.chosen_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            found = pl
            break
    if found is None:
        log.warning("No variant plan for %s/%s", short_id, platform)
        return None
    _save_all(plans)
    log.info("Winner -> variant %d for %s/%s", variant, short_id, platform)
    return found


def assign_variant(item, plan: VariantPlan) -> tuple:
    """Pure helper: return ``(thumbnail_path, title)`` for the active variant.

    ``item`` should expose ``thumbnail_path`` and a ``title``/``key`` (duck-typed
    so it works with a ``QueueItem`` or similar). The uploader can call this to
    pick the thumbnail + title for the active A/B variant.
    """
    v = plan.active_variant
    thumbs = plan.thumb_variants
    titles = plan.title_variants
    thumb = thumbs[v] if 0 <= v < len(thumbs) else (thumbs[0] if thumbs else None)
    title = titles[v] if 0 <= v < len(titles) else None
    if title is None:
        title = getattr(item, "title", None) or getattr(item, "key", None)
    return thumb, title
