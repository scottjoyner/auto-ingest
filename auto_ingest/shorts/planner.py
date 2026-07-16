"""Plan research shorts from a :class:`Brief`.

Planning splits a brief into N shorts. Each short gets:
  * a timed **cue** track (caption lines) built from the hook + a subset of
    points + a source line, spread across the short's duration;
  * a **shot** list of highway B-roll pulled from the curated anchor pool.

Planning is deterministic given ``seed`` so a plan is reproducible, and the
``iterate`` step can regenerate with a new seed / different short count to
explore variations. Nothing here touches moviepy — it is pure scheduling.
"""
from __future__ import annotations

import logging
import uuid
from typing import Dict, List, Optional

from auto_ingest.shorts.models import Brief, Cue, Plan, PlannedShort, Shot

log = logging.getLogger("shorts.planner")

SECS_PER_CUE = 3.2          # approx on-screen time per caption line
HOOK_KIND = "hook"
POINT_KIND = "point"
SOURCE_KIND = "source"


def _distribute_cues(title: str, hook: str, points: List[str], sources: List,
                     duration: float) -> List[Cue]:
    """Build timed cues across ``duration`` seconds."""
    lines: List[tuple] = [(HOOK_KIND, hook)] if hook else []
    lines += [(POINT_KIND, p) for p in points]
    if sources:
        src = "Sources: " + ", ".join(f"[{i + 1}] {s.title}" for i, s in enumerate(sources[:4]))
        lines.append((SOURCE_KIND, src))

    if not lines:
        return []

    n = len(lines)
    # Evenly space cues; last cue ends near duration. Add a small lead-in.
    lead = 0.4
    span = max(duration - lead, 1.0)
    step = span / n
    cues: List[Cue] = []
    for i, (kind, text) in enumerate(lines):
        start = lead + i * step
        end = start + min(step, SECS_PER_CUE)
        cues.append(Cue(start=round(start, 2), end=round(end, 2), text=text, kind=kind))
    return cues


def plan_shorts(brief: Brief, anchors: List[Dict[str, object]], *,
                short_count: int = 3, short_dur: float = 30.0,
                shots_per_short: int = 3, seed: int = 1,
                clip_dur: float = 6.0, min_gap_sec: float = 8.0,
                topic_prefix: str = "") -> Plan:
    """Create a :class:`Plan` of ``short_count`` shorts from a brief."""
    import random
    rnd = random.Random(seed)

    # Choose how many points per short and rotate through them.
    points = list(brief.points) or [brief.hook]
    if not points:
        points = ["(no points curated)"]

    shorts: List[PlannedShort] = []
    for s_idx in range(short_count):
        # rotate a window of points for this short
        start = (s_idx * 2) % max(len(points), 1)
        window = points[start:start + 3] or points[:1]

        cues = _distribute_cues(
            title=brief.title,
            hook=brief.hook if s_idx == 0 else "",
            points=window,
            sources=brief.sources if s_idx == short_count - 1 else [],
            duration=short_dur,
        )

        shots = _pick_shots_deterministic(
            anchors, count=shots_per_short, min_gap_sec=min_gap_sec,
            clip_dur=clip_dur, rng=rnd,
        )

        title = brief.title
        if short_count > 1:
            title = f"{topic_prefix}{brief.title} — Part {s_idx + 1}"

        shorts.append(PlannedShort(
            id=uuid.uuid5(uuid.NAMESPACE_URL, f"{brief.topic}:{seed}:{s_idx}").hex[:10],
            brief_topic=brief.topic,
            title=title,
            cues=cues,
            shots=[Shot(**s) for s in shots],
            notes=f"points[{start}:{start + len(window)}]",
            status="planned",
        ))

    plan = Plan(topic=brief.topic, brief=brief, shorts=shorts, iteration=1)
    log.info("Planned %d shorts for %s (dur=%.0fs, seed=%d)",
             short_count, brief.topic, short_dur, seed)
    return plan


def _pick_shots_deterministic(anchors, *, count, min_gap_sec, clip_dur, rng) -> List[Dict]:
    from auto_ingest.shorts.backdrop import pick_shots
    seed = int(rng.random() * 1e9)
    return pick_shots(anchors, count=count, min_gap_sec=min_gap_sec,
                      clip_dur=clip_dur, rng_seed=seed)


def iterate_plan(prev: Plan, anchors: List[Dict[str, object]], *,
                 short_count: Optional[int] = None, short_dur: Optional[float] = None,
                 seed: Optional[int] = None, shots_per_short: Optional[int] = None,
                 clip_dur: Optional[float] = None, min_gap_sec: Optional[float] = None,
                 reject_ids: Optional[List[str]] = None) -> Plan:
    """Produce the next iteration of a plan (new seed / counts, drop rejects)."""
    reject = set(reject_ids or [])
    kept = [s for s in prev.shorts if s.id not in reject and s.status != "rejected"]
    # Carry over rendered shorts so we don't re-plan what already exists.
    carried = [s for s in kept if s.status == "rendered"]

    new_count = short_count or max(1, len(prev.shorts) - len(carried))
    new_seed = seed if seed is not None else prev.iteration + 7

    fresh = plan_shorts(
        prev.brief,
        anchors,
        short_count=new_count,
        short_dur=short_dur or _max_dur(prev),
        shots_per_short=shots_per_short or _max_shots(prev),
        seed=new_seed,
        clip_dur=clip_dur or 6.0,
        min_gap_sec=min_gap_sec or 8.0,
    )
    fresh.shorts = carried + fresh.shorts
    fresh.iteration = prev.iteration + 1
    fresh.updated_at = __import__("time").time()
    return fresh


def _max_dur(plan: Plan) -> float:
    return max((s.duration() for s in plan.shorts), default=30.0) or 30.0


def _max_shots(plan: Plan) -> int:
    return max((len(s.shots) for s in plan.shorts), default=3) or 3
