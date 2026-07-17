"""Virality scorer for planned research shorts.

A lightweight, explainable heuristic that scores a :class:`Plan` (or a single
:class:`PlannedShort`) on the factors that actually drive shorts retention and
CTR, so we can rank candidates *before* spending render/Neo4j budget:

  * Hook strength      - does the cold-open pose a gap / contradiction / claim?
  * Retention curve    - curiosity gaps, callbacks and a payoff spaced well?
  * Pacing             - cue density, no dead air, no cramming?
  * CTR / thumbnail     - has a high-contrast hook line + a branded lockup?
  * Shareability       - myth/fact or a surprising "reveal" present?

Returns a :class:`ViralityScore` (0-100) with a per-factor breakdown and
actionable ``suggestions``. Pure over the model; no Neo4j / no moviepy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from auto_ingest.shorts.models import PlannedShort

# Words that signal a "curiosity gap" / tension in a hook.
_GAP_WORDS = (
    "wrong", "mistake", "myth", "lie", "secret", "twist", "catch", "trick",
    "surprising", "nobody", "everyone", "they", "don't", "never", "actually",
    "real", "truth", "why", "what if", "until", "hidden", "myster", "shock",
    "debunk", "fixed", "broke", "changed", "miss",
)
_NUMERIC = tuple("0123456789")


@dataclass
class ViralityScore:
    total: float
    factors: Dict[str, float]
    suggestions: List[str]

    def grade(self) -> str:
        if self.total >= 80:
            return "A"
        if self.total >= 65:
            return "B"
        if self.total >= 50:
            return "C"
        if self.total >= 35:
            return "D"
        return "F"

    def to_dict(self) -> Dict[str, object]:
        return {
            "total": round(self.total, 1),
            "grade": self.grade(),
            "factors": {k: round(v, 1) for k, v in self.factors.items()},
            "suggestions": list(self.suggestions),
        }


def _has_gap(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in _GAP_WORDS)


def _hook_strength(cues) -> "tuple[float, List[str]]":
    """Score the cold-open hook on tension + specificity."""
    hooks = [c for c in cues if c.kind == "hook"]
    if not hooks:
        return 0.0, ["Add a hook cue (cold-open)."]
    h = hooks[0].text
    score = 30.0
    tips: List[str] = []
    if _has_gap(h):
        score += 30.0
    else:
        tips.append("Hook lacks a curiosity gap (add 'wrong'/'secret'/'nobody' tension).")
    if any(ch in h for ch in _NUMERIC):
        score += 10.0
    if len(h.split()) <= 9:
        score += 10.0
    else:
        tips.append("Hook is long; trim to <=9 words for a punchy cold-open.")
    if h and h[0].isupper():
        score += 5.0
    return min(100.0, score), tips


def _retention(cues, duration: float) -> "tuple[float, List[str]]":
    """Reward a mid-video callback + a payoff, spaced across the runtime."""
    score = 20.0
    tips: List[str] = []
    has_callback = any(c.kind == "callback" for c in cues)
    has_payoff = any(c.kind == "payoff" for c in cues)
    has_reveal = any(c.kind == "reveal" for c in cues)
    if has_callback:
        score += 20.0
    else:
        tips.append("Add a mid-video callback to hold retention.")
    if has_payoff:
        score += 20.0
    else:
        tips.append("Add an end payoff cue (curiosity reward).")
    if has_reveal:
        score += 15.0
    if has_payoff:
        pay = [c for c in cues if c.kind == "payoff"][-1]
        frac = pay.start / duration if duration else 1.0
        if frac < 0.7:
            score -= 10.0
            tips.append("Move the payoff later (final 25% of the video).")
    pts = sorted((c.start, c.end) for c in cues)
    gap = 0.0
    cur_end = 0.0
    for s, e in pts:
        if s > cur_end + gap:
            gap = max(gap, s - cur_end)
        cur_end = max(cur_end, e)
    if gap > 4.0:
        score -= 10.0
        tips.append(f"~{gap:.0f}s of dead air between cues; tighten spacing.")
    return max(0.0, min(100.0, score)), tips


def _pacing(cues, duration: float) -> "tuple[float, List[str]]":
    """Cue density: not too sparse, not crammed. Target ~1 cue / 3-4s."""
    if duration <= 0:
        return 50.0, []
    density = len(cues) / duration
    score = 70.0
    tips: List[str] = []
    if density < 0.2:
        score -= 25.0
        tips.append("Too few cues; video feels empty. Add points.")
    elif density > 0.4:
        score -= 20.0
        tips.append("Crammed; reduce cue count or lengthen the video.")
    long_cues = [c for c in cues if c.end - c.start > 6.0]
    if long_cues:
        score -= 10.0
        tips.append(f"{len(long_cues)} cue(s) on screen >6s; split or shorten.")
    return max(0.0, min(100.0, score)), tips


def _ctr(cues) -> "tuple[float, List[str]]":
    """Thumbnail/CTR: a short, gap-bearing hook line + branded lockup."""
    hooks = [c for c in cues if c.kind == "hook"]
    score = 40.0
    tips: List[str] = []
    if hooks and len(hooks[0].text.split()) <= 7 and _has_gap(hooks[0].text):
        score += 40.0
    elif hooks:
        score += 15.0
        tips.append("Hook line is CTR-weak; make it <7 words with a gap for the thumbnail.")
    else:
        tips.append("No hook line to burn onto the thumbnail.")
    if any(c.kind == "series" for c in cues):
        score += 10.0  # branded series lockup aids recall
    return min(100.0, score), tips


def _shareability(cues) -> "tuple[float, List[str]]":
    """Myth/fact or a reveal reads as inherently shareable."""
    score = 40.0
    tips: List[str] = []
    kinds = {c.kind for c in cues}
    if "myth" in kinds and "fact" in kinds:
        score += 40.0
    elif "reveal" in kinds:
        score += 30.0
    else:
        tips.append("Add a myth/fact or reveal beat to boost shareability.")
    return min(100.0, score), tips


_WEIGHTS = {
    "hook": 0.25,
    "retention": 0.25,
    "pacing": 0.15,
    "ctr": 0.20,
    "share": 0.15,
}


def score_short(item: PlannedShort) -> ViralityScore:
    """Score one :class:`PlannedShort` (0-100)."""
    cues = item.cues
    duration = item.duration()
    fh, th = _hook_strength(cues)
    fr, tr = _retention(cues, duration)
    fp, tp = _pacing(cues, duration)
    fc, tc = _ctr(cues)
    fs, ts = _shareability(cues)
    factors = {
        "hook": fh,
        "retention": fr,
        "pacing": fp,
        "ctr": fc,
        "share": fs,
    }
    total = sum(factors[k] * _WEIGHTS[k] for k in _WEIGHTS)
    suggestions = th + tr + tp + tc + ts
    return ViralityScore(total=total, factors=factors, suggestions=suggestions)


def score_plan(plan) -> Dict[str, object]:
    """Score every short in a plan; return per-short + aggregate ranking."""
    scored = []
    for s in plan.shorts:
        vs = score_short(s)
        scored.append((s.id, vs))
    shorts_out = {sid: vs.to_dict() for sid, vs in scored}
    ranked = [sid for sid, _ in sorted(scored, key=lambda x: x[1].total, reverse=True)]
    totals = [vs.total for _, vs in scored]
    return {
        "shorts": shorts_out,
        "ranked": ranked,
        "mean": round(sum(totals) / len(totals), 1) if totals else 0.0,
        "best": round(max(totals), 1) if totals else 0.0,
        "worst": round(min(totals), 1) if totals else 0.0,
    }


def best_short(plan) -> Optional[str]:
    """Id of the highest-virality short in a plan, or None if empty."""
    res = score_plan(plan)
    return res["ranked"][0] if res["ranked"] else None
