"""Learning / feedback loop over recorded shorts metrics (pure, no network).

Compares predicted virality (from :mod:`auto_ingest.shorts.virality`) against
actual performance, ranks hooks / topics / platforms, and emits concrete,
data-derived suggestions for the next batch of shorts.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from auto_ingest.shorts.metrics import MetricsRecord

log = logging.getLogger("shorts.feedback")

_GAP_WORDS = (
    "wrong", "mistake", "myth", "lie", "secret", "twist", "catch", "trick",
    "surprising", "nobody", "everyone", "actually", "real", "truth", "why",
    "mystery", "shock", "debunk",
)


def _num(v: Optional[float]) -> float:
    return v if isinstance(v, (int, float)) else 0.0


def _avg(values: List[Optional[float]]) -> Optional[float]:
    nums = [v for v in values if isinstance(v, (int, float))]
    if not nums:
        return None
    return sum(nums) / len(nums)


@dataclass
class _HookInfo:
    text: str
    views: List[float]
    retention: List[float]


def feedback_report(records: List[MetricsRecord]) -> dict:
    """Compare predicted virality vs actuals; rank hooks/topics/platforms.

    Returns a plain dict with: ``by_platform`` (avg views, avg retention),
    ``top_hooks`` (hook texts with best avg_view_pct), ``weak_hooks``,
    ``topic_perf`` (topic -> avg views), and ``pred_vs_actual`` (mean abs error
    of virality vs a normalized views score).
    """
    if not records:
        return {
            "by_platform": {},
            "top_hooks": [],
            "weak_hooks": [],
            "topic_perf": {},
            "pred_vs_actual": {"mean_abs_error": None, "n": 0},
        }

    # Per-platform aggregates.
    by_platform: Dict[str, dict] = {}
    for r in records:
        agg = by_platform.setdefault(r.platform, {"views": [], "retention": []})
        if r.views is not None:
            agg["views"].append(float(r.views))
        if r.avg_view_pct is not None:
            agg["retention"].append(float(r.avg_view_pct))
    by_platform_out = {
        p: {"avg_views": round(_avg(a["views"]) or 0.0, 1),
            "avg_retention": round(_avg(a["retention"]) or 0.0, 1)}
        for p, a in by_platform.items()
    }

    # Hook text -> performance. Hook text is heuristic: we have topic, so we
    # use the topic string; if a topic reads like a gap-hook we group it.
    hook_map: Dict[str, _HookInfo] = {}
    for r in records:
        label = r.topic or r.short_id
        info = hook_map.setdefault(label, _HookInfo(text=label, views=[], retention=[]))
        if r.views is not None:
            info.views.append(float(r.views))
        if r.avg_view_pct is not None:
            info.retention.append(float(r.avg_view_pct))
    hook_rows = []
    for info in hook_map.values():
        av = _avg(info.views)
        ar = _avg(info.retention)
        hook_rows.append({
            "text": info.text,
            "avg_views": round(av, 1) if av is not None else None,
            "avg_view_pct": round(ar, 1) if ar is not None else None,
        })
    hook_rows.sort(key=lambda x: (x["avg_view_pct"] or 0.0), reverse=True)
    top_hooks = [h for h in hook_rows if h["avg_view_pct"] is not None][:5]
    weak_hooks = list(reversed([h for h in hook_rows
                                if h["avg_view_pct"] is not None][-5:]))

    # Topic performance.
    topic_perf: Dict[str, float] = {}
    for r in records:
        if r.views is None:
            continue
        topic_perf.setdefault(r.topic, []).append(float(r.views))
    topic_perf_out = {
        t: round(_avg(v) or 0.0, 1) for t, v in topic_perf.items()
    }

    # Predicted vs actual: normalize views to 0-100 (vs max) and compare MAE.
    view_vals = [r.views for r in records if r.views is not None
                 and r.pred_virality is not None]
    max_views = max((_num(v) for v in view_vals), default=0.0)
    errs: List[float] = []
    for r in records:
        if r.views is None or r.pred_virality is None or max_views == 0.0:
            continue
        norm = 100.0 * _num(r.views) / max_views
        errs.append(abs(norm - _num(r.pred_virality)))
    mae = round(_avg(errs), 1) if errs else None
    pred_vs_actual = {"mean_abs_error": mae, "n": len(errs)}

    return {
        "by_platform": by_platform_out,
        "top_hooks": top_hooks,
        "weak_hooks": weak_hooks,
        "topic_perf": topic_perf_out,
        "pred_vs_actual": pred_vs_actual,
    }


def suggest_next(recs: List[MetricsRecord]) -> List[str]:
    """Human-readable, data-derived suggestions for the next batch."""
    if not recs:
        return ["Not enough data yet — ingest metrics after a few posts."]
    rep = feedback_report(recs)

    suggestions: List[str] = []

    by_plat = rep["by_platform"]
    if len(by_plat) >= 2:
        best_plat = max(by_plat, key=lambda p: by_plat[p]["avg_retention"])
        worst_plat = min(by_plat, key=lambda p: by_plat[p]["avg_retention"])
        if by_plat[best_plat]["avg_retention"] > by_plat[worst_plat]["avg_retention"] + 5:
            suggestions.append(
                f"{worst_plat} retention ({by_plat[worst_plat]['avg_retention']}%) "
                f"< {best_plat} ({by_plat[best_plat]['avg_retention']}%); "
                f"shorten {worst_plat} scripts.")

    top = rep["top_hooks"]
    weak = rep["weak_hooks"]
    if top and weak:
        t = top[0]["avg_views"] or 0.0
        w = weak[0]["avg_views"] or 0.0
        if t > 0 and w > 0 and t >= 2 * w:
            label = top[0]["text"]
            suggestions.append(
                f"Hooks/topics like {label!r} outperformed weak ones by "
                f"{t / w:.1f}x views; use more.")

    # Gap-word hooks vs non-gap.
    gap_views: List[float] = []
    plain_views: List[float] = []
    for r in recs:
        if r.views is None:
            continue
        if any(w in (r.topic or "").lower() for w in _GAP_WORDS):
            gap_views.append(float(r.views))
        else:
            plain_views.append(float(r.views))
    g = _avg(gap_views)
    p = _avg(plain_views)
    if g is not None and p is not None and g > p + 1:
        mult = g / p if p > 0 else 0
        word = "curiosity-gap" if mult >= 1 else "non-gap"
        suggestions.append(
            f"Curiosity-gap topics averaged {mult:.1f}x the views of plain ones; "
            f"lead with {word} hooks.")

    tp = rep["topic_perf"]
    if len(tp) >= 2:
        best_t = max(tp, key=lambda k: tp[k])
        worst_t = min(tp, key=lambda k: tp[k])
        if tp[best_t] > tp[worst_t] * 1.5:
            suggestions.append(
                f"Topic {best_t!r} is your strongest performer "
                f"({tp[best_t]} avg views) — make it a recurring series.")

    pva = rep["pred_vs_actual"]
    if pva["mean_abs_error"] is not None:
        if pva["mean_abs_error"] > 25:
            suggestions.append(
                f"Virality predictions are off by ~{pva['mean_abs_error']} pts "
                f"(MAE); re-tune weights in virality.py for better pre-publish ranking.")
        else:
            suggestions.append(
                f"Virality prediction tracks actuals well (MAE ~{pva['mean_abs_error']} pts).")

    if not suggestions:
        suggestions.append("Data is thin — keep posting and re-ingest for sharper suggestions.")
    return suggestions


# --------------------------------------------------------------------------- #
# Concrete, thresholded actions (P-G9)
# --------------------------------------------------------------------------- #
# Thresholds are deliberate + documented so operators can tune them. They are
# expressed on the same units the platform exports use (CTR as a percentage,
# retention as avg_view_pct percentage, views as raw counts).
ACTION_CTR_RERENDER = 2.0        # CTR (%) below this -> re-render (new thumb/hook)
ACTION_RETENTION_PROMOTE = 60.0  # retention (%) above this -> promote the hook
ACTION_LOW_VIEWS = 100           # avg views below this for a topic -> pause topic


@dataclass
class Action:
    kind: str            # rerender | promote | pause
    target: str          # short_id / hook text / topic
    reason: str          # human-readable threshold explanation
    metric: Optional[float] = None

    def to_dict(self) -> Dict[str, object]:
        return {"kind": self.kind, "target": self.target,
                "reason": self.reason, "metric": self.metric}

    def __str__(self) -> str:  # pragma: no cover - trivial formatting
        return f"[{self.kind}] {self.target}: {self.reason}"


def suggest_actions(records: List[MetricsRecord]) -> List[Action]:
    """Emit concrete, deterministic, thresholded actions from raw metrics.

    Pure (no network). Rules:
      * re-render a short whose CTR < ``ACTION_CTR_RERENDER``%
      * promote a short/hook whose retention (avg_view_pct) >
        ``ACTION_RETENTION_PROMOTE``%
      * pause a topic whose average views < ``ACTION_LOW_VIEWS``

    Deterministic ordering: rerender, then promote, then pause; each group
    sorted by target for stable output.
    """
    rerender: List[Action] = []
    promote: List[Action] = []

    # Per-short CTR / retention rules.
    for r in records:
        if r.ctr is not None and r.ctr < ACTION_CTR_RERENDER:
            rerender.append(Action(
                kind="rerender", target=r.short_id, metric=round(float(r.ctr), 2),
                reason=f"CTR {r.ctr:.2f}% < {ACTION_CTR_RERENDER}% "
                       f"(re-render short with a stronger thumbnail/hook)"))
        if r.avg_view_pct is not None and r.avg_view_pct > ACTION_RETENTION_PROMOTE:
            label = r.topic or r.short_id
            promote.append(Action(
                kind="promote", target=label,
                metric=round(float(r.avg_view_pct), 1),
                reason=f"retention {r.avg_view_pct:.1f}% > "
                       f"{ACTION_RETENTION_PROMOTE}% (promote hook / make a series)"))

    # Per-topic average-views rule (pause underperformers).
    topic_views: Dict[str, List[float]] = {}
    for r in records:
        if r.views is None:
            continue
        topic_views.setdefault(r.topic or "(untitled)", []).append(float(r.views))
    pause: List[Action] = []
    for topic, vals in topic_views.items():
        avg = _avg(vals) or 0.0
        if avg < ACTION_LOW_VIEWS:
            pause.append(Action(
                kind="pause", target=topic, metric=round(avg, 1),
                reason=f"avg_views {avg:.1f} < {ACTION_LOW_VIEWS} "
                       f"(pause topic; low reach)"))

    rerender.sort(key=lambda a: a.target)
    # De-dup promote targets (keep the highest metric per target).
    best_promote: Dict[str, Action] = {}
    for a in promote:
        if a.target not in best_promote or (a.metric or 0) > (best_promote[a.target].metric or 0):
            best_promote[a.target] = a
    promote_sorted = sorted(best_promote.values(), key=lambda a: a.target)
    pause.sort(key=lambda a: a.target)
    return rerender + promote_sorted + pause

