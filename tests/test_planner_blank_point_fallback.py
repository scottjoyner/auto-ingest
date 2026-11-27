"""Regression: ``plan_shorts`` must never emit a blank point cue.

The point fallback in ``plan_shorts`` is meant to guarantee a visible, non-empty
caption even when a Brief has neither curated points nor a hook. But the first
line wraps a falsy hook in a one-element list, so the ``if not points`` guard on
the next line is dead code and ``''`` reaches the renderer as a point cue.

Hermetic: pure models + planner only. No Neo4j, network, mounts, credentials,
model load, live DB or service interaction.
"""
from auto_ingest.shorts import planner
from auto_ingest.shorts.models import Brief

_PLACEHOLDER = "(no points curated)"


def _point_texts(*, hook, points, short_count=1):
    brief = Brief(topic="t", title="T", hook=hook, points=list(points))
    plan = planner.plan_shorts(
        brief, [], short_count=short_count, short_dur=10.0,
        shots_per_short=0, seed=1,
    )
    return [c.text for short in plan.shorts for c in short.cues if c.kind == "point"]


def test_blank_brief_gets_visible_placeholder_point():
    """fail->pass: pristine emits [''] because `or [brief.hook]` is never empty."""
    assert _point_texts(hook="", points=[]) == [_PLACEHOLDER]


def test_blank_brief_never_emits_empty_point_cue():
    """fail->pass: same defect, asserted at the non-blank-caption contract level."""
    assert all(t.strip() for t in _point_texts(hook="", points=[]))


def test_hook_still_used_as_point_when_points_empty():
    """control: the intended hook fallback is preserved (passes pre- and post-fix)."""
    assert _point_texts(hook="only hook", points=[]) == ["only hook"]


def test_curated_points_are_unchanged():
    """control: curated points are untouched (passes pre- and post-fix)."""
    assert _point_texts(hook="ignored hook", points=["p1", "p2"]) == ["p1", "p2"]
