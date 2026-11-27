"""Hermetic regression tests for the strided point partition in ``plan_shorts``.

The partition is documented as disjoint ("every point appears in exactly one
short and none are dropped"). When there are fewer curated points than shorts,
the old ``points[start:start + stride] or points[:1]`` fallback repeated the
first point in every extra short. These tests use only in-memory ``Brief``
objects -- no driver, DB, network, mount, credential or model load.
"""
from __future__ import annotations

from auto_ingest.shorts.models import Brief
from auto_ingest.shorts.planner import plan_shorts

PLACEHOLDER = "(no points curated)"


def _brief(points):
    return Brief(topic="t", title="T", hook="h", points=list(points))


def _point_cues(plan):
    return [
        cue.text
        for short in plan.shorts
        for cue in short.cues
        if cue.kind == "point"
    ]


def test_extra_shorts_do_not_repeat_an_earlier_point():
    """Fewer points than shorts: the first point must not be duplicated."""
    plan = plan_shorts(_brief(["p-alpha", "p-beta", "p-gamma"]), anchors=[],
                       short_count=5, driver=None)
    seen = _point_cues(plan)
    curated = [t for t in seen if t != PLACEHOLDER]
    assert curated.count("p-alpha") == 1, seen
    assert len(curated) == len(set(curated)), "a curated point was repeated across shorts"


def test_extra_short_uses_placeholder_not_first_point():
    """The empty strided slice must yield the explicit placeholder."""
    plan = plan_shorts(_brief(["p-alpha", "p-beta", "p-gamma"]), anchors=[],
                       short_count=5, driver=None)
    assert len(plan.shorts) == 5
    tails = [
        [c.text for c in short.cues if c.kind == "point"]
        for short in plan.shorts[3:]
    ]
    assert tails == [[PLACEHOLDER], [PLACEHOLDER]], tails


def test_all_curated_points_still_covered_exactly_once():
    """Control: disjoint+complete partition is unchanged when points >= shorts."""
    points = [f"p{i}" for i in range(10)]
    plan = plan_shorts(_brief(points), anchors=[], short_count=3, driver=None)
    assert sorted(_point_cues(plan)) == sorted(points)
    assert len(_point_cues(plan)) == len(set(_point_cues(plan)))


def test_equal_points_and_shorts_has_no_placeholder():
    """Control: no empty slice means no placeholder is introduced."""
    plan = plan_shorts(_brief(["p1", "p2", "p3"]), anchors=[],
                       short_count=3, driver=None)
    assert sorted(_point_cues(plan)) == ["p1", "p2", "p3"]
    assert PLACEHOLDER not in _point_cues(plan)
