"""Round-27 hermetic tests: pure scoring/aggregation contract of ``shorts.virality``.

The existing suite (``tests/test_shorts.py``) checks only coarse weak/strong
scores, one plan ranking and the ``share`` factor.  The deterministic grading
thresholds, ``to_dict`` rounding/copy semantics, ``score_plan`` aggregation and
``best_short`` contract were unpinned by any test or report; the only prior
sweep in this module (sweep22x) covered word-boundary gap-token matching in
``_has_gap`` -- a different function and a different invariant.

Every test here is pure: in-memory dataclasses only.  No filesystem, graph,
service, network, credential, model, sleep or clock dependency is used.
"""
from __future__ import annotations

import pytest

from auto_ingest.shorts import virality
from auto_ingest.shorts.models import Brief, Cue, Plan, PlannedShort


def _cue(start: float, end: float, text: str, kind: str = "line") -> Cue:
    return Cue(start=start, end=end, text=text, kind=kind)


def _short(sid: str, *kinds: str, hook_text: str = "You've been using LLMs wrong") -> PlannedShort:
    """Build a deterministic planned short with a hook and the given cue kinds."""
    cues = [_cue(0.4, 3.4, hook_text, kind="hook")]
    t = 3.6
    for kind in kinds:
        cues.append(_cue(round(t, 2), round(t + 2.6, 2), f"{kind} line", kind=kind))
        t += 2.8
    return PlannedShort(id=sid, brief_topic="t", title="T", cues=cues)


def _plan(*shorts: PlannedShort) -> Plan:
    return Plan(
        topic="t",
        brief=Brief(topic="t", title="T", hook="h"),
        shorts=list(shorts),
    )


# --------------------------------------------------------------------------- #
# ViralityScore.grade / to_dict
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "total, expected",
    [
        (100.0, "A"),
        (80.0, "A"),
        (79.9, "B"),
        (65.0, "B"),
        (64.9, "C"),
        (50.0, "C"),
        (49.9, "D"),
        (35.0, "D"),
        (34.9, "F"),
        (0.0, "F"),
    ],
)
def test_grade_boundaries_are_strict_lower_bounds(total: float, expected: str) -> None:
    score = virality.ViralityScore(total=total, factors={}, suggestions=[])
    assert score.grade() == expected


def test_to_dict_rounds_total_and_factors_and_carries_grade() -> None:
    score = virality.ViralityScore(
        total=79.96,
        factors={"hook": 33.333, "ctr": 0.0},
        suggestions=["a"],
    )
    out = score.to_dict()
    assert out["total"] == round(79.96, 1)
    assert out["grade"] == score.grade()
    assert out["factors"] == {"hook": round(33.333, 1), "ctr": 0.0}
    assert out["suggestions"] == ["a"]


def test_to_dict_returns_copies_not_live_state() -> None:
    score = virality.ViralityScore(
        total=50.0, factors={"hook": 1.0}, suggestions=["keep"]
    )
    out = score.to_dict()
    out["suggestions"].append("extra")
    out["factors"]["hook"] = 999.0
    out["total"] = -1.0
    assert score.suggestions == ["keep"]
    assert score.factors == {"hook": 1.0}
    assert score.total == 50.0


# --------------------------------------------------------------------------- #
# score_plan aggregation
# --------------------------------------------------------------------------- #
def test_score_plan_empty_plan_is_all_zero() -> None:
    assert virality.score_plan(_plan()) == {
        "shorts": {},
        "ranked": [],
        "mean": 0.0,
        "best": 0.0,
        "worst": 0.0,
    }


def test_score_plan_aggregates_match_per_short_scores() -> None:
    shorts = [
        _short("weak", "point", "point"),
        _short("mid", "callback"),
        _short("strong", "callback", "reveal", "payoff", "myth", "fact"),
    ]
    result = virality.score_plan(_plan(*shorts))
    totals = [virality.score_short(s).total for s in shorts]
    assert set(result["shorts"]) == {s.id for s in shorts}
    assert result["mean"] == round(sum(totals) / len(totals), 1)
    assert result["best"] == round(max(totals), 1)
    assert result["worst"] == round(min(totals), 1)


def test_score_plan_rank_is_complete_and_descending() -> None:
    shorts = [
        _short("weak", "point", "point"),
        _short("mid", "callback"),
        _short("strong", "callback", "reveal", "payoff", "myth", "fact"),
    ]
    result = virality.score_plan(_plan(*shorts))
    assert len(result["ranked"]) == len(shorts)
    assert set(result["ranked"]) == {s.id for s in shorts}
    ranked_totals = [result["shorts"][sid]["total"] for sid in result["ranked"]]
    assert ranked_totals == sorted(ranked_totals, reverse=True)
    assert result["ranked"][0] == "strong"


def test_score_plan_ties_preserve_input_order_and_are_deterministic() -> None:
    # Two shorts with byte-identical cues score identically; the stable sort
    # must keep input order, so permuting the same rows permutes the ranking.
    a = _short("a", "callback", "reveal", "payoff", "myth", "fact")
    b = _short("b", "callback", "reveal", "payoff", "myth", "fact")
    first = virality.score_plan(_plan(a, b))
    swapped = virality.score_plan(_plan(b, a))
    assert first["shorts"]["a"]["total"] == first["shorts"]["b"]["total"]
    assert first["ranked"] == ["a", "b"]
    assert swapped["ranked"] == ["b", "a"]
    assert virality.score_plan(_plan(a, b)) == first


def test_best_short_agrees_with_top_rank_and_handles_empty() -> None:
    assert virality.best_short(_plan()) is None
    plan = _plan(
        _short("weak", "point"),
        _short("strong", "callback", "reveal", "payoff", "myth", "fact"),
    )
    assert virality.best_short(plan) == virality.score_plan(plan)["ranked"][0]
    assert virality.best_short(plan) == "strong"


def test_score_plan_is_pure_over_its_plan() -> None:
    plan = _plan(_short("a", "callback", "myth", "fact", "reveal", "payoff"))
    before = [s.to_dict() for s in plan.shorts]
    first = virality.score_plan(plan)
    second = virality.score_plan(plan)
    assert first == second
    assert [s.to_dict() for s in plan.shorts] == before


# --------------------------------------------------------------------------- #
# bounded scoring helpers
# --------------------------------------------------------------------------- #
def test_factors_and_total_are_bounded() -> None:
    score = virality.score_short(
        _short("s", "callback", "reveal", "payoff", "myth", "fact")
    )
    assert 0.0 <= score.total <= 100.0
    assert score.factors
    for value in score.factors.values():
        assert 0.0 <= value <= 100.0


def test_pacing_and_retention_helpers_stay_bounded() -> None:
    assert virality._pacing([], 0.0) == (50.0, [])
    assert virality._pacing([], -5.0) == (50.0, [])
    cues = [
        _cue(0.0, 1.0, "hook", "hook"),
        _cue(20.0, 21.0, "payoff", "payoff"),
        _cue(21.0, 22.0, "callback", "callback"),
    ]
    score, tips = virality._retention(cues, 22.0)
    assert 0.0 <= score <= 100.0
    assert any("dead air" in tip for tip in tips)
