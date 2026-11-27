"""Deterministic action ordering from performance metrics (r23).

``auto_ingest.shorts.feedback.suggest_actions()`` documents a stable-output
contract in its own docstring: rerender, then promote, then pause; each group
sorted by target; promote targets de-duplicated keeping the highest retention.
The existing suite (``tests/test_metrics.py``) pins only threshold membership
and "two identical calls agree", so record-order dependence -- the normal state
when platform exports are re-fetched or delivered concurrently -- is invisible
to it. These tests pin the ordering contract itself.

Hermetic: in-memory ``MetricsRecord`` objects only. No filesystem, DB, graph,
network, credential, service or model is touched.
"""
from __future__ import annotations

from auto_ingest.shorts.feedback import suggest_actions
from auto_ingest.shorts.metrics import MetricsRecord


def _rec(short_id, topic, *, ctr=None, retention=None, views=None,
         platform="youtube"):
    """A metrics row with only the fields under test populated."""
    return MetricsRecord(
        short_id=short_id, platform=platform, topic=topic, published_at="",
        ctr=ctr, avg_view_pct=retention, views=views,
    )


def _triples(actions):
    return [(a.kind, a.target, a.metric) for a in actions]


def test_group_order_is_rerender_promote_pause_regardless_of_input_order():
    # Deliberately supplied in the *reverse* of the promised group order.
    recs = [
        _rec("zdead", "robotics", ctr=5.0, retention=40.0, views=10),
        _rec("mhi", "transformers", ctr=5.0, retention=88.0, views=800),
        _rec("alow", "rag", ctr=1.0, retention=30.0, views=500),
    ]
    assert [a.kind for a in suggest_actions(recs)] == [
        "rerender", "promote", "pause",
    ]


def test_within_group_actions_are_sorted_by_target():
    # Scrambled targets in every group; output must be lexical by target.
    recs = [
        _rec("zulu", "zeta", ctr=1.0, retention=30.0, views=500),
        _rec("alpha", "beta", ctr=1.5, retention=30.0, views=500),
        _rec("mike", "mid", ctr=1.2, retention=30.0, views=500),
        _rec("sr", "robotics", ctr=5.0, retention=40.0, views=10),
        _rec("sa", "agents", ctr=5.0, retention=40.0, views=20),
    ]
    actions = suggest_actions(recs)
    assert [a.target for a in actions if a.kind == "rerender"] == [
        "alpha", "mike", "zulu",
    ]
    assert [a.target for a in actions if a.kind == "pause"] == [
        "agents", "robotics",
    ]


def test_output_is_invariant_under_input_permutation():
    """The core contract: re-ordered/exported-later rows give byte-equal output."""
    recs = [
        _rec("r2", "rag", ctr=1.0, retention=30.0, views=500),
        _rec("p1", "transformers", ctr=5.0, retention=72.0, views=800),
        _rec("q1", "robotics", ctr=5.0, retention=40.0, views=10),
        _rec("r1", "agents", ctr=1.5, retention=30.0, views=500),
        _rec("p2", "vision", ctr=5.0, retention=65.0, views=900),
        _rec("q2", "music", ctr=5.0, retention=40.0, views=20),
    ]
    expected = [a.to_dict() for a in suggest_actions(recs)]
    assert len(expected) == 6  # exercises all three groups
    for permuted in (list(reversed(recs)), recs[2:] + recs[:2],
                     [recs[i] for i in (4, 0, 5, 1, 3, 2)]):
        assert [a.to_dict() for a in suggest_actions(permuted)] == expected


def test_promote_dedup_keeps_highest_retention_per_target():
    recs = [
        _rec("low", "agents", ctr=5.0, retention=61.0, views=800),
        _rec("high", "agents", ctr=5.0, retention=88.0, views=800),
        _rec("other", "vision", ctr=5.0, retention=70.0, views=800),
    ]
    promotes = [a for a in suggest_actions(recs) if a.kind == "promote"]
    assert [a.target for a in promotes] == ["agents", "vision"]
    assert promotes[0].metric == 88.0
    assert "88.0%" in promotes[0].reason


def test_pause_uses_topic_average_not_first_or_last_record():
    # "steady" has one tiny and one large row -> mean 505 >= 100 -> no pause.
    # "thin" has two tiny rows -> mean 15 < 100 -> pause with that mean.
    recs = [
        _rec("s1", "steady", ctr=5.0, retention=30.0, views=10),
        _rec("s2", "steady", ctr=5.0, retention=30.0, views=1000),
        _rec("t1", "thin", ctr=5.0, retention=30.0, views=10),
        _rec("t2", "thin", ctr=5.0, retention=30.0, views=20),
    ]
    pauses = [a for a in suggest_actions(recs) if a.kind == "pause"]
    assert [a.target for a in pauses] == ["thin"]
    assert pauses[0].metric == 15.0


def test_threshold_boundaries_are_strict():
    recs = [
        _rec("ctr_eq", "t_ctr_eq", ctr=2.0, retention=30.0, views=500),
        _rec("ctr_below", "t_ctr_below", ctr=1.99, retention=30.0, views=500),
        _rec("ret_eq", "t_ret_eq", ctr=5.0, retention=60.0, views=500),
        _rec("ret_above", "t_ret_above", ctr=5.0, retention=60.01, views=500),
        _rec("views_eq", "t_views_eq", ctr=5.0, retention=30.0, views=100),
        _rec("views_below", "t_views_below", ctr=5.0, retention=30.0, views=99),
    ]
    actions = suggest_actions(recs)
    assert [a.target for a in actions if a.kind == "rerender"] == ["ctr_below"]
    assert [a.target for a in actions if a.kind == "promote"] == ["t_ret_above"]
    assert [a.target for a in actions if a.kind == "pause"] == ["t_views_below"]


def test_missing_metric_fields_produce_no_actions():
    # Control: a row with no numeric signals must not spin up a spurious action.
    assert suggest_actions([_rec("nometric", "", ctr=None, retention=None,
                                 views=None)]) == []


def test_empty_input_still_empty():
    # Control over the empty-input guard.
    assert suggest_actions([]) == []
