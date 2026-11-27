"""De-duplicate rerender actions by target (r19).

``auto_ingest.shorts.feedback.suggest_actions()`` de-duplicates *promote*
targets but, before this fix, emitted one ``rerender`` action per metrics row:
a short delivered twice with different CTRs produced two rerender actions whose
stable-sort tie on ``target`` preserved delivery order, so the documented
"deterministic / stable output" contract broke under re-fetched or concurrently
delivered exports.

Hermetic: in-memory ``MetricsRecord`` objects only. No filesystem, DB, graph,
network, credential, service or model is touched.
"""
from __future__ import annotations

from auto_ingest.shorts.feedback import suggest_actions
from auto_ingest.shorts.metrics import MetricsRecord


def _rec(short_id, *, ctr=None, topic="rag", retention=30.0, views=500,
         platform="youtube"):
    """A metrics row with only the fields under test populated."""
    return MetricsRecord(
        short_id=short_id, platform=platform, topic=topic, published_at="",
        ctr=ctr, avg_view_pct=retention, views=views,
    )


def _rerenders(recs):
    return [a for a in suggest_actions(recs) if a.kind == "rerender"]


def _triples(actions):
    return [(a.kind, a.target, a.metric) for a in actions]


def test_duplicate_short_yields_single_rerender():
    """A short delivered twice must produce exactly one rerender action."""
    recs = [_rec("s1", ctr=1.5), _rec("s1", ctr=0.5)]
    rr = _rerenders(recs)
    assert [a.target for a in rr] == ["s1"]


def test_rerender_dedup_keeps_lowest_ctr():
    """Of duplicate rows, the strongest rerender signal (lowest CTR) survives."""
    recs = [_rec("s1", ctr=1.5), _rec("s1", ctr=0.5)]
    rr = _rerenders(recs)
    assert len(rr) == 1
    assert rr[0].metric == 0.5
    assert "0.50%" in rr[0].reason


def test_rerender_dedup_is_order_invariant():
    """Permuting duplicate rows must give byte-identical action output."""
    recs = [
        _rec("b", ctr=1.2),
        _rec("a", ctr=1.9),
        _rec("b", ctr=0.4),
        _rec("a", ctr=0.7),
        _rec("c", ctr=1.0),
    ]
    expected = [a.to_dict() for a in suggest_actions(recs)]
    assert [a.target for a in _rerenders(recs)] == ["a", "b", "c"]
    for permuted in (
        list(reversed(recs)),
        recs[2:] + recs[:2],
        [recs[i] for i in (3, 0, 4, 1, 2)],
    ):
        assert [a.to_dict() for a in suggest_actions(permuted)] == expected


def test_distinct_shorts_still_get_distinct_rerenders():
    """Control: de-dup must not collapse genuinely different shorts."""
    recs = [_rec("z", ctr=1.0), _rec("a", ctr=1.5), _rec("m", ctr=0.5)]
    assert [a.target for a in _rerenders(recs)] == ["a", "m", "z"]


def test_rerender_dedup_does_not_disturb_promote_or_pause():
    """Control: rerender de-dup is independent of the other two groups."""
    recs = [
        _rec("dup", ctr=1.5, retention=88.0, views=800, topic="agents"),
        _rec("dup", ctr=0.5, retention=88.0, views=800, topic="agents"),
        _rec("dead", ctr=5.0, retention=40.0, views=10, topic="robotics"),
    ]
    actions = _triples(suggest_actions(recs))
    assert [a for a in actions if a[0] == "rerender"] == [("rerender", "dup", 0.5)]
    assert [a for a in actions if a[0] == "promote"] == [("promote", "agents", 88.0)]
    assert [a for a in actions if a[0] == "pause"] == [("pause", "robotics", 10.0)]


def test_no_rerender_for_ctr_at_or_above_threshold():
    """Control: strict CTR threshold is preserved by the de-dup change."""
    assert _rerenders([_rec("s1", ctr=2.0)]) == []
    assert _rerenders([_rec("s1", ctr=5.0)]) == []


def test_empty_input_still_empty():
    """Control over the empty-input guard."""
    assert suggest_actions([]) == []
