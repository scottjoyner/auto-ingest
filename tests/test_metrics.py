"""Tests for the analytics + feedback loop (metrics.py / feedback.py).

Run with system python3:  python3 -m pytest tests/test_metrics.py
"""
from __future__ import annotations

from auto_ingest.shorts.feedback import feedback_report, suggest_next
from auto_ingest.shorts.metrics import (
    MetricsRecord,
    ingest_dicts,
    load_metrics,
    record_metric,
    upsert_metric,
)
from auto_ingest.shorts.models import Cue, PlannedShort


def _rec(**kw) -> MetricsRecord:
    base = dict(short_id="s1", platform="youtube", topic="llm",
                published_at="2025-01-01", fetched_at="2025-01-02",
                views=100, avg_view_pct=50.0)
    base.update(kw)
    return MetricsRecord(**base)


def test_record_load_roundtrip(tmp_path):
    store = tmp_path / "metrics.jsonl"
    record_metric(_rec(views=120), path=store)
    record_metric(_rec(short_id="s2", views=200), path=store)
    out = load_metrics(path=store)
    assert len(out) == 2
    assert {r.short_id for r in out} == {"s1", "s2"}
    assert out[0].views == 120


def test_upsert_overwrites(tmp_path):
    store = tmp_path / "metrics.jsonl"
    upsert_metric(_rec(views=10), path=store)
    upsert_metric(_rec(views=999), path=store)
    out = load_metrics(path=store)
    assert len(out) == 1
    assert out[0].views == 999
    # different platform is a distinct key
    upsert_metric(_rec(platform="tiktok", views=5), path=store)
    assert len(load_metrics(path=store)) == 2


def test_ingest_dicts_missing_fields_none(tmp_path):
    store = tmp_path / "metrics.jsonl"
    rows = [
        {"short_id": "a1", "views": 500, "avg_view_percentage": 62.5,
         "topic": "transformers"},
        {"short_id": "a2", "plays": "1,200", "retention": "40%", "title": "cnn"},
        {"short_id": "a3"},
    ]
    n = ingest_dicts("youtube", rows, path=store)
    assert n == 3
    out = load_metrics(path=store)
    by_id = {r.short_id: r for r in out}
    assert by_id["a1"].views == 500
    assert by_id["a1"].avg_view_pct == 62.5
    assert by_id["a2"].views == 1200
    assert by_id["a2"].avg_view_pct == 40.0
    assert by_id["a3"].views is None
    assert by_id["a3"].avg_view_pct is None
    assert by_id["a3"].topic == ""


def test_feedback_report_structure(tmp_path):
    recs = [
        _rec(short_id="a", topic="llm wrong", views=1000, avg_view_pct=70.0,
             platform="youtube", pred_virality=80.0),
        _rec(short_id="b", topic="cnn plain", views=200, avg_view_pct=30.0,
             platform="youtube", pred_virality=40.0),
        _rec(short_id="c", topic="instagram reel", views=400, avg_view_pct=45.0,
             platform="instagram", pred_virality=60.0),
    ]
    rep = feedback_report(recs)
    assert set(rep.keys()) == {
        "by_platform", "top_hooks", "weak_hooks", "topic_perf", "pred_vs_actual"
    }
    assert "youtube" in rep["by_platform"]
    assert rep["by_platform"]["youtube"]["avg_views"] == 600.0
    assert rep["top_hooks"][0]["text"] == "llm wrong"
    assert rep["weak_hooks"][0]["text"] == "cnn plain"
    assert rep["topic_perf"]["llm wrong"] == 1000.0
    assert rep["pred_vs_actual"]["n"] == 3
    assert rep["pred_vs_actual"]["mean_abs_error"] is not None


def test_suggest_next_sparse_empty():
    assert suggest_next([]) == [
        "Not enough data yet — ingest metrics after a few posts."
    ]


def test_suggest_next_derives_recs(tmp_path):
    recs = [
        _rec(short_id="a", topic="llm wrong", views=1000, avg_view_pct=70.0,
             platform="youtube", pred_virality=80.0),
        _rec(short_id="b", topic="cnn plain", views=200, avg_view_pct=30.0,
             platform="youtube", pred_virality=40.0),
        _rec(short_id="c", topic="instagram reel", views=50, avg_view_pct=20.0,
             platform="instagram", pred_virality=60.0),
    ]
    sugg = suggest_next(recs)
    assert len(sugg) >= 3
    joined = " ".join(sugg).lower()
    assert "instagram" in joined or "shorten" in joined
    assert "gap" in joined or "views" in joined


def test_publish_prediction_roundtrip(tmp_path):
    store = tmp_path / "metrics.jsonl"
    cues = [Cue(0.4, 3.4, "You've been using LLMs wrong", kind="hook"),
            Cue(4.0, 6.0, "reveal line", kind="reveal"),
            Cue(7.0, 9.0, "payoff", kind="payoff")]
    item = PlannedShort(id="x1", brief_topic="llm", title="T", cues=cues)
    from auto_ingest.shorts import metrics

    metrics.store_prediction_for_short(item, "youtube", path=store)
    out = load_metrics(path=store)
    assert len(out) == 1
    assert out[0].pred_virality is not None
    assert 0.0 <= out[0].pred_virality <= 100.0


def test_metrics_schema_version_written(tmp_path):
    from auto_ingest.shorts.metrics import METRICS_SCHEMA_VERSION
    store = tmp_path / "metrics.jsonl"
    record_metric(_rec(), path=store)
    import json as _json
    line = store.read_text(encoding="utf-8").splitlines()[0]
    assert _json.loads(line)["_v"] == METRICS_SCHEMA_VERSION


def test_metrics_missing_version_defaults_v1(tmp_path, caplog):
    import json as _json
    store = tmp_path / "metrics.jsonl"
    # Write a legacy row with no _v.
    legacy = {"short_id": "old", "platform": "youtube", "topic": "llm",
              "published_at": "2025-01-01", "views": 50}
    store.write_text(_json.dumps(legacy) + "\n", encoding="utf-8")
    with caplog.at_level("WARNING"):
        out = load_metrics(path=store)
    assert len(out) == 1
    assert out[0].short_id == "old"
    assert any("missing _v" in r.message for r in caplog.records)


def test_metrics_version_roundtrip(tmp_path):
    store = tmp_path / "metrics.jsonl"
    record_metric(_rec(short_id="rt", views=321), path=store)
    out = load_metrics(path=store)
    # Re-serialize and reload — value + version survive.
    assert out[0].to_dict()["_v"] == 1
    upsert_metric(out[0], path=store)
    again = load_metrics(path=store)
    assert again[0].views == 321


def test_suggest_actions_thresholds():
    from auto_ingest.shorts.feedback import (
        ACTION_CTR_RERENDER,
        ACTION_LOW_VIEWS,
        ACTION_RETENTION_PROMOTE,
        suggest_actions,
    )
    recs = [
        # low CTR -> rerender
        _rec(short_id="lowctr", ctr=1.0, views=500, avg_view_pct=30.0,
             topic="rag"),
        # high retention -> promote
        _rec(short_id="hiret", ctr=5.0, views=800, avg_view_pct=72.0,
             topic="transformers"),
        # low views topic -> pause
        _rec(short_id="deadtopic", ctr=4.0, views=10, avg_view_pct=40.0,
             topic="robotics"),
    ]
    actions = suggest_actions(recs)
    kinds = {(a.kind, a.target) for a in actions}
    assert ("rerender", "lowctr") in kinds
    assert ("promote", "transformers") in kinds
    assert ("pause", "robotics") in kinds
    # Threshold sanity.
    assert ACTION_CTR_RERENDER == 2.0
    assert ACTION_RETENTION_PROMOTE == 60.0
    assert ACTION_LOW_VIEWS == 100
    # Deterministic: same input -> same output ordering.
    assert [a.to_dict() for a in suggest_actions(recs)] == \
        [a.to_dict() for a in suggest_actions(recs)]


def test_suggest_actions_empty():
    from auto_ingest.shorts.feedback import suggest_actions
    assert suggest_actions([]) == []
