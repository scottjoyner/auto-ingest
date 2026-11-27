"""Regression: pred_vs_actual must not drop matched posts when all views are 0.

Hermetic: pure functions only, no store file, no network, no clock.
Run with: python3 -m pytest tests/test_feedback_zero_views.py
"""
from __future__ import annotations

from auto_ingest.shorts.feedback import feedback_report, suggest_next
from auto_ingest.shorts.metrics import MetricsRecord


def _rec(short_id: str, *, views: int, pred: float) -> MetricsRecord:
    return MetricsRecord(short_id=short_id, platform="youtube", topic="t",
                         published_at="2025-01-01", views=views,
                         pred_virality=pred)


def test_zero_views_still_count_matched_posts():
    recs = [_rec("a", views=0, pred=60.0), _rec("b", views=0, pred=40.0)]
    pva = feedback_report(recs)["pred_vs_actual"]
    # Both rows have a view count and a prediction: they are matched posts, and
    # their normalized actual is 0.0, so MAE is the mean prediction (50).
    assert pva == {"mean_abs_error": 50.0, "n": 2}


def test_zero_views_flags_miscalibration_in_suggestions():
    recs = [_rec("a", views=0, pred=60.0), _rec("b", views=0, pred=40.0)]
    sugg = " ".join(suggest_next(recs)).lower()
    assert "off by" in sugg
    assert "keep posting" not in sugg


def test_nonzero_views_behaviour_unchanged():
    recs = [_rec("a", views=1000, pred=80.0), _rec("b", views=200, pred=40.0)]
    # normalized actuals: 100 and 20 -> abs errors 20 and 20 -> MAE 20.
    assert feedback_report(recs)["pred_vs_actual"] == {
        "mean_abs_error": 20.0, "n": 2
    }
