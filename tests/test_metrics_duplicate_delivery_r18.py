"""Duplicate-delivery idempotency at the metrics ingestion boundary (r18).

Platform analytics exports are re-fetched and re-delivered: a client retry, an
overlapping date window, or a second pull of the same report puts the same
``short_id`` into the export more than once. ``ingest_dicts`` persists each row
through ``upsert_metric``, whose durable identity is ``(short_id, platform)``.
The contract pinned here is that a duplicate delivery converges to a single
stored record (latest value wins) instead of accumulating rows.

Why this is a real failure surface: ``feedback_report`` averages views and
retention over *every* stored record, and ``suggest_actions`` turns those
aggregates into promote/pause decisions. A duplicate row is not harmless noise
— it double-counts one short, skews ``avg_views`` / ``avg_retention``, and can
flip a threshold. The existing ``test_upsert_overwrites`` pins the helper
directly; nothing exercises the duplicate through the real ingestion entry
points (``ingest_dicts`` / ``ingest_platform_csv``), which is where a
re-delivered export actually lands.

All fixtures are a temp JSONL store (``tmp_path``), in-memory dicts, and a tiny
temp CSV. No live DB, Neo4j, network, credential, service, or model is touched.
"""
from __future__ import annotations

import csv
from pathlib import Path

from auto_ingest.shorts.feedback import feedback_report
from auto_ingest.shorts.metrics import (
    ingest_dicts,
    ingest_platform_csv,
    load_metrics,
)


def _row(
    short_id: str = "s1",
    views: int = 100,
    avg_view_pct: float = 50.0,
    topic: str = "transformers",
    **extra: object,
) -> dict:
    row = {
        "short_id": short_id,
        "topic": topic,
        "views": views,
        "avg_view_pct": avg_view_pct,
        "published_at": "2025-01-01",
        "fetched_at": "2025-01-02",
    }
    row.update(extra)
    return row


def test_identical_row_delivered_twice_collapses_to_one_record(tmp_path: Path) -> None:
    store = tmp_path / "metrics.jsonl"
    row = _row()

    written = ingest_dicts("youtube", [row, dict(row)], path=store)

    assert written == 2  # both deliveries were processed
    out = load_metrics(path=store)
    assert len(out) == 1  # but only one durable record exists
    assert out[0].short_id == "s1"
    assert out[0].views == 100


def test_redelivery_converges_to_latest_value(tmp_path: Path) -> None:
    store = tmp_path / "metrics.jsonl"

    ingest_dicts("youtube", [_row(views=100)], path=store)
    # A re-fetch of the same export reports a later, larger view count.
    ingest_dicts("youtube", [_row(views=400, fetched_at="2025-01-03")], path=store)

    out = load_metrics(path=store)
    assert len(out) == 1
    assert out[0].views == 400
    assert out[0].fetched_at == "2025-01-03"


def test_dedup_identity_is_composite_short_id_and_platform(tmp_path: Path) -> None:
    store = tmp_path / "metrics.jsonl"

    ingest_dicts("youtube", [_row(views=100), _row(views=250)], path=store)
    ingest_dicts("tiktok", [_row(views=7)], path=store)

    records = load_metrics(path=store)
    assert len(records) == 2  # the two youtube deliveries did not accumulate
    out = {(r.short_id, r.platform): r for r in records}
    assert out[("s1", "youtube")].views == 250
    assert out[("s1", "tiktok")].views == 7


def test_platform_csv_duplicate_rows_are_idempotent(tmp_path: Path) -> None:
    store = tmp_path / "metrics.jsonl"
    csv_path = tmp_path / "export.csv"
    fields = [
        "short_id", "topic", "views", "avg_view_pct", "published_at", "fetched_at",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerow(_row(views=100))
        writer.writerow(_row(views=100))  # duplicate delivery inside the export
        writer.writerow(_row(short_id="s2", views=900, topic="myths"))

    n = ingest_platform_csv("youtube", str(csv_path), path=store)

    assert n == 3
    records = load_metrics(path=store)
    assert len(records) == 2  # the in-export duplicate did not accumulate
    by_id = {r.short_id: r.views for r in records}
    assert by_id == {"s1": 100, "s2": 900}


def test_duplicate_delivery_does_not_skew_feedback_aggregates(tmp_path: Path) -> None:
    """The downstream consequence: one short must not be counted twice."""
    store = tmp_path / "metrics.jsonl"

    ingest_dicts("youtube", [_row(views=100), _row(views=400)], path=store)

    rep = feedback_report(load_metrics(path=store))
    # One record with the latest count -> 400, not (100 + 400) / 2 == 250.
    assert rep["by_platform"]["youtube"]["avg_views"] == 400.0
    assert rep["topic_perf"]["transformers"] == 400.0


def test_distinct_rows_are_not_over_deduped(tmp_path: Path) -> None:
    """Control: the dedup key must not swallow legitimately distinct shorts."""
    store = tmp_path / "metrics.jsonl"

    written = ingest_dicts(
        "youtube",
        [_row(short_id="a", views=10), _row(short_id="b", views=20),
         _row(short_id="c", views=30)],
        path=store,
    )

    assert written == 3
    assert len(load_metrics(path=store)) == 3
