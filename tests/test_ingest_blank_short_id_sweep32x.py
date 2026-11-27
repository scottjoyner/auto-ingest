"""Regression: whitespace-only short IDs must not be ingested as real metrics.

`sweep32x` — offline/hermetic: a tiny JSONL store under pytest's tmp_path is the
only I/O; no network, DB, credentials, media or live fleet state.
"""
from __future__ import annotations

from auto_ingest.shorts.metrics import ingest_dicts, load_metrics


def test_spaces_only_short_id_is_skipped(tmp_path):
    """The 'no short_id' skip guard must hold for whitespace-only IDs."""
    store = tmp_path / "metrics.jsonl"
    n = ingest_dicts("youtube", [{"short_id": "   ", "views": 100}], path=store)
    assert n == 0
    assert load_metrics(path=store) == []


def test_tab_newline_short_id_is_skipped(tmp_path):
    store = tmp_path / "metrics.jsonl"
    n = ingest_dicts("youtube", [{"short_id": "\t\n", "views": 200}], path=store)
    assert n == 0
    assert load_metrics(path=store) == []


def test_whitespace_alias_id_is_skipped(tmp_path):
    """The same guard applies when the ID comes from a column alias."""
    store = tmp_path / "metrics.jsonl"
    n = ingest_dicts("youtube", [{"id": " \t ", "views": 300}], path=store)
    assert n == 0
    assert load_metrics(path=store) == []


def test_empty_short_id_control(tmp_path):
    """Control: the pre-existing empty-string guard is unchanged."""
    store = tmp_path / "metrics.jsonl"
    n = ingest_dicts("youtube", [{"short_id": "", "views": 400}], path=store)
    assert n == 0
    assert load_metrics(path=store) == []


def test_valid_short_id_control(tmp_path):
    """Control: a real ID is still ingested with its numeric fields."""
    store = tmp_path / "metrics.jsonl"
    n = ingest_dicts("youtube", [{"short_id": "real1", "views": 10}], path=store)
    assert n == 1
    rows = load_metrics(path=store)
    assert [r.short_id for r in rows] == ["real1"]
    assert rows[0].views == 10
