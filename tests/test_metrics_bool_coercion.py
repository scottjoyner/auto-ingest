"""Hermetic regression: platform-export booleans are not metric numbers.

Python ``bool`` subclasses ``int``. A malformed platform export carrying JSON
``true``/``false`` in a numeric column must be treated as a missing value, not
fabricated into ``1``/``0``. No network, no credentials, no live store: all
fixtures are in-memory dicts or pytest ``tmp_path`` files.
"""
from __future__ import annotations

from auto_ingest.shorts.metrics import _coerce_num, ingest_dicts, load_metrics


def test_bool_is_not_an_integer_metric():
    assert _coerce_num(True, as_int=True) is None
    assert _coerce_num(False, as_int=True) is None


def test_bool_is_not_a_float_metric():
    assert _coerce_num(True, as_int=False) is None
    assert _coerce_num(False, as_int=False) is None


def test_genuine_numbers_and_strings_still_coerce():
    assert _coerce_num(7, as_int=True) == 7
    assert _coerce_num(0, as_int=True) == 0
    assert _coerce_num(2.5, as_int=False) == 2.5
    assert _coerce_num("1,200", as_int=True) == 1200
    assert _coerce_num("40%", as_int=False) == 40.0
    assert _coerce_num(None, as_int=True) is None
    assert _coerce_num("", as_int=False) is None


def test_ingest_dicts_does_not_fabricate_counts_from_booleans(tmp_path):
    store = tmp_path / "metrics.jsonl"
    rows = [
        {"short_id": "b1", "views": True, "likes": False, "ctr": True},
        {"short_id": "b2", "views": 250, "ctr": 1.5},
    ]
    assert ingest_dicts("youtube", rows, path=store) == 2
    by_id = {r.short_id: r for r in load_metrics(path=store)}
    assert by_id["b1"].views is None
    assert by_id["b1"].likes is None
    assert by_id["b1"].ctr is None
    assert by_id["b2"].views == 250
    assert by_id["b2"].ctr == 1.5
