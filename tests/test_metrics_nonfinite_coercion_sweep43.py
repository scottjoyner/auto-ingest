"""Hermetic regression tests for non-finite platform-export metric coercion.

Covers ``auto_ingest.shorts.metrics._coerce_num`` and the pure ``ingest_dicts``
path. No Neo4j, no network, no credentials, no model, no live store: only
in-memory dicts and pytest ``tmp_path``.
"""
import json

import pytest

from auto_ingest.shorts import metrics


@pytest.mark.parametrize("raw", ["inf", "-inf", "Infinity", "1e309"])
def test_infinite_count_string_is_rejected_not_an_uncaught_crash(raw):
    """A count column carrying 'inf'/'1e309' must coerce to None, not raise.

    Base code did ``int(float(s))`` inside ``except ValueError`` only;
    ``int(float('inf'))`` raises OverflowError and kills the whole ingest.
    """
    assert metrics._coerce_num(raw, as_int=True) is None


@pytest.mark.parametrize("raw", ["nan", "inf", "-inf", "1e309"])
def test_nonfinite_float_string_is_rejected_not_persisted(raw):
    """Float metric columns must not admit NaN/Infinity into the JSONL store."""
    assert metrics._coerce_num(raw, as_int=False) is None


def test_ingest_dicts_survives_an_infinite_views_row(tmp_path):
    """One bad export row must not abort ingestion of the remaining rows."""
    store = tmp_path / "metrics.jsonl"
    rows = [
        {"short_id": "bad", "views": "inf", "ctr": "nan"},
        {"short_id": "good", "views": "42", "ctr": "3.5%"},
    ]
    written = metrics.ingest_dicts("youtube", rows, path=store)
    assert written == 2

    recs = metrics.load_metrics(path=store)
    by_id = {r.short_id: r for r in recs}
    assert by_id["bad"].views is None
    assert by_id["bad"].ctr is None
    assert by_id["good"].views == 42
    assert by_id["good"].ctr == pytest.approx(3.5)

    # Non-standard NaN/Infinity literals must never reach the store.
    text = store.read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
    for line in text.splitlines():
        json.loads(line)  # every stored row must be strict JSON


def test_valid_values_still_coerce_as_before():
    """Control: ordinary export values keep their existing behaviour."""
    assert metrics._coerce_num("1,234", as_int=True) == 1234
    assert metrics._coerce_num("3.5%", as_int=False) == pytest.approx(3.5)
    assert metrics._coerce_num(7, as_int=True) == 7
    assert metrics._coerce_num(None, as_int=False) is None
    assert metrics._coerce_num("not-a-number", as_int=True) is None
