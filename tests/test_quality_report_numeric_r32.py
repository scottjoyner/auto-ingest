"""Hermetic contract for ``quality_report`` numeric coercion (r32).

``quality_report.validate_rows`` is a pure CSV row-consistency checker: it
decides whether each dashcam metadata row has a numeric latitude/longitude and
reports a validation issue when it does not.  The gate behind that decision is
``is_float``.  A missing CSV cell reaches it as ``None`` (or another non-string
object), and that must be reported as "not numeric" rather than crashing the
whole validation pass.

Everything here is in-memory: no CSV file, mount, DB, network or service.
"""
from __future__ import annotations

import quality_report as qr


def _row(**overrides):
    """One metadata row with all keys validate_rows reads."""
    row = {"Key": "camA", "MPH": "55", "Lat": "40.0", "Long": "-73.0", "Frame": "60"}
    row.update(overrides)
    return row


# --- is_float: coercion robustness ----------------------------------------

def test_is_float_accepts_numeric_strings():
    assert qr.is_float("12.5") is True
    assert qr.is_float("-3") is True
    assert qr.is_float("1e3") is True


def test_is_float_rejects_non_numeric_string():
    assert qr.is_float("not-a-number") is False
    assert qr.is_float("") is False


def test_is_float_rejects_none_without_raising():
    # A missing CSV cell is None; it must be "not numeric", never a TypeError.
    assert qr.is_float(None) is False


def test_is_float_rejects_other_non_string_objects_without_raising():
    for value in ([], {}, object()):
        assert qr.is_float(value) is False


def test_is_float_accepts_real_numbers():
    assert qr.is_float(12) is True
    assert qr.is_float(3.5) is True


# --- validate_rows: fail-closed on missing/None coordinates ---------------

def test_validate_rows_empty_or_single_row_reports_nothing():
    assert qr.validate_rows([]) == []
    assert qr.validate_rows([_row()]) == []  # first row is the baseline


def test_validate_rows_reports_none_latitude_as_not_numeric():
    rows = [_row(), _row(Lat=None)]
    issues = qr.validate_rows(rows)
    assert len(issues) == 1
    assert "Lat is not numeric" in issues[0]
    assert "Key=camA" in issues[0]


def test_validate_rows_reports_none_longitude_as_not_numeric():
    rows = [_row(), _row(Long=None)]
    issues = qr.validate_rows(rows)
    assert len(issues) == 1
    assert "Long is not numeric" in issues[0]


def test_validate_rows_non_string_latitude_does_not_crash():
    # An unexpected object in a cell must be reported, not raised through.
    rows = [_row(), _row(Lat=object())]
    issues = qr.validate_rows(rows)
    assert len(issues) == 1
    assert "Lat is not numeric" in issues[0]


def test_validate_rows_still_flags_large_numeric_lat_jump():
    rows = [_row(Lat="40.0"), _row(Lat="41.5")]
    issues = qr.validate_rows(rows)
    assert len(issues) == 1
    assert "Lat changed by" in issues[0]
    assert "exceeds threshold" in issues[0]


def test_validate_rows_still_flags_large_numeric_long_jump():
    rows = [_row(Long="-73.0"), _row(Long="-74.5")]
    issues = qr.validate_rows(rows)
    assert len(issues) == 1
    assert "Long changed by" in issues[0]
