"""Hermetic contract tests for personal-recall output formatting (r36).

Pins ``auto_ingest.personal.recall._fmt`` and ``print_table`` against
non-finite floats.  A cosine score or a distance that is infinite must never
be rendered as the literal text ``inf``/``-inf`` in the CLI table: every other
non-finite value (NaN) is already blanked, and an ``inf`` cell is not a score.

No Neo4j driver, model, network, file or clock IO.  Only pure functions and
stdout capture by pytest.
"""
from __future__ import annotations

import math

import auto_ingest.personal.recall as recall


def test_fmt_nan_is_blank():
    assert recall._fmt(float("nan")) == ""


def test_fmt_positive_infinity_is_blank():
    # Regression: currently renders as "inf".
    assert recall._fmt(float("inf")) == ""


def test_fmt_negative_infinity_is_blank():
    # Regression: currently renders as "-inf".
    assert recall._fmt(float("-inf")) == ""


def test_fmt_finite_small_keeps_four_decimals():
    assert recall._fmt(0.123456) == "0.1235"


def test_fmt_finite_large_keeps_two_decimals():
    assert recall._fmt(1234.5678) == "1234.57"


def test_fmt_finite_negative_large_keeps_sign_and_two_decimals():
    assert recall._fmt(-1234.5) == "-1234.50"


def test_fmt_none_is_blank():
    assert recall._fmt(None) == ""


def test_fmt_int_is_plain_string():
    assert recall._fmt(42) == "42"


def test_fmt_string_is_unchanged():
    assert recall._fmt("2024-01-01") == "2024-01-01"


def test_fmt_zero_is_not_blanked():
    assert recall._fmt(0.0) == "0.0000"


def test_print_table_blanks_inf_cell(capsys):
    rows = [
        {"sha256": "a", "score": float("inf")},
        {"sha256": "b", "score": 0.5},
    ]
    recall.print_table(rows, [("score", "score"), ("sha256", "sha256")])
    out = capsys.readouterr().out
    assert "inf" not in out
    assert "a" in out and "b" in out
    assert "0.5000" in out


def test_print_table_keeps_finite_cells(capsys):
    rows = [{"sha256": "x", "score": math.pi}]
    recall.print_table(rows, [("score", "score"), ("sha256", "sha256")])
    out = capsys.readouterr().out
    assert "3.1416" in out
    assert "x" in out
