"""An impossible camera date must not be replaced by a looser fallback date.

`parse_key_datetime_utc_from_string` tried specific patterns first, but a
`strptime` failure was swallowed and parsing fell through to a directory-date
plus time pattern. A clip named ``20260230_120000`` under ``/2026/02/28/`` was
therefore re-keyed to a plausible-looking 2026-02-28 instant — the join key now
names a moment the camera never recorded.
"""
from __future__ import annotations

from auto_ingest.ingest.discovery import parse_key_datetime_utc_from_string as parse


def test_impossible_date_does_not_fall_through_to_directory_date():
    assert parse("/2026/02/28/clip_20260230_120000.jpg") is None


def test_impossible_fourteen_digit_stamp_is_rejected():
    assert parse("clip_20260230000000.jpg") is None


def test_a_valid_camera_date_still_parses():
    result = parse("/2026/02/28/clip_20260228_120000.jpg")
    assert result is not None
    assert (result.year, result.month, result.day) == (2026, 2, 28)


def test_the_directory_date_fallback_still_works_for_undated_names():
    result = parse("/2026/03/01/clip_090000.jpg")
    assert result is not None
    assert (result.year, result.month, result.day) == (2026, 3, 1)
