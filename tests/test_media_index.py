"""Hermetic regression tests for media_index date extraction (sweep23).

``_date_for`` used to accept the first shape-matching digit run in a path
without checking that it was a real calendar date. A resolution run such as
``1920x1080`` was read as ``1080_2024`` -> ``1080-20-24``, which both poisoned
the catalog with an impossible date and hid the real capture date later in the
filename. These tests run on synthetic paths / tiny tmp files only.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import media_index


def test_date_for_plain_date_still_works():
    assert media_index._date_for(Path("/data/20240315_clip.mp4")) == "2024-03-15"


def test_date_for_ignores_impossible_resolution_run():
    """A 1920x1080 run must not become '1080-20-24' or shadow the real date."""
    p = Path("/data/1920x1080_20240315_clip.mp4")
    assert media_index._date_for(p) == "2024-03-15"
    assert media_index._date_for(p) != "1080-20-24"


def test_date_for_impossible_month_falls_back_to_mtime(tmp_path):
    """An impossible month (13) is not a date; fall back to a deterministic mtime."""
    p = tmp_path / "video_20241301_clip.mp4"
    p.write_bytes(b"")
    ts = datetime(2023, 7, 4, 12, 0, tzinfo=timezone.utc).timestamp()
    os.utime(p, (ts, ts))
    assert media_index._date_for(p) == "2023-07-04"


def test_build_catalog_has_no_impossible_dates(tmp_path):
    """build()/scan_root() must not emit impossible dates into the catalog."""
    media = tmp_path / "capture"
    media.mkdir()
    (media / "1920x1080_20240315_clip.mp4").write_bytes(b"x")
    cat = media_index.build([str(media)], progress=False)
    assert len(cat["video"]) == 1
    assert cat["video"][0]["date"] == "2024-03-15"
    for kind in ("video", "image", "audio"):
        for entry in cat[kind]:
            date = entry["date"]
            if date is not None:
                datetime.strptime(date, "%Y-%m-%d")  # raises if impossible
