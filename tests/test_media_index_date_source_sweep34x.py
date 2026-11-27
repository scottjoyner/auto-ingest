"""Regression tests: media_index date extraction must prefer the media filename.

Hermetic: tiny empty files under pytest tmp_path, fixed constants, no network,
no model load, no DB, no camera, no credentials, no live mounts.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from media_index import _classify, _date_for, scan_root

EARLY = "IMG_20260714_120000.mp4"
LATE = "IMG_20260801_090000.mp4"


def _make_tree(tmp_path: Path) -> Path:
    """Two files sharing one date-bearing ancestor directory (YYYY/MM/DD)."""
    root = tmp_path / "catalog"
    day = root / "2025" / "12" / "31"
    day.mkdir(parents=True)
    for name in (LATE, EARLY):  # reverse insertion: equal inherited dates keep wrong order
        (day / name).write_bytes(b"")
    return root


def test_date_for_prefers_filename_capture_date(tmp_path):
    """DEFECT: a parent directory date must not shadow the filename date."""
    root = _make_tree(tmp_path)
    p = root / "2025" / "12" / "31" / EARLY
    assert _date_for(p) == "2026-07-14"


def test_scan_root_records_filename_dates(tmp_path):
    """DEFECT: catalog entries must carry each file's own capture date."""
    root = _make_tree(tmp_path)
    cat = scan_root(str(root), progress=False)
    dates = sorted(e["date"] for e in cat["video"])
    assert dates == ["2026-07-14", "2026-08-01"]


def test_date_for_distinguishes_siblings_in_one_dated_directory(tmp_path):
    """DEFECT: sibling media in one dated tree must keep distinct capture dates."""
    root = _make_tree(tmp_path)
    day = root / "2025" / "12" / "31"
    assert (_date_for(day / EARLY), _date_for(day / LATE)) == (
        "2026-07-14",
        "2026-08-01",
    )


def test_date_for_still_uses_ancestor_when_filename_has_no_date(tmp_path):
    """CONTROL: legacy trees with date-less filenames keep the old fallback."""
    p = tmp_path / "photos" / "2025" / "06" / "18" / "DSC_001.jpg"
    assert _date_for(p) == "2025-06-18"


def test_date_for_still_uses_mtime_when_no_date_in_path(tmp_path):
    """CONTROL: the mtime fallback is unchanged (fixed UTC timestamp)."""
    p = tmp_path / "clip.mp4"
    p.write_bytes(b"")
    epoch = datetime(2024, 3, 9, 12, 0, tzinfo=timezone.utc).timestamp()
    os.utime(p, (epoch, epoch))
    assert _date_for(p) == "2024-03-09"


def test_classify_control():
    """CONTROL: classification is untouched by this change."""
    assert _classify(".MP4") == "video"
    assert _classify(".jpg") == "image"
    assert _classify(".WAV") == "audio"
    assert _classify(".txt") is None
