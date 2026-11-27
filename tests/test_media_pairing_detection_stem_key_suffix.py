"""A detection csv names exactly one clip stem: ``{key}_YOLOv8n.csv`` -> ``key``.

``detection_stem`` used an ``rsplit("_YOLOv8n", 1)`` *after* the anchored,
case-insensitive regex had already removed the single real detection suffix.
That second, case-sensitive strip removes a legitimate trailing ``_YOLOv8n``
from the clip key itself, so a clip whose key ends in that marker is paired
with a different (usually absent) file and reported as missing media.

These tests are hermetic: pure string/path logic plus tiny local fixtures, no
DB, network, model load, service or mount.
"""

from __future__ import annotations

from auto_ingest.dashcam.media_pairing import detection_stem, find_file_keys, missing_media

_KEY = "2026_0829_123850_F_YOLOv8n"


def test_key_ending_in_marker_is_not_stripped_twice():
    """The csv for a clip whose key ends in _YOLOv8n must recover that whole key."""
    assert detection_stem(f"{_KEY}_YOLOv8n.csv") == _KEY


def test_end_to_end_pairing_of_a_key_ending_in_marker(tmp_path):
    """The clip is present and playable; it must not be reported as missing."""
    d = tmp_path / "2026" / "08" / "29"
    d.mkdir(parents=True)
    (d / f"{_KEY}.mp4").write_bytes(b"clip")
    (d / f"{_KEY}_YOLOv8n.csv").write_bytes(b"k,v\n")
    assert find_file_keys(str(d)) == [_KEY]
    assert missing_media(str(d)) == []


def test_ordinary_key_is_still_recovered():
    """Control: a key without an embedded marker is unchanged (regression guard)."""
    assert detection_stem("2026_0829_123850_F_YOLOv8n.csv") == "2026_0829_123850_F"


def test_case_variant_suffix_is_still_recovered():
    """Control: the suffix regex is case-insensitive; an ordinary key survives."""
    assert detection_stem("2026_0829_123850_F_yolov8n.CSV") == "2026_0829_123850_F"


def test_lowercase_embedded_marker_is_not_stripped():
    """Control: the old case-sensitive rsplit happened to leave this key intact."""
    assert detection_stem("K_yolov8n_YOLOv8n.csv") == "K_yolov8n"


def test_non_detection_file_is_not_a_detection():
    """Control: ordinary media files are still rejected."""
    assert detection_stem("2026_0829_123850_F.mp4") is None
