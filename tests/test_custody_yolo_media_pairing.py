"""The detector pairs a YOLO csv with its clip by name, in one directory.

`staging.py` files media and detection CSVs into the same YYYY/MM/DD directory,
which is what `walk_date_dirs` already descends into, so the pairing is a
sibling lookup. It used to probe for exactly `{key}.MP4`.

That probe is the reason these tests exist. Staging deliberately preserves the
source card's extension, so a card that wrote `.mp4` produced keys whose clips
were present, playable, and skipped - one log line each, no detections, no error.
An empty result is indistinguishable from a quiet stretch of road.
"""
from __future__ import annotations

import pytest

from auto_ingest.dashcam.media_pairing import (
    detection_stem,
    find_file_keys,
    find_sibling_media,
    missing_media,
)


def _date_dir(tmp_path):
    d = tmp_path / "2026" / "08" / "29"
    d.mkdir(parents=True)
    return d


@pytest.mark.parametrize("name", ["2026_0829_123850_F.MP4", "2026_0829_123850_F.mp4"])
def test_the_clip_is_found_whatever_case_its_extension_carries(tmp_path, name):
    """The bug. `.MP4` is what this dashcam writes; `.mp4` is what another
    firmware writes, and the exact-name probe silently skipped all of them."""
    d = _date_dir(tmp_path)
    (d / name).write_bytes(b"not really a video")
    assert find_sibling_media(str(d), "2026_0829_123850_F") == str(d / name)


@pytest.mark.parametrize("ext", [".MOV", ".mkv", ".Mkv"])
def test_other_recordable_containers_pair_too(tmp_path, ext):
    d = _date_dir(tmp_path)
    (d / f"2026_0829_123850_F{ext}").write_bytes(b"clip")
    assert find_sibling_media(str(d), "2026_0829_123850_F") is not None


def test_an_empty_clip_is_still_missing(tmp_path):
    """A zero-byte file is a truncated recording, not a usable one."""
    d = _date_dir(tmp_path)
    (d / "2026_0829_123850_F.MP4").write_bytes(b"")
    assert find_sibling_media(str(d), "2026_0829_123850_F") is None


def test_a_non_media_sibling_is_not_mistaken_for_the_clip(tmp_path):
    d = _date_dir(tmp_path)
    (d / "2026_0829_123850_F.YOLOv8n.csv").write_bytes(b"k,v\n")
    (d / "2026_0829_123850_F.txt").write_bytes(b"notes")
    assert find_sibling_media(str(d), "2026_0829_123850_F") is None


def test_find_file_keys_accepts_a_lowercase_card(tmp_path):
    """End to end through the csv, which is where the loss was observed."""
    d = _date_dir(tmp_path)
    (d / "2026_0829_123850_F.mp4").write_bytes(b"clip")
    (d / "2026_0829_123850_R.MP4").write_bytes(b"clip")
    (d / "2026_0829_123850_F_YOLOv8n.csv").write_bytes(b"k,v\n")
    (d / "2026_0829_123850_R_YOLOv8n.csv").write_bytes(b"k,v\n")
    assert find_file_keys(str(d)) == ["2026_0829_123850_F", "2026_0829_123850_R"]


def test_a_csv_whose_clip_is_absent_is_reported_not_invented(tmp_path):
    d = _date_dir(tmp_path)
    (d / "2026_0829_123850_F_YOLOv8n.csv").write_bytes(b"k,v\n")
    assert find_file_keys(str(d)) == []


def test_a_staged_clip_and_its_csv_are_found_together(tmp_path):
    """The layout contract, asserted from the detector's side: same directory,
    same stem, so the csv resolves to its clip without a path search."""
    from auto_ingest.custody.staging import destination_for

    media_key = "DCIM/Movie/2026_0829_123850_F.MP4"
    det_key = "yolo/2026_0829_123850_F_YOLOv8n.csv"
    d = _date_dir(tmp_path)
    for src in (media_key, det_key):
        landed = destination_for(src).destination_key
        assert landed.startswith("2026/08/29/"), landed
        (tmp_path / landed).write_bytes(b"x")
    assert find_file_keys(str(d)) == ["2026_0829_123850_F"]


def test_the_detection_suffix_is_matched_case_insensitively(tmp_path):
    assert detection_stem("2026_0829_123850_F_YOLOv8n.csv") == "2026_0829_123850_F"
    assert detection_stem("2026_0829_123850_F_yolov8n.CSV") == "2026_0829_123850_F"
    assert detection_stem("2026_0829_123850_F.mp4") is None


def test_a_csv_without_its_clip_is_reported_not_swallowed(tmp_path):
    """"No detections" and "never looked at" are indistinguishable downstream
    otherwise, so the gap is named."""
    d = _date_dir(tmp_path)
    (d / "2026_0829_123850_F_YOLOv8n.csv").write_bytes(b"k,v\n")
    (d / "2026_0829_123850_R.MP4").write_bytes(b"clip")
    (d / "2026_0829_123850_R_YOLOv8n.csv").write_bytes(b"k,v\n")
    assert missing_media(str(d)) == [("2026_0829_123850_F",
                                      "2026_0829_123850_F_YOLOv8n.csv")]
    assert find_file_keys(str(d)) == ["2026_0829_123850_R"]


def test_an_unreadable_directory_is_empty_not_an_exception(tmp_path):
    missing_dir = tmp_path / "not-there"
    assert find_file_keys(str(missing_dir)) == []
    assert find_sibling_media(str(missing_dir), "any") is None
    assert missing_media(str(missing_dir)) == []


def test_importing_the_pairing_logic_needs_no_gpu_image():
    """The regression this module exists to prevent: pure path logic stranded
    behind moviepy/cv2/ultralytics, so it could only be tested on a machine that
    had all three."""
    # Reaching this line at all is the assertion: collection imported
    # media_pairing through the package, and moviepy/cv2/ultralytics are absent
    # on this machine. `assert not _HEAVY` would say nothing extra.
    from auto_ingest.dashcam import media_pairing

    assert media_pairing.MEDIA_EXTENSIONS == (".mp4", ".mov", ".mkv")


def test_the_lazy_package_still_exposes_its_original_names():
    import auto_ingest.dashcam as dashcam

    assert set(dashcam.__all__) == {"run_yolo_embeddings", "yolo_embeddings"}
    with pytest.raises(AttributeError):
        dashcam.definitely_not_a_real_attribute
