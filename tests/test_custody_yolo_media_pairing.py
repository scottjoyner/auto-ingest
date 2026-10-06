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

import os

import pytest

from auto_ingest.dashcam.media_pairing import (
    detection_stem,
    find_file_keys,
    find_sibling_media,
    is_quarantined,
    missing_media,
    walk_date_dirs,
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


# ---------------------------------------------------------------------------
# The quarantine namespace must not be walked as work
# ---------------------------------------------------------------------------
# `custody stage` routes detections whose clip is absent into
# `orphaned-detections/`, and that path is a YYYY/MM/DD shape. A walker that
# recognises date directories therefore descends into it and finds no clip for
# anything - on the real card, 2,934 directories and one warning each, per pass,
# forever. The files are meant to be kept, not processed.

def _tree(root):
    """A scan root holding live date trees and a quarantined one."""
    for rel in ("2026/08/29", "2026/08/30",
                "orphaned-detections/2024/07/13",
                "orphaned-detections/2024/07/14"):
        (root / rel).mkdir(parents=True)
    return root


def test_quarantined_date_trees_are_not_offered_as_work(tmp_path):
    found = walk_date_dirs(str(_tree(tmp_path)))
    assert [p.split("/")[-3:] for p in found] == [
        ["2026", "08", "29"], ["2026", "08", "30"]]


def test_the_exclusion_is_pruned_at_the_walk_not_filtered_after(tmp_path):
    """Pruning, not filtering.

    The point of mutating ``dirs`` inside the walk is that os.walk never descends
    into the excluded tree at all. Filtering the results afterwards would produce
    the same list while still stat()ing 2,934 directories on a CIFS share - which
    is the cost being avoided. So this asserts on what os.walk *yielded*, not on
    what came back.
    """
    root = _tree(tmp_path)
    real_walk = os.walk
    yielded = []

    def recording_walk(*args, **kwargs):
        for item in real_walk(*args, **kwargs):
            yielded.append(item[0])
            yield item

    os.walk = recording_walk
    try:
        walk_date_dirs(str(root))
    finally:
        os.walk = real_walk

    assert yielded, "the recording walk saw nothing at all"
    assert not any("orphaned-detections" in p for p in yielded), (
        "the quarantine tree was descended into and only filtered on the way out")


def test_the_exclusion_can_be_turned_off(tmp_path):
    """A tree that is not a quarantine today may be tomorrow's live work."""
    assert len(walk_date_dirs(str(_tree(tmp_path)), exclude=())) == 4


def test_is_quarantined_matches_a_component_not_a_substring(tmp_path):
    base = "/nas/fileserver/dashcam/sdcard-ingest"
    assert is_quarantined(f"{base}/orphaned-detections/2024/07/13")
    assert is_quarantined("/orphaned-detections")
    assert not is_quarantined(f"{base}/2026/08/29")
    # A directory merely containing the word is not the quarantine namespace.
    assert not is_quarantined(f"{base}/orphaned-detections-old/2024/07/13")
    assert not is_quarantined("/nas/fileserver/dashcam")
    assert not is_quarantined("/nas/fileserver/dashcam/orphaned-detectionsx/2024")


def test_quarantine_names_replace_the_default_rather_than_add_to_it(tmp_path):
    """Passing `exclude` is a complete statement of what is quarantined, so a
    caller who names their own namespace is not silently also subject to ours -
    and, equally, is not accidentally opted out of it.
    """
    root = _tree(tmp_path)
    (root / "keep-me-out" / "2024" / "07" / "13").mkdir(parents=True)

    ours = walk_date_dirs(str(root))
    assert not any("orphaned-detections" in p for p in ours)
    assert any("keep-me-out" in p for p in ours), "default does not know that name"

    theirs = walk_date_dirs(str(root), exclude=("keep-me-out",))
    assert not any("keep-me-out" in p for p in theirs)
    assert any("orphaned-detections" in p for p in theirs), (
        "a custom list replaces the default; it is not merged with it")


def test_a_real_quarantine_tree_pairs_nothing_and_is_still_preserved(tmp_path):
    """The end state, stated as a test: the files stay on disk, untouched, and
    simply are not offered to the detector."""
    quarantine = tmp_path / "orphaned-detections" / "2024" / "07" / "13"
    quarantine.mkdir(parents=True)
    csv = quarantine / "2024_0713_112243_F_YOLOv8n.csv"
    csv.write_text("Key,vehicle_id\n2024_0713_112243_F_1_car_1, 1\n")

    assert find_file_keys(str(quarantine)) == []
    assert missing_media(str(quarantine)) == [
        ("2024_0713_112243_F", "2024_0713_112243_F_YOLOv8n.csv")]
    assert csv.exists(), "preserved, not cleaned up"
    assert walk_date_dirs(str(tmp_path)) == []


def test_neither_wired_entry_point_reintroduces_the_unpruned_walk():
    """Two scripts reach this tree: runall.sh -> dashcam_yolo_embeddings_ents.py,
    and bulk_ingest_dashcam.sh -> auto_ingest/dashcam/yolo_embeddings.py.

    Patching one would have left the scheduled run walking the quarantine, so both
    are asserted to delegate rather than to open-code the walk. The check is on the
    body: a copy of the old `for root, dirs, files in os.walk(...)` loop is
    exactly what this guards against.
    """
    import ast
    import pathlib

    repo = pathlib.Path(__file__).resolve().parents[1]
    for rel in ("dashcam_yolo_embeddings_ents.py",
                "auto_ingest/dashcam/yolo_embeddings.py"):
        source = (repo / rel).read_text(encoding="utf-8")
        tree = ast.parse(source)
        fn = next(node for node in tree.body
                  if isinstance(node, ast.FunctionDef)
                  and node.name == "walk_date_dirs")
        walked = any(isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Attribute)
                     and node.func.attr == "walk"
                     for node in ast.walk(fn))
        assert not walked, (
            f"{rel} open-codes the walk again and will descend into quarantines")
        assert "media_pairing" in ast.unparse(fn), (
            f"{rel} must delegate to the shared walker")
