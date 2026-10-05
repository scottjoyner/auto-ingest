"""The staging layout the ingest pipeline will actually find.

These are the cases the surveyed card produced and the shapes the discovery
code in `auto_ingest/ingest/transcripts.py:94-99` and
`auto_ingest/dashcam/yolo_embeddings.py:1695-1697` actually match. Every one of
them was a real bug during development:

* the date directory was built by slicing, and the key carries underscores
  between its fields, so `[4:6]` is ``_0`` and the path came out
  ``2026/_0/82`` - plausible, and undiscoverable;
* the key pattern stopped at the six time digits and dropped the camera suffix,
  so the front and rear cameras of one moment staged onto one path;
* with the suffix restored, the alternation ``(F|R|FR)`` matched ``F`` first and
  left the ``R`` behind, staging ``..._FR.mp4`` as ``..._F.mp4``.

A staged layout that silently produces the wrong filename is worse than one that
refuses, because the copy succeeds and the pipeline finds nothing.
"""
from __future__ import annotations

import pytest

from auto_ingest.custody.staging import (
    DETECTION_CSV_ROLE,
    OTHER_ROLE,
    RTTM_ROLE,
    TRANSCRIPT_TXT_ROLE,
    VIDEO_ROLE,
    classify,
    date_directory,
    derive_key,
    destination_for,
    in_scope,
    plan_staging,
    split_camera,
)

# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name, expected", [
    ("2026_0829_123850_F.MP4", VIDEO_ROLE),
    ("clip.mp4", VIDEO_ROLE),
    ("MOVI0000.avi", VIDEO_ROLE),
    ("interview.m4a", "audio"),
    ("speech.flac", "audio"),
    ("2024_0713_112243_F_YOLOv8n.csv", DETECTION_CSV_ROLE),
    ("2025_0202_171732_medium_transcription.txt", TRANSCRIPT_TXT_ROLE),
    ("2025_0202_171732_speakers.rttm", RTTM_ROLE),
    ("2026_0101_x_metadata.csv", "metadata_csv"),
    ("random.csv", "other_csv"),
    ("app.py", OTHER_ROLE),
    ("locations_2024-08-20.json", OTHER_ROLE),
])
def test_classification(name, expected):
    assert classify(name) == expected


def test_detection_csv_wins_over_the_generic_csv_arm():
    """A dashcam detection must not be filed as a plain csv."""
    assert classify("2024_0713_112243_F_YOLOv8n.csv") == DETECTION_CSV_ROLE
    assert classify("2024_0713_112243_F_YOLOv8n.csv") != "other_csv"


def test_classification_is_case_insensitive():
    """Source is vfat and destination exFAT; the card carries both spellings."""
    assert classify("2026_0829_123850_F.MP4") == classify("2026_0829_123850_f.mp4")
    assert classify("x_YOLOv8n.csv") == DETECTION_CSV_ROLE
    assert classify("x_yolov8n.csv") == DETECTION_CSV_ROLE


def test_another_programs_files_are_out_of_scope():
    """The surveyed card was 69% out of scope by file count."""
    assert in_scope(VIDEO_ROLE) is True
    assert in_scope("audio") is True
    assert in_scope(OTHER_ROLE) is False
    assert in_scope("other_csv") is False


# ---------------------------------------------------------------------------
# Key derivation
# ---------------------------------------------------------------------------

def test_a_key_is_found_in_the_name_or_the_path():
    """canonicalize_key searches the full path too (transcripts.py:236-242)."""
    assert derive_key("DCIM/2026_0829_123850_F.MP4") == "2026_0829_123850_F"
    assert derive_key("2026/08/29/plain-name.mp4") is None
    assert derive_key("2026/08/29/2026_0829_123850.mp4") == "2026_0829_123850"


def test_the_camera_suffix_is_part_of_the_key():
    """It lives in the filename; a pattern stopping at the digits drops it."""
    front = derive_key("DCIM/2026_0829_123850_F.MP4")
    rear = derive_key("DCIM/2026_0829_123850_R.MP4")
    assert front != rear, "front and rear of one moment must not share a key"


def test_fr_is_not_matched_as_f():
    """Alternation order: (F|R|FR) matches F and strands the R."""
    assert derive_key("a/2024_0101_002438_FR.mp4") == "2024_0101_002438_FR"
    assert derive_key("a/2024_0101_002438_RF.mp4") == "2024_0101_002438_RF"
    assert derive_key("a/2025_0202_171732_BC-2.mp4") == "2025_0202_171732_BC-2"


def test_a_longer_digit_run_does_not_match_its_first_fourteen():
    assert derive_key("a/2026_0829_1238509.mp4") is None


def test_split_camera():
    assert split_camera("2026_0829_123850_F") == ("2026_0829_123850", "_F")
    assert split_camera("2026_0829_123850_BC-2") == ("2026_0829_123850", "_BC-2")
    assert split_camera("2026_0829_123850") == ("2026_0829_123850", None)


def test_no_key_means_no_key():
    """Undated media is reported, never given a fabricated one."""
    assert derive_key("VIDEO/MOVI0000.avi") is None
    obj = destination_for("VIDEO/MOVI0000.avi")
    assert obj.stageable is False
    assert obj.destination_key is None
    assert obj.reason == "no_YYYY_MMDD_HHMMSS_in_name_or_path"


# ---------------------------------------------------------------------------
# The date directory
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key, expected", [
    ("2026_0829_123850", "2026/08/29"),
    ("2025_0102_171732", "2025/01/02"),
    ("2024_0101_002438_FR", "2024/01/01"),
    ("2025_0202_171732_BC-2", "2025/02/02"),
])
def test_date_directory(key, expected):
    """The key carries underscores, so this cannot be done by slicing."""
    assert date_directory(key) == expected


def test_date_directory_rejects_a_non_key():
    with pytest.raises(ValueError):
        date_directory("not-a-key")


# ---------------------------------------------------------------------------
# Staged destinations
# ---------------------------------------------------------------------------

def test_media_keeps_its_camera_suffix():
    """Per-clip objects are distinct files; the detector pairs them exactly."""
    front = destination_for("DCIM/2026_0829_123850_F.MP4").destination_key
    rear = destination_for("DCIM/2026_0829_123850_R.MP4").destination_key
    assert front == "2026/08/29/2026_0829_123850_F.MP4"
    assert rear == "2026/08/29/2026_0829_123850_R.MP4"
    assert front != rear


def test_per_recording_sidecars_drop_the_camera_suffix():
    """One transcript per moment, shared by both cameras."""
    obj = destination_for("2025_0202_171732_F_medium_transcription.txt")
    assert obj.destination_key == "2025/02/02/2025_0202_171732.txt"
    assert obj.camera == "_F"
    assert obj.recording == "2025_0202_171732"


def test_a_detection_csv_pairs_with_its_clip_by_exact_stem():
    """yolo_embeddings.py:1695-1697 opens {stem}.MP4 where stem = rsplit(_YOLOv8n)."""
    csv_key = destination_for("yolo/2024_0713_112243_F_YOLOv8n.csv").destination_key
    clip_key = destination_for("DCIM/2024_0713_112243_F.MP4").destination_key
    assert csv_key == "2024/07/13/2024_0713_112243_F_YOLOv8n.csv"
    assert clip_key == "2024/07/13/2024_0713_112243_F.MP4"
    assert csv_key.rsplit("_YOLOv8n", 1)[0] + ".MP4" == clip_key


def test_out_of_scope_objects_are_reported_not_dropped():
    # A timestamp is derived here, and it is still refused: content, not timing,
    # decides scope.
    png = destination_for("heatmap/2024_0101_002438_F_heatmap.png")
    assert png.stageable is False
    assert png.reason == "out_of_scope:other"


def test_an_out_of_scope_object_is_reported_as_out_of_scope_not_as_unkeyed():
    """Both of these have no derivable key. Only one of them has a problem.

    `overland/locations_*.json` would never be staged whatever it were named, so
    reporting "no timestamp" describes a fault that does not exist and buries the
    one that does. `VIDEO/MOVI0000.avi` is a different case entirely: genuine
    dashcam footage that is unstaged only because nothing identifies its moment,
    and that is a decision an operator needs to see.
    """
    location = destination_for("overland/locations_2024-08-20-08-35-14.json")
    assert location.stageable is False
    assert location.reason == "out_of_scope:other"

    undated_clip = destination_for("VIDEO/MOVI0000.avi")
    assert undated_clip.stageable is False
    assert undated_clip.reason == "no_YYYY_MMDD_HHMMSS_in_name_or_path"


# ---------------------------------------------------------------------------
# The plan over a whole card
# ---------------------------------------------------------------------------

CARD_KEYS = (
    "DCIM/2026_0829_123850_F.MP4",
    "DCIM/2026_0829_123850_R.MP4",
    "DCIM/2026_0829_124157_F.MP4",
    "yolo/2024_0713_112243_F_YOLOv8n.csv",
    "yolo/2024_0713_112243_R_YOLOv8n.csv",
    "2025_0202_171732_medium_transcription.txt",
    "2025_0202_171732_speakers.rttm",
    "heatmap/2024_0101_002438_F_heatmap.png",
    "overland/locations_2024-08-20-08-35-14.json",
    "overland/app.py",
    "VIDEO/MOVI0000.avi",
)


def test_plan_splits_stageable_from_not():
    plan = plan_staging(CARD_KEYS)
    assert len(plan.staged) == 7
    assert len(plan.unstaged) == 4


def test_plan_has_no_collisions_on_a_well_formed_card():
    """The front/rear pair must not collapse, or the copy would lose one."""
    assert plan_staging(CARD_KEYS).collisions() == ()


def test_plan_surfaces_a_genuine_collision_instead_of_resolving_it():
    """Two different sources claiming one staged path is a decision, not a merge."""
    plan = plan_staging((
        "a/2026_0829_123850_F.MP4",
        "b/2026_0829_123850_F.MP4",
    ))
    assert plan.collisions() == ("2026/08/29/2026_0829_123850_F.MP4",)


def test_plan_is_deterministic():
    """Two runs over an unchanged card must produce an identical plan."""
    forward = plan_staging(CARD_KEYS)
    backward = plan_staging(tuple(reversed(CARD_KEYS)))
    assert [o.source_key for o in forward.staged] == \
        [o.source_key for o in backward.staged]
    assert forward.to_dict() == backward.to_dict()


def test_plan_summary_is_bounded_by_reasons_not_files():
    summary = plan_staging(CARD_KEYS).to_dict()
    assert summary["staged_files"] == 7
    assert summary["unstaged_files"] == 4
    # the counts are per reason/role, never one row per object
    assert set(summary["unstaged_by_reason"]) <= {
        "no_YYYY_MMDD_HHMMSS_in_name_or_path", "out_of_scope:other",
    }
    assert summary["by_role"][VIDEO_ROLE] == 3
    assert len(summary["by_role"]) <= 12


def test_a_plan_with_no_objects_is_empty_not_an_error():
    plan = plan_staging(())
    assert plan.staged == () and plan.unstaged == ()
    assert plan.collisions() == ()
