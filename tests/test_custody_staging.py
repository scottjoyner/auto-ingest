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

import datetime

import pytest

from auto_ingest.custody.staging import (
    DETECTION_CSV_ROLE,
    OTHER_ROLE,
    RTTM_ROLE,
    TRANSCRIPT_TXT_ROLE,
    VIDEO_ROLE,
    classify,
    date_directory,
    date_from_mtime,
    derive_key,
    destination_for,
    in_scope,
    plan_staging,
    split_camera,
)

# From `.discovery`, not from `transcripts`: these are the same objects
# (transcripts re-exports them), but discovery imports no ML stack, so this guard
# against the real naming contract runs on a machine with no torch and no GPU
# image - which is exactly where a custody test belongs.
from auto_ingest.ingest import discovery as _tx
from auto_ingest.ingest.discovery import canonicalize_key

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
    """One transcript per moment, shared by both cameras.

    The camera goes; the model tag stays. Dropping the tag as well is the bug
    this file kept catching - see test_renaming_a_sidecar_into_the_pipeline_is_what_would_break_it.
    """
    obj = destination_for("2025_0202_171732_F_medium_transcription.txt")
    assert obj.destination_key == "2025/02/02/2025_0202_171732_medium_transcription.txt"
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


# ---------------------------------------------------------------------------
# The destination names must satisfy the pipeline's own patterns
# ---------------------------------------------------------------------------
# Asserting against a hand-copied regex proves only that the copy of the regex
# agrees with itself. These import the real patterns, so a change to discovery
# that invalidates the layout fails here rather than silently at ingest time.

#: canonicalize_key converts a local filename stamp to UTC, so this is NOT
#: "2025_0202_171732" and asserting that would encode a bug. Derived from the
#: pipeline itself rather than hardcoded, so it stays true under any TZ.
CANONICAL_1717 = canonicalize_key("2025_0202_171732", "2025/02/02/2025_0202_171732_medium_transcription.txt")


def _key_as_the_pipeline_computes_it(landed: str) -> str:
    """Reproduce what discovery does after a pattern matches.

    transcripts.py strips the matched tail and hands the remainder plus the full
    path to canonicalize_key. Calling it with the whole filename instead would
    test a call site that does not exist.
    """
    import posixpath

    name = posixpath.basename(landed)
    for pattern in (_tx.PAT_TRANS_JSON_TXT, _tx.PAT_ENTITIES, _tx.PAT_RTTM,
                    _tx.PAT_META_CSV, _tx.PAT_TRANS_CSV):
        m = pattern.search(name)
        if m:
            return canonicalize_key(name[:m.start()], landed)
    raise AssertionError(f"no discovery pattern matched {landed}")


def test_staged_sidecars_are_matched_by_the_real_discovery_patterns():
    cases = {
        "2025_0202_171732_medium_transcription.txt": _tx.PAT_TRANS_JSON_TXT,
        "2025_0202_171732_transcription.csv": _tx.PAT_TRANS_CSV,
        "2025_0202_171732_speakers.rttm": _tx.PAT_RTTM,
        "2025_0202_171732_metadata.csv": _tx.PAT_META_CSV,
        "2025_0202_171732_transcription_entities.csv": _tx.PAT_ENTITIES,
        "2026_0829_123850_F.MP4": _tx.PAT_MEDIA,
    }
    for source_key, pattern in cases.items():
        landed = destination_for(source_key).destination_key
        assert landed, source_key
        assert pattern.search(landed), (
            f"{source_key} stages to {landed}, which "
            f"{pattern.pattern} does not match"
        )


def test_a_staged_sidecar_still_yields_a_canonical_key():
    """Discovery finds the file, then canonicalize_key must rebuild its key from
    the staged location - otherwise the sidecar joins nothing."""
    for source_key in ("2025_0202_171732_medium_transcription.txt",
                       "2025_0202_171732_speakers.rttm"):
        landed = destination_for(source_key).destination_key
        assert _key_as_the_pipeline_computes_it(landed) == CANONICAL_1717


def test_renaming_a_sidecar_into_the_pipeline_is_what_would_break_it():
    """The bug this guards, stated as a negative test.

    Dropping the marker is not a cosmetic choice: `2025_0202_171732.txt` matches
    nothing at all, and every model that transcribed one moment would claim the
    same path.
    """
    broken = "2026/02/02/2025_0202_171732.txt"
    assert not _tx.PAT_TRANS_JSON_TXT.search(broken)
    assert not _tx.PAT_RTTM.search(broken)
    assert not _tx.PAT_META_CSV.search(broken)


def test_per_recording_sidecars_of_both_cameras_share_one_transcript():
    front = destination_for("2025_0202_171732_F_medium_transcription.txt")
    rear = destination_for("2025_0202_171732_R_medium_transcription.txt")
    assert front.destination_key == rear.destination_key
    assert front.camera == "_F" and rear.camera == "_R"
    assert _tx.PAT_TRANS_JSON_TXT.search(front.destination_key)


def test_per_clip_objects_of_both_cameras_never_share_a_path():
    """The complementary rule: the detector opens {stem}.MP4 by exact name."""
    pairs = [
        ("DCIM/Movie/2026_0829_123850_F.MP4", "yolo/2026_0829_123850_F_YOLOv8n.csv"),
        ("DCIM/Movie/2026_0829_123850_R.MP4", "yolo/2026_0829_123850_R_YOLOv8n.csv"),
    ]
    for media, det in pairs:
        m = destination_for(media).destination_key
        d = destination_for(det).destination_key
        assert m != d
        stem = d.rsplit("_YOLOv8n", 1)[0]
        assert stem.endswith(m.rsplit(".", 1)[0]), (
            f"{d} does not resolve to {m} via the detector's rsplit"
        )


def test_a_path_that_merely_looks_like_a_date_yields_no_key():
    """`2026/08/29` is not a moment. Folding the directory into a key would
    invent `2026_0829_29` and stage four unrelated clips onto one path."""
    obj = destination_for("2026/08/29/plain-name.MP4")
    assert obj.stageable is False
    assert obj.reason == "no_YYYY_MMDD_HHMMSS_in_name_or_path"


def test_the_date_directory_is_the_one_canonicalize_key_looks_for():
    """It re-searches the full path for /YYYY/MM/DD/ when the name has no
    stamp (transcripts.py:236-242), so the layout feeds its date lookup."""
    landed = destination_for("2026_0829_123850_F.MP4").destination_key
    assert landed.startswith("2026/08/29/")
    import re
    assert re.search(r"/(?P<Y>\d{4})/(?P<M>\d{2})/(?P<D>\d{2})/", "/" + landed)


def test_the_model_tag_is_what_distinguishes_two_transcriptions():
    """Two models, one moment: distinct files, and both discoverable."""
    small = destination_for("2025_0202_171732_small_transcription.txt")
    large = destination_for("2025_0202_171732_large-v2_transcription.txt")
    assert small.destination_key != large.destination_key
    for obj in (small, large):
        assert _tx.PAT_TRANS_JSON_TXT.search(obj.destination_key)
        assert _key_as_the_pipeline_computes_it(obj.destination_key) == CANONICAL_1717


# ---------------------------------------------------------------------------
# Local directory, UTC key
# ---------------------------------------------------------------------------

def _media_key_as_discovery_computes_it(landed: str) -> str:
    """Mirror transcripts.py:856, which hands canonicalize_key the basename with
    its extension removed."""
    import posixpath
    return canonicalize_key(posixpath.basename(landed).rsplit(".", 1)[0], landed)


def test_the_date_directory_is_local_while_the_canonical_key_is_utc():
    """An evening clip lands in its local date directory and keys to the next
    UTC day. Both halves are deliberate and neither is a bug.

    Staging groups by the digits the operator sees on the card, so 23:30 on the
    29th is filed under the 29th. `canonicalize_key` converts a local stamp to UTC
    (:222-224), so the same clip joins the rest of the corpus as the 30th.

    This is safe because staging always keeps the whole stamp in the filename.
    Discovery therefore parses the NAME and never consults the directory - the
    `/YYYY/MM/DD/` fallback at transcripts.py:236-242 exists for names that carry
    no stamp at all, which staging never produces.
    """
    media = destination_for("DCIM/2026_0829_233033_F.MP4")
    detection = destination_for("yolo/2026_0829_233033_F_YOLOv8n.csv")

    assert media.destination_key.startswith("2026/08/29/"), "filed by local date"
    key = _media_key_as_discovery_computes_it(media.destination_key)
    assert key.startswith("2026_0830"), f"keyed in UTC, got {key}"
    assert _media_key_as_discovery_computes_it(detection.destination_key) == key, (
        "media and its CSV must still join despite the local/UTC split"
    )


def test_a_bodily_date_directory_never_becomes_the_key():
    """`2026/08/29` is not a moment. Folding it in would stage a whole day onto
    one path and collide it with every other clip from that day."""
    obj = destination_for("2026/08/29/clip.MP4")
    assert obj.stageable is False
    assert obj.reason == "no_YYYY_MMDD_HHMMSS_in_name_or_path"


# ---------------------------------------------------------------------------
# Clips and detections that do not meet
# ---------------------------------------------------------------------------

def test_pairing_counts_a_fresh_card_with_no_detections_yet():
    plan = plan_staging([
        "DCIM/Movie/2026_0829_123850_F.MP4",
        "DCIM/Movie/2026_0829_123850_R.MP4",
    ])
    assert plan.pairing() == {
        "clips": 2, "detections": 0, "paired": 0,
        "detections_without_clip": 0, "clips_without_detection": 2,
    }


def test_pairing_counts_a_clip_with_its_detection():
    plan = plan_staging([
        "DCIM/Movie/2026_0829_123850_F.MP4",
        "yolo/2026_0829_123850_F_YOLOv8n.csv",
    ])
    assert plan.pairing()["paired"] == 1
    assert plan.pairing()["detections_without_clip"] == 0
    assert plan.pairing()["clips_without_detection"] == 0


def test_pairing_reports_orphaned_detections_rather_than_hiding_them():
    """The state the real card is actually in: detections from a session whose
    media was already archived, plus fresh media with no detections yet.

    Nothing is wrong, and nothing pairs. Counted, because "2,934 staged files,
    none paired" otherwise reads as a naming failure.
    """
    plan = plan_staging([
        "DCIM/Movie/2026_0829_123850_F.MP4",
        "yolo/2024_0713_112243_F_YOLOv8n.csv",
        "yolo/2024_0713_112243_R_YOLOv8n.csv",
    ])
    summary = plan.pairing()
    assert summary == {
        "clips": 1, "detections": 2, "paired": 0,
        "detections_without_clip": 2, "clips_without_detection": 1,
    }


def test_pairing_does_not_pair_across_date_directories():
    """Same stem, different day: still unpaired. The detector looks in one
    directory, so a stem match across days is a coincidence, not a match."""
    plan = plan_staging([
        "DCIM/Movie/2026_0829_123850_F.MP4",
        "yolo/2026_0830_123850_F_YOLOv8n.csv",
    ])
    assert plan.pairing()["paired"] == 0


def test_pairing_ignores_per_recording_sidecars():
    """A transcript shares the stem but is not a detection, and must not be
    counted as one."""
    plan = plan_staging([
        "DCIM/Movie/2026_0829_123850_F.MP4",
        "2026_0829_123850_medium_transcription.txt",
    ])
    assert plan.pairing()["detections"] == 0
    assert plan.pairing()["clips"] == 1


# ---------------------------------------------------------------------------
# Undated media: the mtime fallback
# ---------------------------------------------------------------------------
# A DVR writes MOVI0000.avi with nothing identifying when it was recorded. These
# three files are the real ones off the card: no creation date in the AVI header,
# no ID3 chunk, and TAG:date=2010-06-29 identical in all of them because it is
# the encoder's build stamp. The filesystem mtime is the only date that exists.

def test_undated_media_is_not_staged_unless_the_operator_says_so():
    """The default matters more than the feature. Inventing a date for footage
    whose provenance is unknown is a decision, not a fallback."""
    assert destination_for("VIDEO/MOVI0000.avi").stageable is False
    assert destination_for("VIDEO/MOVI0000.avi").reason == (
        "no_YYYY_MMDD_HHMMSS_in_name_or_path")


def test_an_mtime_dates_the_group_without_renaming_the_file():
    mtime = datetime.datetime(2024, 8, 19, 17, 19, tzinfo=datetime.timezone.utc).timestamp()
    obj = destination_for("VIDEO/MOVI0000.avi", mtime=mtime, allow_mtime_key=True)
    assert obj.stageable is True
    assert obj.destination_key == "2024/08/19/MOVI0000.avi"
    assert obj.key_source == "mtime", "provenance must be recorded, not implied"
    assert obj.key is None, "no canonical key is claimed for an undated file"


def test_a_named_file_never_falls_back_to_its_mtime():
    """A file that knows its own date must not be re-dated by the filesystem,
    which on a card records when the copy was written."""
    mtime = datetime.datetime(1999, 1, 1, tzinfo=datetime.timezone.utc).timestamp()
    obj = destination_for("DCIM/2026_0829_123850_F.MP4", mtime=mtime,
                          allow_mtime_key=True)
    assert obj.destination_key == "2026/08/29/2026_0829_123850_F.MP4"
    assert obj.key_source == "filename"


def test_an_unset_card_clock_is_refused_rather_than_filed_under_1969():
    """The real card's directories carry 1969-12-31. Filing footage under
    1969/12/31/ would be a confident, wrong answer."""
    assert date_from_mtime(0) is None
    assert date_from_mtime(1) is None
    assert date_from_mtime(631_152_000 - 1) is None, "just below the floor"
    # Local, not UTC: this host is UTC-5/-4, so the floor instant is still the
    # previous evening here. That is the point - the group matches `ls`.
    local = datetime.datetime.fromtimestamp(631_152_000)
    assert date_from_mtime(631_152_000) == local.strftime("%Y/%m/%d")
    assert date_from_mtime(631_152_000) is not None


def test_the_mtime_fallback_still_declines_a_non_media_file():
    mtime = datetime.datetime(2024, 8, 19, tzinfo=datetime.timezone.utc).timestamp()
    obj = destination_for("overland/locations.json", mtime=mtime, allow_mtime_key=True)
    assert obj.stageable is False
    assert obj.reason == "out_of_scope:other"


def test_plan_reports_how_many_dates_were_not_camera_written():
    mtime = datetime.datetime(2024, 8, 19, tzinfo=datetime.timezone.utc).timestamp()
    plan = plan_staging(
        ["DCIM/2026_0829_123850_F.MP4", "VIDEO/MOVI0000.avi"],
        mtimes={"VIDEO/MOVI0000.avi": mtime},
        allow_mtime_key=True,
    )
    assert plan.by_key_source() == {"filename": 1, "mtime": 1}
    assert plan.collisions() == ()


def test_plan_without_the_flag_reports_no_mtime_dates():
    plan = plan_staging(["VIDEO/MOVI0000.avi"])
    assert plan.by_key_source() == {}
    assert len(plan.unstaged) == 1


def test_the_recorded_layout_carries_the_date_provenance(tmp_path):
    """The record that outlives the run has to distinguish a camera-written date
    from a filesystem one. Without this the ledger claims a date and not its
    reliability."""
    import json

    from auto_ingest.custody.executor import write_staged_ledger

    mtime = datetime.datetime(2024, 8, 19, 17, 19,
                              tzinfo=datetime.timezone.utc).timestamp()
    plan = plan_staging(
        ["DCIM/2026_0829_123850_F.MP4", "VIDEO/MOVI0000.avi"],
        mtimes={"VIDEO/MOVI0000.avi": mtime},
        allow_mtime_key=True,
    )
    path = write_staged_ledger(tmp_path / "b", plan)
    rows = {r["source_key"]: r
            for r in (json.loads(line) for line in path.read_text().splitlines())}
    assert rows["DCIM/2026_0829_123850_F.MP4"]["key_source"] == "filename"
    assert rows["VIDEO/MOVI0000.avi"]["key_source"] == "mtime"


# ---------------------------------------------------------------------------
# Orphaned detections: preserved, quarantined, never dropped
# ---------------------------------------------------------------------------
# On the real card, 2,934 detection files describe recordings from 2024 that are
# on neither the card nor the NAS. They are the only surviving record of that
# footage. So the requirement is not "handle them" - it is that nothing here may
# lose them, and that they must not sit in the live date tree looking paired.

def test_a_detection_without_its_clip_is_routed_to_a_labelled_namespace():
    plan = plan_staging([
        "DCIM/Movie/2026_0829_123850_F.MP4",
        "yolo/2026_0829_123850_F_YOLOv8n.csv",
        "yolo/2024_0713_112243_F_YOLOv8n.csv",
    ])
    landed = {o.source_key: o.destination_key for o in plan.staged}
    assert landed["yolo/2024_0713_112243_F_YOLOv8n.csv"] == (
        "orphaned-detections/2024/07/13/2024_0713_112243_F_YOLOv8n.csv")
    # The paired pair is untouched.
    assert landed["DCIM/Movie/2026_0829_123850_F.MP4"] == (
        "2026/08/29/2026_0829_123850_F.MP4")
    assert landed["yolo/2026_0829_123850_F_YOLOv8n.csv"] == (
        "2026/08/29/2026_0829_123850_F_YOLOv8n.csv")


def test_routing_preserves_every_object_and_changes_no_collisions():
    """The load-bearing property: routing moves paths, it never drops anything and
    it cannot introduce a collision, because prefixing is injective."""
    keys = [
        "DCIM/Movie/2026_0829_123850_F.MP4",
        "DCIM/Movie/2026_0829_123850_R.MP4",
        "yolo/2026_0829_123850_F_YOLOv8n.csv",
        "yolo/2026_0829_123850_R_YOLOv8n.csv",
        "yolo/2024_0713_112243_F_YOLOv8n.csv",
        "yolo/2024_0713_112243_R_YOLOv8n.csv",
        "yolo/2024_0714_101500_F_YOLOv8n.csv",
    ]
    routed = plan_staging(keys)
    plain = plan_staging(keys, orphan_prefix="")

    assert len(routed.staged) == len(keys) == len(plain.staged)
    assert routed.collisions() == plain.collisions() == ()
    assert len({o.source_key for o in routed.staged}) == len(keys)
    assert len({o.destination_key for o in routed.staged}) == len(keys)


def test_orphans_are_named_rather_than_swallowed():
    plan = plan_staging([
        "DCIM/Movie/2026_0829_123850_F.MP4",
        "yolo/2026_0829_123850_F_YOLOv8n.csv",
        "yolo/2024_0713_112243_F_YOLOv8n.csv",
        "yolo/2024_0713_112243_R_YOLOv8n.csv",
    ])
    orphans = plan.unpaired_detections()
    assert {o.source_key for o in orphans} == {
        "yolo/2024_0713_112243_F_YOLOv8n.csv",
        "yolo/2024_0713_112243_R_YOLOv8n.csv",
    }
    assert plan.pairing()["detections_without_clip"] == 2
    assert plan.pairing()["paired"] == 1


def test_pairing_is_decided_after_every_object_has_a_place():
    """A clip planned last still rescues its own csv. Deciding during the first
    pass would orphan it - the same bug as reading a file before the writer
    finished."""
    # The csv sorts before the clip, so a single forward pass would see no clip.
    plan = plan_staging([
        "DCIM/Movie/2026_0829_123850_F.MP4",
        "yolo/2026_0829_123850_F_YOLOv8n.csv",
    ])
    assert plan.unpaired_detections() == ()
    assert not any(o.destination_key.startswith("orphaned-detections")
                   for o in plan.staged)


def test_same_stem_on_a_different_day_is_still_an_orphan():
    """Coincidence, not a match: the detector looks beside the csv."""
    plan = plan_staging([
        "DCIM/Movie/2026_0829_123850_F.MP4",
        "yolo/2026_0830_123850_F_YOLOv8n.csv",
    ])
    assert len(plan.unpaired_detections()) == 1
    assert plan.staged[1].destination_key.startswith("orphaned-detections/")


def test_routing_can_be_turned_off_when_the_dates_line_up_again():
    """A later card may carry the media these detections describe. The namespace
    is a default, not a sentence - and with everything paired it is empty anyway."""
    keys = ["DCIM/Movie/2024_0713_112243_F.MP4",
            "yolo/2024_0713_112243_F_YOLOv8n.csv"]
    assert plan_staging(keys).unpaired_detections() == ()
    assert plan_staging(keys, orphan_prefix="").unpaired_detections() == ()
    assert plan_staging(keys, orphan_prefix=None).unpaired_detections() == ()


def test_a_clip_is_never_routed_as_an_orphan():
    """Only detections can be orphaned. A clip with no csv is simply not yet
    detected, which is the normal state of fresh footage."""
    plan = plan_staging(["DCIM/Movie/2026_0829_123850_F.MP4"])
    assert plan.unpaired_detections() == ()
    assert plan.staged[0].destination_key == "2026/08/29/2026_0829_123850_F.MP4"
