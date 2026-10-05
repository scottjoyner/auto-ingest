"""
Regression tests for the sidecar-stem reducer in auto_ingest.ingest.transcripts.

`file_key_from_name` used to strip the sidecar with
`_([A-Za-z0-9\\-\\._]+)_transcription`, whose character class spans "_", "-" and ".".
That made the match start at the FIRST "_" in the name and swallow the name as well
as the model id, so `IMG_4821_transcription.json` reduced to `IMG` and every `IMG_*`
recording collapsed onto one key (their transcripts then shared -- and overwrote -- one
mapping entry). Dashcam names survived only by accident: `canonicalize_key` re-searches
the full path for the `YYYY_MMDD_HHMMSS` timestamp the reducer had just eaten.

Technique is the one from test_ml_pure.py: the module is imported under a
`unittest.mock.patch` on `sys.modules` that stubs torch/transformers/neo4j/etc., so no
ML stack is loaded. Only deterministic, ML-free helpers are called.
"""
import importlib
import logging
import os
import sys
import unittest.mock as mock

import pytest

_STUBS = {
    "torch": mock.MagicMock(),
    "torchaudio": mock.MagicMock(),
    "torch.nn": mock.MagicMock(),
    "torch.nn.functional": mock.MagicMock(),
    "transformers": mock.MagicMock(),
    "soundfile": mock.MagicMock(),
    "faiss": mock.MagicMock(),
    "speechbrain": mock.MagicMock(),
    "speechbrain.pretrained": mock.MagicMock(),
    "pyannote": mock.MagicMock(),
    "pyannote.audio": mock.MagicMock(),
    "ultralytics": mock.MagicMock(),
    "ultralytics.utils": mock.MagicMock(),
    "ultralytics.utils.plotting": mock.MagicMock(),
    "cv2": mock.MagicMock(),
    "moviepy": mock.MagicMock(),
    "moviepy.editor": mock.MagicMock(),
    "PIL": mock.MagicMock(),
    "matplotlib": mock.MagicMock(),
    "neo4j": mock.MagicMock(),
    "neo4j.exceptions": mock.MagicMock(),
}


@pytest.fixture(scope="module")
def T():
    with mock.patch.dict(sys.modules, _STUBS):
        return importlib.import_module("auto_ingest.ingest.transcripts")


def _key(T, name, path=None):
    """The composition discover_keys() actually uses (transcripts.py:832-833)."""
    return T.canonicalize_key(T.file_key_from_name(name), path or ("/audio/" + name))


# ---------------------------------------------------------------------------
# 1. The five real filenames the greedy regex destroyed
# ---------------------------------------------------------------------------
def test_real_sidecar_names_reduce_to_their_stem(T):
    assert T.file_key_from_name("meeting_standup_2026-03-04_medium_transcription.json") == "meeting_standup_2026-03-04"
    assert T.file_key_from_name("2026-03-04_standup_medium_transcription.json") == "2026-03-04_standup"
    assert T.file_key_from_name("standup-2026-03-04_large_transcription.json") == "standup-2026-03-04"
    assert T.file_key_from_name("IMG_4821_transcription.json") == "IMG_4821"
    assert T.file_key_from_name("2025_0202_171732_medium_transcription.txt") == "2025_0202_171732"


# ---------------------------------------------------------------------------
# 2. The dashcam key shape is untouched (this is what the whole graph keys on)
# ---------------------------------------------------------------------------
def test_dashcam_key_shape_unchanged(T):
    # LOCAL_TZ is read per call inside _to_localized(); pin it so the UTC key is
    # deterministic on any machine (17:17:32 America/New_York == 22:17:32 UTC).
    with mock.patch.object(T, "LOCAL_TZ", "America/New_York"):
        for stem in ("2025_0202_171732_medium", "2025_0202_171732_large-v3",
                     "2025_0202_171732_tiny.en", "2025_0202_171732_small"):
            name = f"{stem}_transcription.txt"
            assert T.file_key_from_name(name) == "2025_0202_171732"
            assert _key(T, name, f"/dashcam/2025/02/02/20250202_171732/{name}") == "2025_0202_221732"
        # the model-id-less csv form the writers emit (whisper_audio_chunked.py:347):
        # same key, and the full stamp survives the reducer this time
        assert T.file_key_from_name("2025_0202_171732_transcription.csv") == "2025_0202_171732"
        assert _key(T, "2025_0202_171732_transcription.csv",
                    "/dashcam/2025/02/02/20250202_171732/2025_0202_171732_transcription.csv") == "2025_0202_221732"
    # bodycam camera variant: the _BC-1 is part of the stem, not a model id
    assert T.file_key_from_name("2025_0202_171732_BC-1_medium_transcription.txt") == "2025_0202_171732_BC-1"


# ---------------------------------------------------------------------------
# 3. IMG_* no longer collapses
# ---------------------------------------------------------------------------
def test_img_recordings_do_not_collapse(T):
    stems = {n: T.file_key_from_name(f"{n}_transcription.json")
             for n in ("IMG_4821", "IMG_9912", "IMG_0007")}
    assert stems == {"IMG_4821": "IMG_4821", "IMG_9912": "IMG_9912", "IMG_0007": "IMG_0007"}
    assert "IMG" not in set(stems.values())
    # three distinct recordings -> three distinct keys, so no two transcripts land in
    # one mapping entry (select_best_json would then pick one and drop the other)
    assert len({_key(T, f"{n}_transcription.json") for n in stems}) == 3


# ---------------------------------------------------------------------------
# 4. Two different recordings never share a key, end to end through discover_keys
# ---------------------------------------------------------------------------
def _scan_roots(T, tmp_path, monkeypatch):
    for attr in ("SCAN_ROOTS", "RTTM_DIRS"):
        monkeypatch.setattr(T, attr, [str(tmp_path)])
    for attr in ("DASHCAM_ROOT", "OLD_DASHCAM_ROOT"):
        monkeypatch.setattr(T, attr, str(tmp_path))
    monkeypatch.setattr(T, "LOCAL_TZ", "America/New_York")
    return tmp_path


def test_two_recordings_one_day_do_not_share_a_key(T, tmp_path, monkeypatch):
    """End to end through discover_keys, on the sidecar shapes it actually discovers:
    PAT_TRANS_JSON_TXT matches ``*_transcription.txt`` and PAT_TRANS_CSV matches
    ``*_transcription.csv`` (a bare ``*_transcription.json`` matches neither and is
    never discovered)."""
    root = _scan_roots(T, tmp_path, monkeypatch)
    for name in ("standup-2026-03-04_medium_transcription.txt",
                 "IMG_4821_medium_transcription.txt", "IMG_4821_large-v3_transcription.txt",
                 "IMG_4821_transcription.csv",
                 "IMG_9912_medium_transcription.txt", "IMG_9912_transcription.csv"):
        (root / name).write_text("{}")

    mapping = T.discover_keys()
    # one key per recording, and no key holding another recording's sidecars
    # (os.walk order is not stable, so compare sorted)
    assert {k: (sorted(os.path.basename(p) for p in v["json_all"]),
                sorted(os.path.basename(p) for p in v["csv_all"])) for k, v in mapping.items()} == {
        "standup-2026-03-04": (["standup-2026-03-04_medium_transcription.txt"], []),
        "IMG_4821": (["IMG_4821_large-v3_transcription.txt", "IMG_4821_medium_transcription.txt"],
                     ["IMG_4821_transcription.csv"]),
        "IMG_9912": (["IMG_9912_medium_transcription.txt"], ["IMG_9912_transcription.csv"]),
    }
    assert "IMG" not in mapping


def test_same_day_recordings_sharing_a_size_word_do_not_share_a_key(T, tmp_path, monkeypatch):
    """`medium`/`large` in a name is not a shared identity."""
    root = _scan_roots(T, tmp_path, monkeypatch)
    for name in ("kickoff_medium_notes_medium_transcription.txt",
                 "kickoff_medium_brief_medium_transcription.txt",
                 "kickoff_medium_notes_transcription.csv"):
        (root / name).write_text("{}")

    mapping = T.discover_keys()
    assert set(mapping) == {"kickoff_medium_notes", "kickoff_medium_brief"}
    assert [os.path.basename(p) for p in mapping["kickoff_medium_notes"]["json_all"]] == [
        "kickoff_medium_notes_medium_transcription.txt"]
    assert [os.path.basename(p) for p in mapping["kickoff_medium_notes"]["csv_all"]] == [
        "kickoff_medium_notes_transcription.csv"]
    assert [os.path.basename(p) for p in mapping["kickoff_medium_brief"]["json_all"]] == [
        "kickoff_medium_brief_medium_transcription.txt"]


# ---------------------------------------------------------------------------
# 5. Multi-part names with dots and dashes
# ---------------------------------------------------------------------------
def test_multi_part_names_with_dots_and_dashes_survive(T):
    assert T.file_key_from_name("my.recording.2026-03-04_medium_transcription.json") == "my.recording.2026-03-04"
    assert T.file_key_from_name("team-sync--2026.03.04--standup_large-v2_transcription.txt") == "team-sync--2026.03.04--standup"
    assert T.file_key_from_name("a.b.c.d.e_small_transcription.txt") == "a.b.c.d.e"
    assert T.file_key_from_name("standup-2026-03-04_BC-2_large_transcription.txt") == "standup-2026-03-04_BC-2"
    # a dot in the trailing token is a filename component, not a model id
    assert T.file_key_from_name("clip.v2_transcription.json") == "clip.v2"
    assert T.file_key_from_name("IMG_4821_transcription.json") == "IMG_4821"


# ---------------------------------------------------------------------------
# 6. _entities / _entites both strip
# ---------------------------------------------------------------------------
def test_entities_suffix_variants_strip(T):
    for marker in ("entities", "entites"):
        name = f"2025_0202_171732_medium_transcription_{marker}.csv"
        assert T.file_key_from_name(name) == "2025_0202_171732"
        assert T.file_key_from_name(f"IMG_4821_transcription_{marker}.csv") == "IMG_4821"
        # case-insensitive, as the writers/PAT_ENTITIES are
        assert T.file_key_from_name(f"standup_large_transcription_{marker.upper()}.csv") == "standup"


# ---------------------------------------------------------------------------
# 7. The reducer never returns an empty stem
# ---------------------------------------------------------------------------
def test_reducer_never_returns_empty(T):
    names = [
        "IMG_4821_transcription.json", "medium_transcription.json", "tiny_transcription.txt",
        "_transcription.json", "_medium_transcription.json", "transcription.json",
        "_speakers.rttm", "_metadata.csv", ".json", "_entities.csv",
        "2025_0202_171732_transcription_entities.csv", "distil-large-v3_transcription.txt",
        "large_transcription.json", "_large-v3_transcription.txt",
    ]
    for name in names:
        got = T.file_key_from_name(name)
        assert got, f"{name!r} reduced to an empty stem"
        assert got.strip() == got or got == got.strip(), name
    # canonicalize_key is unchanged: an empty *sanitised* base is the only thing that
    # reaches its stable_id fallback, and a reduced stem always sanitises to something
    assert T.canonicalize_key(T.file_key_from_name("_medium_transcription.json"), "/audio/_medium_transcription.json") != ""


# ---------------------------------------------------------------------------
# 8. _speakers / _metadata stripping still works
# ---------------------------------------------------------------------------
def test_speakers_and_metadata_stripping_still_works(T):
    assert T.file_key_from_name("x_speakers.rttm") == "x"
    assert T.file_key_from_name("x_metadata.csv") == "x"
    assert T.file_key_from_name("2025_0202_171732_speakers.rttm") == "2025_0202_171732"
    assert T.file_key_from_name("2025_0202_171732_metadata.csv") == "2025_0202_171732"
    assert T.file_key_from_name("2025_0202_171732_BC-1_speakers.rttm") == "2025_0202_171732_BC-1"
    assert T.file_key_from_name("IMG_4821_metadata.csv") == "IMG_4821"


# ---------------------------------------------------------------------------
# 9. Every model id this repo writes is stripped whole
# ---------------------------------------------------------------------------
def test_every_repo_model_id_strips(T):
    # TRANSCRIPTION_MODEL_IDS is derived from the producers, not guessed; see its
    # comment. Assert each id, plus the ids the repo's own docs name.
    assert T.TRANSCRIPTION_MODEL_IDS == (
        "large-v3", "large-v2", "large", "turbo",
        "medium.en", "medium",
        "small.en", "small",
        "base.en", "base",
        "tiny.en", "tiny",
    )
    for model_id in T.TRANSCRIPTION_MODEL_IDS:
        for stem in ("2025_0202_171732", "IMG_4821", "standup-2026-03-04"):
            for ext in ("txt", "json"):
                name = f"{stem}_{model_id}_transcription.{ext}"
                assert T.file_key_from_name(name) == stem, name
    # the two real filenames spelled out in the repo (audio_copy.sh:32,
    # auto_ingest/content/build_summaries.py:65-67)
    assert T.file_key_from_name("20250823050742_large-v3_transcription.txt") == "20250823050742"
    assert T.file_key_from_name("2025_0202_171732_large-v3_transcription.txt") == "2025_0202_171732"
    # a model id this build does not know yet still strips when it is recognisably one
    assert T.file_key_from_name("IMG_4821_distil-large-v3_transcription.txt") == "IMG_4821"
    assert T.file_key_from_name("IMG_4821_large-v4_transcription.txt") == "IMG_4821"
    assert T.file_key_from_name("Systran_faster_whisper_large-v3_transcription.txt") == "Systran_faster_whisper"
    # ... and model_rank() still ranks the tag it recovers from a filename
    assert T.model_rank("large-v3") < T.model_rank("medium")


# ---------------------------------------------------------------------------
# 10. The collapse hazard is reported, never silently merged
# ---------------------------------------------------------------------------
def test_media_identity_stem_folds_camera_variants_only(T):
    # renditions / camera variants of ONE clip are not a collision
    assert T.media_identity_stem("/a/2025_0202_171732_F.mp4") == T.media_identity_stem("/a/2025_0202_171732_R.mp4")
    assert T.media_identity_stem("/a/2025_0202_171732.m4a") == T.media_identity_stem("/a/2025_0202_171732.mp3")
    assert T.media_identity_stem("/a/2025_0202_171732_BC-2.mp4") == "2025_0202_171732"
    # different recordings keep different identities
    assert T.media_identity_stem("/a/IMG_4821.m4a") != T.media_identity_stem("/a/IMG_9912.m4a")
    assert T.media_identity_stem("/a/kickoff_notes.m4a") != T.media_identity_stem("/a/kickoff_brief.m4a")


def test_key_collision_is_reported_bounded_and_non_fatal(T, tmp_path, monkeypatch, caplog):
    """Two distinct recordings CAN still land on one key (same start second), so the
    ingest reports it instead of merging silently. Bounded: one warning, N keys."""
    root = _scan_roots(T, tmp_path, monkeypatch)
    # same YYYY_MMDD_HHMMSS second -> canonicalize_key rebuilds the same key for both
    for name in ("standup_recording_2025_0202_171732.m4a", "interview_recording_2025_0202_171732.m4a",
                 "2025_0202_171732_F.mp4", "2025_0202_171732_R.mp4"):
        (root / name).write_text("{}")

    with caplog.at_level(logging.WARNING, logger="ingest_transcripts"):
        mapping = T.discover_keys()

    assert set(mapping) == {"2025_0202_221732"}                 # one key, unchanged shape
    assert len(mapping["2025_0202_221732"]["media_all"]) == 4   # nothing dropped
    warnings = [r for r in caplog.records if "key-collision" in r.getMessage()]
    assert len(warnings) == 1                            # one line, not one per file
    msg = warnings[0].getMessage()
    assert "1 key(s) hold media from more than one recording (3 recording(s) on them)" in msg
    # stems sorted for a stable line; the dashcam clip's F/R camera variants folded into
    # ONE identity (they are two views of one recording, deliberately one key)
    assert ("2025_0202_221732: 2025_0202_171732 | interview_recording_2025_0202_171732"
            " | standup_recording_2025_0202_171732") in msg
    assert "_F" not in msg and "_R" not in msg


def test_collision_report_is_bounded(T, tmp_path, monkeypatch, caplog):
    root = _scan_roots(T, tmp_path, monkeypatch)
    n = T.MAX_KEY_COLLISION_SAMPLES + 5
    for i in range(n):                       # one colliding key per distinct second
        for stem in ("standup_recording", "interview_recording"):
            (root / f"{stem}_2025_0202_1717{i:02d}.m4a").write_text("{}")

    with caplog.at_level(logging.WARNING, logger="ingest_transcripts"):
        mapping = T.discover_keys()

    assert len(mapping) == n                              # every path kept, keyed by second
    warnings = [r for r in caplog.records if "key-collision" in r.getMessage()]
    assert len(warnings) == 1
    msg = warnings[0].getMessage()
    assert f"{n} key(s) hold media from more than one recording (2 recording(s) on each)" not in msg
    assert f"{n} key(s) hold media from more than one recording ({2 * n} recording(s) on them)" in msg
    assert msg.count("; ") == T.MAX_KEY_COLLISION_SAMPLES - 1   # samples capped
    assert msg.endswith(" ...")                                  # and the rest only counted


def test_no_collision_warning_when_keys_are_distinct(T, tmp_path, monkeypatch, caplog):
    root = _scan_roots(T, tmp_path, monkeypatch)
    for name in ("IMG_4821.m4a", "IMG_9912.m4a", "standup-2026-03-04.m4a"):
        (root / name).write_text("{}")

    with caplog.at_level(logging.WARNING, logger="ingest_transcripts"):
        mapping = T.discover_keys()

    assert len(mapping) == 3
    assert not [r for r in caplog.records if "key-collision" in r.getMessage()]


# ---------------------------------------------------------------------------
# Model-tag extraction and ranking (select_best_json)
# ---------------------------------------------------------------------------

def test_the_model_tag_is_the_last_token_not_the_whole_prefix(T):
    """The tag must not swallow the key prefix.

    `_([A-Za-z0-9\\-\\._]+)` spans `_`, so for
    `2025_0202_171732_large-v3_transcription.txt` it captured
    `0202_171732_large-v3`. MODEL_PREF.index() then missed, and large-v3,
    large-v2 and large ALL fell through to the substring fallback at rank 100.
    """
    assert T.extract_model_tag_from_json_txt(
        "2025_0202_171732_large-v3_transcription.txt") == "large-v3"
    assert T.extract_model_tag_from_json_txt(
        "2025_0202_171732_large-v2_transcription.txt") == "large-v2"
    assert T.extract_model_tag_from_json_txt(
        "meeting_standup_medium_transcription.txt") == "medium"


def test_model_preference_order_is_actually_honoured(T):
    """The bug was a three-way tie: large-v3 did not beat large."""
    def rank(suffix):
        return T.model_rank(T.extract_model_tag_from_json_txt(
            f"2025_0202_171732_{suffix}_transcription.txt"))

    assert rank("large-v3") == 0
    assert rank("large-v3") < rank("large-v2") < rank("large") < rank("medium")
    assert rank("medium") < rank("small") < rank("tiny")
    assert rank("large-v3") < rank("medium.en")


def test_a_tag_with_a_still_present_prefix_still_ranks_by_model(T):
    """Older on-disk names must not tie. Longest matching suffix wins."""
    assert T.model_rank("0202_171732_large-v3") < T.model_rank("0202_171732_large")
    assert T.model_rank("x_medium") < T.model_rank("x_small")
    # No model in the name at all is still the worst case.
    assert T.model_rank("4821") > T.model_rank("large-v3")


def test_select_best_json_prefers_large_v3(T, tmp_path):
    """End to end: given all three, the strongest model must win.

    `select_best_json` sorts on (in_audio_base, rank, -segments, -mtime, path),
    so an equal rank would be broken by segment count or mtime rather than by
    model - which is how the wrong transcript could be chosen.
    """
    written = {}
    for suffix in ("large", "large-v2", "large-v3"):
        path = tmp_path / f"2025_0202_171732_{suffix}_transcription.json"
        path.write_text('{"segments": [{"start": 0.0, "end": 1.0}]}', encoding="utf-8")
        written[suffix] = str(path)
    with mock.patch.object(T, "is_in_audio_base", return_value=True):
        best = T.select_best_json(sorted(written.values()), [])
    assert best == written["large-v3"], "the strongest model must win"


def test_no_model_tag_is_still_selectable(T, tmp_path):
    """An untagged sidecar must not become unselectable."""
    a = tmp_path / "plain_transcription.json"
    b = tmp_path / "2025_0202_171732_medium_transcription.json"
    for p in (a, b):
        p.write_text('{"segments": [{"start": 0.0, "end": 1.0}]}', encoding="utf-8")
    with mock.patch.object(T, "is_in_audio_base", return_value=True):
        best = T.select_best_json([str(a), str(b)], [])
    assert best == str(b), "a tagged transcript outranks an untagged one"
