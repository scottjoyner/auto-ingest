"""Round-38 hermetic contract: RTTM speaker-attribution pure helpers.

Target: ``auto_ingest/ingest/transcripts.py`` (priority repo ``auto-ingest``).

This pins the deterministic, in-memory diarization surface that turns whisper
word-level segments plus an RTTM speaker track into utterances and
segment->speaker edges.  No report from rounds 1-37 and no existing test
mentioned any of these helpers; the only neighbouring coverage is
``auto_ingest/tests/test_ml_pure.py`` which stops at ``overlap`` and
``load_rttm`` and never touches the functions below.

Every helper here is pure: plain dicts/tuples in, plain dicts/lists out.  The
module is imported with the heavy ML stack stubbed (the established technique
in ``test_ml_pure.py``), but no inference, graph, service, credential, network,
live DB/WAL/cron, backup or fixture-file I/O occurs.  ``stable_id`` is md5 of
``part + '|'`` over each part; expected ids are computed locally rather than
imported, so a change to the public id shape fails this contract.
"""

from __future__ import annotations

import hashlib
import importlib
import sys
import unittest.mock as mock
from copy import deepcopy

import pytest


def _sid(*parts: str) -> str:
    """Independent restatement of discovery.stable_id's 4-part id shape."""
    h = hashlib.md5()
    for p in parts:
        h.update((p or "").encode("utf-8", errors="ignore"))
        h.update(b"|")
    return h.hexdigest()


@pytest.fixture(scope="module")
def T():
    """Import transcripts with its ML-only imports stubbed, as test_ml_pure does."""
    stubs = {
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
    }
    with mock.patch.dict(sys.modules, stubs):
        return importlib.import_module("auto_ingest.ingest.transcripts")


# ---------------------------------------------------------------------------
# words_from_segments
# ---------------------------------------------------------------------------

def test_words_from_segments_flattens_strips_and_sorts(T):
    segments = [
        {"id": "s2", "start": 2.0, "end": 3.0, "words": [
            {"word": "second", "start": 2.2, "end": 2.8},
            {"word": "   ", "start": 2.9, "end": 2.95},  # whitespace-only dropped
        ]},
        {"id": "s1", "start": 0.0, "end": 1.0, "words": [
            {"word": " world ", "start": 0.6, "end": 0.9},
            {"word": "hello", "start": 0.1, "end": 0.5},
        ]},
    ]
    original = deepcopy(segments)
    out = T.words_from_segments(segments)
    assert out == [
        {"text": "hello", "start": 0.1, "end": 0.5},
        {"text": "world", "start": 0.6, "end": 0.9},
        {"text": "second", "start": 2.2, "end": 2.8},
    ]
    assert segments == original  # pure: caller's data untouched


def test_words_from_segments_falls_back_to_segment_timing(T):
    seg = {"id": "s", "start": 5.0, "end": 6.0, "words": [{"word": "x"}]}
    assert T.words_from_segments([seg]) == [{"text": "x", "start": 5.0, "end": 6.0}]

    # 0.0 is falsy, so an explicit zero word time also falls back to the
    # segment clock -- pin the documented `or` semantics, not an assumption.
    seg0 = {"id": "s", "start": 5.0, "end": 6.0,
            "words": [{"word": "y", "start": 0.0, "end": 0.0}]}
    assert T.words_from_segments([seg0]) == [{"text": "y", "start": 5.0, "end": 6.0}]


def test_words_from_segments_empty_inputs(T):
    assert T.words_from_segments([]) == []
    assert T.words_from_segments([{"id": "s", "start": 0, "end": 1}]) == []
    assert T.words_from_segments([{"id": "s", "words": None}]) == []


# ---------------------------------------------------------------------------
# utterances_from_rttm_with_words
# ---------------------------------------------------------------------------

def _word_segments():
    return [
        {"id": "s1", "start": 0.0, "end": 1.0, "text": "hello world", "words": [
            {"word": "hello", "start": 0.1, "end": 0.5},
            {"word": "world", "start": 0.6, "end": 0.9},
        ]},
        {"id": "s2", "start": 2.0, "end": 3.0, "text": "second", "words": [
            {"word": "second", "start": 2.2, "end": 2.8},
        ]},
    ]


def test_utterances_with_words_groups_by_midpoint_and_preserves_index(T):
    rttm = [(0.0, 2.0, "spkA"), (2.0, 4.0, "spkB")]
    out = T.utterances_from_rttm_with_words(rttm, _word_segments())
    assert out == [
        {"id": _sid("utt", "spkA", "0.100", "0.900"),
         "speaker_label": "spkA", "start": 0.1, "end": 0.9,
         "text": "hello world", "segment_id": None, "idx": 0},
        {"id": _sid("utt", "spkB", "2.200", "2.800"),
         "speaker_label": "spkB", "start": 2.2, "end": 2.8,
         "text": "second", "segment_id": None, "idx": 1},
    ]


def test_utterances_with_words_skips_empty_and_out_of_range_tracks(T):
    rttm = [(10.0, 11.0, "spkC"), (0.0, 1.0, "spkA")]
    out = T.utterances_from_rttm_with_words(rttm, _word_segments())
    # The empty track is skipped entirely; idx is the ORIGINAL rttm index (1).
    assert [u["idx"] for u in out] == [1]
    assert out[0]["speaker_label"] == "spkA"
    assert out[0]["text"] == "hello world"
    assert T.utterances_from_rttm_with_words(rttm, []) == []


def test_utterances_with_words_midpoint_boundaries_are_inclusive(T):
    seg = [{"id": "s", "start": 0.0, "end": 2.0,
            "words": [{"word": "edge", "start": 1.0, "end": 1.0}]}]  # mid == 1.0
    assert T.utterances_from_rttm_with_words([(1.0, 1.0, "spk"), (0.0, 1.0, "spk")], seg) == [
        {"id": _sid("utt", "spk", "1.000", "1.000"),
         "speaker_label": "spk", "start": 1.0, "end": 1.0,
         "text": "edge", "segment_id": None, "idx": 0},
        {"id": _sid("utt", "spk", "1.000", "1.000"),
         "speaker_label": "spk", "start": 1.0, "end": 1.0,
         "text": "edge", "segment_id": None, "idx": 1},
    ]


def test_utterances_with_words_ids_are_distinct_per_speaker_and_shape(T):
    rttm = [(0.0, 1.0, "spkA"), (0.0, 1.0, "spkB")]
    seg = [{"id": "s", "start": 0.0, "end": 1.0,
            "words": [{"word": "hi", "start": 0.2, "end": 0.4}]}]
    a, b = T.utterances_from_rttm_with_words(rttm, seg)
    assert a["id"] != b["id"]
    assert len(a["id"]) == 32  # md5 hex
    assert a["start"] == b["start"] == 0.2 and a["end"] == b["end"] == 0.4


# ---------------------------------------------------------------------------
# utterances_from_rttm_dominant_segment
# ---------------------------------------------------------------------------

def test_dominant_segment_assigns_max_overlap_and_unknown_fallback(T):
    rttm = [(0.0, 2.0, "spkA"), (2.0, 4.0, "spkB")]
    segs = [
        {"id": "s1", "start": 0.0, "end": 1.0, "text": "one"},
        {"id": "s2", "start": 2.0, "end": 3.0, "text": "two"},
        {"id": "s3", "start": 10.0, "end": 11.0, "text": "none"},
    ]
    out = T.utterances_from_rttm_dominant_segment(rttm, segs)
    assert [u["speaker_label"] for u in out] == ["spkA", "spkB", "UNKNOWN"]
    assert [u["text"] for u in out] == ["one", "two", "none"]
    assert [u["segment_id"] for u in out] == ["s1", "s2", "s3"]
    assert [u["idx"] for u in out] == [0, 1, 2]
    assert out[0]["id"] == _sid("utt", "spkA", "s1", "0.000", "1.000")
    assert out[2]["id"] == _sid("utt", "UNKNOWN", "s3", "10.000", "11.000")


def test_dominant_segment_tie_keeps_first_track_and_degenerate_is_unknown(T):
    rttm = [(0.0, 2.0, "spkA"), (2.0, 4.0, "spkB")]
    segs = [
        {"id": "tie", "start": 1.5, "end": 2.5, "text": "x"},   # 0.5 / 0.5 -> first wins
        {"id": "rev", "start": 0.0, "end": 0.0, "text": "y"},   # zero-length -> UNKNOWN
        {"id": "bad", "start": 3.0, "end": 1.0, "text": "z"},   # inverted -> UNKNOWN
    ]
    out = T.utterances_from_rttm_dominant_segment(rttm, segs)
    assert [u["speaker_label"] for u in out] == ["spkA", "UNKNOWN", "UNKNOWN"]
    assert out[0]["id"] == _sid("utt", "spkA", "tie", "1.500", "2.500")


def test_dominant_segment_empty_and_missing_id(T):
    assert T.utterances_from_rttm_dominant_segment([], [{"id": "s", "start": 0, "end": 1}])[0]["speaker_label"] == "UNKNOWN"
    out = T.utterances_from_rttm_dominant_segment([(0, 1, "spk")],
                                                   [{"start": 0, "end": 1, "text": "m"}])
    assert out[0]["id"] == _sid("utt", "spk", "None", "0.000", "1.000")


# ---------------------------------------------------------------------------
# _sum_overlap
# ---------------------------------------------------------------------------

def test_sum_overlap_accumulates_per_speaker_and_drops_zero(T):
    rttm = [(0.0, 2.0, "spkA"), (2.0, 4.0, "spkB"),
            (1.0, 1.5, "spkA"), (10.0, 11.0, "spkC")]
    assert T._sum_overlap(0.0, 3.0, rttm) == {"spkA": 2.5, "spkB": 1.0}
    assert T._sum_overlap(0.0, 3.0, []) == {}


# ---------------------------------------------------------------------------
# compute_segment_speaker_overlaps
# ---------------------------------------------------------------------------

_SMAP = {"spkA": {"id": "gidA"}, "spkB": {"id": "gidB"},
         "UNKNOWN": {"id": "gidU"}}


def test_segment_overlaps_best_and_edges(T):
    segs = [{"id": "s1", "start": 0.0, "end": 3.0},
            {"id": "s2", "start": 10.0, "end": 11.0}]
    rttm = [(0.0, 2.0, "spkA"), (2.0, 4.0, "spkB")]
    best, edges = T.compute_segment_speaker_overlaps(rttm, segs, _SMAP)
    assert best == {
        "s1": {"speaker_label": "spkA", "speaker_id": "gidA"},
        "s2": {"speaker_label": "UNKNOWN", "speaker_id": "gidU"},
    }
    assert edges == [
        {"segment_id": "s1", "speaker_id": "gidA", "label": "spkA",
         "overlap": 2.0, "proportion": pytest.approx(2.0 / 3.0)},
        {"segment_id": "s1", "speaker_id": "gidB", "label": "spkB",
         "overlap": 1.0, "proportion": pytest.approx(1.0 / 3.0)},
    ]


def test_segment_overlaps_min_proportion_is_inclusive(T):
    segs = [{"id": "s1", "start": 0.0, "end": 4.0}]
    rttm = [(0.0, 4.0, "spkA"), (0.0, 1.0, "spkB")]  # totals 4.0 / 1.0 -> props 0.8 / 0.2
    _, edges = T.compute_segment_speaker_overlaps(rttm, segs, _SMAP, min_proportion=0.2)
    assert [e["label"] for e in edges] == ["spkA", "spkB"]  # 0.2 >= 0.2 kept
    _, tight = T.compute_segment_speaker_overlaps(rttm, segs, _SMAP, min_proportion=0.21)
    assert [e["label"] for e in tight] == ["spkA"]


def test_segment_overlaps_unknown_id_fallback_when_label_unmapped(T):
    partial = {"UNKNOWN": {"id": "gidU"}}  # no spkA/spkB entries
    segs = [{"id": "s1", "start": 0.0, "end": 3.0}]
    rttm = [(0.0, 2.0, "spkA"), (2.0, 4.0, "spkB")]
    best, edges = T.compute_segment_speaker_overlaps(rttm, segs, partial)
    assert best["s1"] == {"speaker_label": "spkA", "speaker_id": "gidU"}
    assert {e["speaker_id"] for e in edges} == {"gidU"}
