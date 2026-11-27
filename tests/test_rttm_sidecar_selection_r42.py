"""Round-42: hermetic tests for RTTM *sidecar selection* (not speaker attribution).

Why this is a new test area
---------------------------
Prior rounds exercised two neighbouring but different concerns in
``auto_ingest.ingest.transcripts``:

* ``test_transcript_keys.py`` / ``test_ml_pure.py`` cover *media* key reduction
  (``file_key_from_name``, ``canonicalize_key``, ``media_identity_stem``) and
  model-tag ranking (``extract_model_tag_from_json_txt``, ``model_rank``,
  ``select_best_json``).
* The "RTTM speaker attribution" work covers the *content* of an RTTM once it is
  loaded (``utterances_from_rttm_*``, ``compute_segment_speaker_overlaps``,
  ``words_from_segments``, ``overlap``).

Neither covers the **index-selection step in between**: given an ingest key,
which of the discovered ``*_speakers.rttm`` sidecars does the pipeline attach?
That is ``_rttm_key_from_name`` (normalise a sidecar filename into an index key)
and ``pick_rttm_for_key`` (choose one path deterministically from that index).
Both are pure functions over strings and dicts. No prior review names either
function (``grep -rl`` over ``/home/scott/opencode-reviews/*.md`` -> 0 files),
and neither is referenced by any existing test module. This file locks in their
deterministic, side-effect-free contract with synthetic in-memory inputs only:
no Neo4j, no media, no ML inference, no filesystem outside pytest's ``tmp_path``.

The module is imported under a ``sys.modules`` stub (the established technique
in ``test_ml_pure.py`` / ``test_transcript_keys.py``) so the torch/transformers/
neo4j stack is never loaded.
"""

import importlib
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


# ---------------------------------------------------------------------------
# 1. _rttm_key_from_name: sidecar filename -> index key
# ---------------------------------------------------------------------------


def test_speakers_rttm_suffixes_strip_case_insensitively(T):
    """The writer emits ``*_speakers.rttm`` in mixed case across producers."""
    assert T._rttm_key_from_name("2025_0202_171732_speakers.rttm") == "2025_0202_171732"
    assert T._rttm_key_from_name("2025_0202_171732.RTTM") == "2025_0202_171732"
    assert T._rttm_key_from_name("x_SPEAKERS.RTTM") == "x"
    assert T._rttm_key_from_name("IMG_4821_speakers.Rttm") == "IMG_4821"


def test_speakers_suffix_strips_even_without_rttm_extension(T):
    """``_speakers`` is removed independently of the ``.rttm`` extension."""
    assert T._rttm_key_from_name("x_speakers") == "x"
    assert T._rttm_key_from_name("x_speakers_speakers.rttm") == "x_speakers"


def test_non_sidecar_and_non_rttm_names_are_untouched(T):
    """The reducer is deliberately narrow: it must not eat ordinary names."""
    assert T._rttm_key_from_name("plain.rttm") == "plain"
    assert T._rttm_key_from_name("no_sidecar.txt") == "no_sidecar.txt"

    # Dots and dashes are part of the key, not sidecar markers.
    assert T._rttm_key_from_name("a.b.c_speakers.rttm") == "a.b.c"
    assert T._rttm_key_from_name("team-sync--2026.03.04_speakers.rttm") == "team-sync--2026.03.04"


# ---------------------------------------------------------------------------
# 2. pick_rttm_for_key: deterministic sidecar selection
# ---------------------------------------------------------------------------


def test_exact_key_match_wins_over_substring_candidates(T):
    """An exact index hit is authoritative even if a substring key is closer."""
    idx = {"exact": ["/exact.rttm"], "exact_long": ["/near.rttm"]}
    assert T.pick_rttm_for_key("exact", idx) == "/exact.rttm"


def test_exact_key_prefers_shallowest_then_shortest_path(T):
    """Within one key: fewest path separators, then shortest path string."""
    idx = {
        "keyA": [
            "/deep/dir/z.rttm",  # 3 separators
            "/mid/dir/y.rttm",  # 2 separators
            "/shallow.rttm",  # 1 separator  <- winner
        ],
    }
    assert T.pick_rttm_for_key("keyA", idx) == "/shallow.rttm"

    idx_equal_depth = {"k": ["/aa.rttm", "/b.rttm"]}  # both 1 separator
    assert T.pick_rttm_for_key("k", idx_equal_depth) == "/b.rttm"


def test_substring_fallback_ranks_by_length_difference(T):
    """No exact hit: closest key length wins; ties break on the shorter key."""
    idx = {
        "2025_0202_171732_extra": ["/a/x.rttm"],  # diff 6
        "2025_0202_171732_extra_longer": ["/b/y.rttm"],  # diff 12
        "20": ["/short.rttm"],  # diff 15
        "other": ["/o.rttm"],  # not a substring
    }
    assert T.pick_rttm_for_key("2025_0202_171732", idx) == "/a/x.rttm"

    # Equal absolute length difference -> the shorter candidate key is chosen.
    idx_tie = {"abcdef": ["/long.rttm"], "abcd": ["/short.rttm"]}
    assert T.pick_rttm_for_key("abcde", idx_tie) == "/short.rttm"


def test_substring_match_is_bidirectional(T):
    """The fallback accepts an index key contained in the query key as well."""
    idx = {"xyz": ["/x.rttm"], "unrelated": ["/u.rttm"]}
    assert T.pick_rttm_for_key("abcdefxyz", idx) == "/x.rttm"


def test_no_candidate_and_empty_index_return_none(T):
    """Absence is explicit; selection never invents a path."""
    assert T.pick_rttm_for_key("missing", {"a": ["/a.rttm"], "b": ["/b.rttm"]}) is None
    assert T.pick_rttm_for_key("anything", {}) is None


def test_selection_is_pure_over_the_index(T):
    """Selection must not reorder, drop or otherwise mutate its input."""
    import copy

    idx = {
        "k": ["/zzzz.rttm", "/a.rttm", "/mid/b.rttm"],
        "k_extra": ["/e.rttm"],
    }
    before = copy.deepcopy(idx)
    # "/a.rttm" wins on (fewest separators, shortest path) -- not on input order.
    assert T.pick_rttm_for_key("k", idx) == "/a.rttm"
    assert T.pick_rttm_for_key("absent", idx) is None
    assert idx == before
