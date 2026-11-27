"""Explicit caption word timings from an export must be best-effort.

`_cue_word_timings` is the pure helper that turns a cue's text plus optional
per-word timings into karaoke windows. Its contract is an even split when no
usable timings are supplied. A single malformed row in a platform export must
not abort the whole render pass with a conversion error.
"""

from auto_ingest.shorts.captions import _cue_word_timings


def test_malformed_explicit_timing_falls_back_to_even_split():
    got = _cue_word_timings(
        "one two three", 0.0, 3.0,
        [{"word": "one", "start": "n/a", "end": 1.0}],
    )
    assert got == [("one", 0.0, 1.0), ("two", 1.0, 2.0), ("three", 2.0, 3.0)]


def test_valid_explicit_timings_are_still_honoured():
    got = _cue_word_timings(
        "one two three", 0.0, 3.0,
        [{"word": "one", "start": 0.5, "end": 1.5}],
    )
    assert got == [("one", 0.5, 1.5)]
