"""Hermetic contract tests for ``auto_ingest.shorts.captions._cue_word_timings`` (r24).

Why this is a new test area
---------------------------
``_cue_word_timings`` is the single helper that turns a caption cue's text and
time window into the ``(word, word_start, word_end)`` triples the karaoke
renderer consumes (``compose.compose_scripted_short``). It is a pure function:
no I/O, graph, network, credential, model or service is involved.

It was previously unpinned:

* ``tests/test_shorts.py`` (the only caption/shorts suite) never imports or calls
  it; a bounded ``git grep`` for ``_cue_word_timings``/``word_timings`` over the
  test tree returned no rows.
* No ``opencode-reviews`` report mentions cue word timings or this helper.
* ``git log --all --oneline -- auto_ingest/shorts/captions.py`` shows only the
  historical gap-analysis commit ``45c8e882``; no ``agent/*`` or sweep branch
  from rounds 1-23 modified the module (byte-identical between the canonical
  refs ``c4b3bac`` and ``f24b264``).

These tests pin the helper's documented, deterministic contract: one timing per
word, monotone starts, a minimum per-word visibility (``MIN_VIS = 0.18``), a
clamped span, explicit-timing precedence with incomplete entries dropped,
fallback to an even split, and byte-identical output for repeated inputs.
All assertions are in-memory; nothing is written anywhere.
"""
from __future__ import annotations

import pytest

from auto_ingest.shorts.captions import _cue_word_timings

MIN_VIS = 0.18


def _visibility(timing):
    return timing[2] - timing[1]


def test_empty_and_whitespace_text_yields_no_timings():
    for text in ("", "   ", "\t\n  "):
        assert _cue_word_timings(text, 0.0, 3.0, None) == []


def test_even_split_one_entry_per_word_in_text_order():
    out = _cue_word_timings("alpha beta gamma", 0.0, 3.0, None)
    assert [word for word, _, _ in out] == ["alpha", "beta", "gamma"]
    assert len(out) == 3
    assert out[0][1] == pytest.approx(0.0)
    assert out[-1][2] == pytest.approx(3.0)
    for earlier, later in zip(out, out[1:]):
        assert later[1] >= earlier[1]
        assert later[1] == pytest.approx(earlier[2])
    for timing in out:
        assert _visibility(timing) >= MIN_VIS - 1e-9


def test_visibility_never_drops_below_minimum_even_on_a_short_span():
    out = _cue_word_timings("a b c d", 0.0, 0.2, None)
    assert [word for word, _, _ in out] == ["a", "b", "c", "d"]
    for timing in out:
        assert _visibility(timing) >= MIN_VIS - 1e-9
    # Starts advance by one quarter of the cue span (0.2 / 4 = 0.05)...
    assert [s for _, s, _ in out][1] == pytest.approx(0.05)
    # ...and MIN_VIS can extend the last word past the nominal cue end.
    assert out[-1][2] > 0.2


def test_non_positive_span_clamps_then_honours_minimum_visibility():
    zero_span = _cue_word_timings("a", 5.0, 5.0, None)
    negative_span = _cue_word_timings("a", 5.0, 4.0, None)
    # The 0.1 span clamp is then extended by MIN_VIS (0.18) for a single word.
    assert zero_span == [("a", 5.0, pytest.approx(5.18))]
    assert negative_span == zero_span


def test_explicit_complete_timings_take_precedence_verbatim():
    explicit = [
        {"word": "a", "start": 0, "end": 0.2},
        {"word": " b ", "start": "0.3", "end": "0.4"},
        {"word": "c", "start": 1, "end": 2},
    ]
    out = _cue_word_timings("a b c", 0.0, 9.0, explicit)
    assert out == [("a", 0.0, 0.2), ("b", 0.3, 0.4), ("c", 1.0, 2.0)]


def test_explicit_entries_missing_start_or_end_are_dropped():
    partial = [{"word": "a", "start": 0, "end": 0.2}, {"word": "b"}]
    assert _cue_word_timings("a b", 0.0, 9.0, partial) == [("a", 0.0, 0.2)]
    # A blank word is still emitted when both endpoints are present.
    no_word = [{"start": 1, "end": 2}]
    assert _cue_word_timings("a b", 0.0, 9.0, no_word) == [("", 1.0, 2.0)]


def test_all_incomplete_explicit_timings_fall_back_to_even_split():
    explicit = [{"word": "a"}, {"word": "b", "start": 1}]
    assert _cue_word_timings("a b", 0.0, 2.0, explicit) == [
        ("a", 0.0, 1.0),
        ("b", 1.0, 2.0),
    ]


def test_identical_inputs_produce_byte_identical_timings():
    args = ("repeated words repeated", 1.25, 4.75, None)
    first = _cue_word_timings(*args)
    assert first == _cue_word_timings(*args)
    assert first == _cue_word_timings(*args)


def test_punctuation_and_non_ascii_words_are_preserved_verbatim():
    text = "Café—ok?  naïve 42%"
    out = _cue_word_timings(text, 0.0, 3.0, None)
    assert [word for word, _, _ in out] == text.split()
