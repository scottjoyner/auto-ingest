"""Hermetic regression tests for short-duration curiosity-reveal cues (r21).

`hook_bank.curiosity_reveal` builds the question/reveal cue pair used by the
shorts planner.  For pathological short runtimes the original arithmetic could
emit an inverted (end < start), negative or out-of-runtime cue, which downstream
caption/render code treats as a valid timed line.  These tests pin the timing
contract: both cues must satisfy ``0 <= start <= end <= duration`` for *every*
runtime, while the normal (>=6s) output stays byte-for-byte unchanged.

Pure over the model: no Neo4j, no moviepy, no network, no DB/WAL, no fixtures.
"""
from __future__ import annotations

import pytest

from auto_ingest.shorts.hook_bank import curiosity_reveal

# Short runtimes where the unclamped arithmetic breaks the timing contract.
_SHORT_DURATIONS = [5.0, 3.0, 1.0, 0.4, 0.0]


@pytest.mark.parametrize("duration", _SHORT_DURATIONS)
def test_short_runtime_never_emits_inverted_or_out_of_range_cues(duration):
    cues = curiosity_reveal("why does it work?", "the math is simple", duration)
    assert [c.kind for c in cues] == ["question", "reveal"]
    for cue in cues:
        assert cue.start >= 0.0, (duration, cue)
        assert cue.end >= cue.start, (duration, cue)
        assert cue.end <= duration, (duration, cue)


def test_explicit_start_short_runtime_never_inverts():
    cues = curiosity_reveal("q?", "a", duration=2.0, start=1.0)
    for cue in cues:
        assert 0.0 <= cue.start <= cue.end <= 2.0, cue


@pytest.mark.parametrize(
    "duration,expected",
    [
        (
            12.0,
            [("question", 5.62, 7.82), ("reveal", 7.82, 10.62)],
        ),
        (
            30.0,
            [("question", 13.72, 15.92), ("reveal", 15.92, 18.72)],
        ),
        (
            6.0,
            [("question", 2.92, 5.12), ("reveal", 5.12, 5.6)],
        ),
    ],
)
def test_normal_runtime_output_is_unchanged(duration, expected):
    cues = curiosity_reveal("why does it work?", "the math is simple", duration)
    actual = [(c.kind, c.start, c.end) for c in cues]
    assert actual == expected
