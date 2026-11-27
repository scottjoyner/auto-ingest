"""Hermetic regression contract for PlannedShort.duration() (r47).

A persisted Plan is JSON and may contain null, string, bool or non-finite
timing values. duration() feeds the virality scorer and planner iteration cap;
a single bad cue/shot timing must not abort the whole scoring/planning pass.
All cases are pure in-memory data (or a tiny tmp_path JSON plan) — no DB,
network, media, clock or service.
"""

import math

import pytest

from auto_ingest.shorts.models import Cue, PlannedShort, Shot


def _short(*, cues=(), shots=()):
    return PlannedShort(id="s1", brief_topic="t", title="T", cues=list(cues), shots=list(shots))


# --- valid inputs keep their exact previous behaviour -----------------------


def test_valid_cues_duration_is_max_end():
    s = _short(cues=[Cue(0.0, 3.0, "a"), Cue(3.0, 7.5, "b")])
    assert s.duration() == 7.5


def test_valid_shots_duration_is_sum_when_no_cues():
    s = _short(shots=[Shot("k", "/x", 0.0, 2.5), Shot("k2", "/y", 0.0, 4.0)])
    assert s.duration() == 6.5


def test_empty_short_duration_is_zero_float():
    d = _short().duration()
    assert d == 0.0 and isinstance(d, float)


@pytest.mark.parametrize("end,expected", [(3, 3.0), (0, 0.0), (2.25, 2.25)])
def test_integer_and_float_cue_ends_yield_floats(end, expected):
    d = _short(cues=[Cue(0.0, end, "a")]).duration()
    assert d == expected and isinstance(d, float)


# --- malformed cue ends -----------------------------------------------------


def test_none_cue_end_is_ignored_not_raised():
    s = _short(cues=[Cue(0.0, None, "a"), Cue(0.0, 5.0, "b")])
    assert s.duration() == 5.0


def test_single_none_cue_end_yields_zero_float_not_none():
    d = _short(cues=[Cue(0.0, None, "a")]).duration()
    assert d == 0.0 and isinstance(d, float)


def test_string_cue_end_is_ignored():
    d = _short(cues=[Cue(0.0, "7.5", "a")]).duration()
    assert d == 0.0 and isinstance(d, float)


def test_nan_cue_end_is_ignored():
    assert _short(cues=[Cue(0.0, float("nan"), "a"), Cue(0.0, 5.0, "b")]).duration() == 5.0


def test_inf_cue_end_is_ignored():
    assert _short(cues=[Cue(0.0, float("inf"), "a"), Cue(0.0, 5.0, "b")]).duration() == 5.0


def test_bool_cue_end_is_ignored():
    assert _short(cues=[Cue(0.0, True, "a"), Cue(0.0, 5.0, "b")]).duration() == 5.0


# --- malformed shot durations ----------------------------------------------


def test_none_shot_duration_is_ignored():
    assert _short(shots=[Shot("k", "/x", 0.0, None), Shot("k2", "/y", 0.0, 4.0)]).duration() == 4.0


def test_string_shot_duration_is_ignored():
    d = _short(shots=[Shot("k", "/x", 0.0, "3")]).duration()
    assert d == 0.0 and isinstance(d, float)


def test_non_finite_shot_durations_are_ignored():
    s = _short(
        shots=[
            Shot("a", "/a", 0.0, float("nan")),
            Shot("b", "/b", 0.0, 2.0),
            Shot("c", "/c", 0.0, float("-inf")),
        ]
    )
    assert s.duration() == 2.0


@pytest.mark.parametrize(
    "cues,shots",
    [
        ([Cue(0.0, None, "a")], []),
        ([Cue(0.0, "x", "a")], []),
        ([], [Shot("k", "/x", 0.0, None)]),
        ([Cue(0.0, float("nan"), "a")], [Shot("k", "/x", 0.0, "bad")]),
    ],
)
def test_duration_always_returns_a_finite_float(cues, shots):
    d = _short(cues=cues, shots=shots).duration()
    assert isinstance(d, float) and math.isfinite(d)
