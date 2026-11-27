"""Hermetic contract tests for the pure helpers in ``auto_ingest.shorts.thumbnail``.

Round 31: no test in the repository exercised ``_frame_contrast``, ``_best_frame``
or ``_wrap``; ``make_thumbnail`` was only ever monkeypatched from
``tests/test_abtest.py``. This module pins the render-side frame-selection and
hook-text wrapping contract with in-memory stubs only: no moviepy, no font or
fixture-file IO, no network, DB, service, model or live pipeline.
"""
from __future__ import annotations

import numpy as np
import pytest

from auto_ingest.shorts import thumbnail as thumb


def _uniform(value: int = 0) -> np.ndarray:
    return np.full((6, 6, 3), value, dtype=np.uint8)


def _checker(lo: int = 0, hi: int = 255) -> np.ndarray:
    arr = np.full((6, 6, 3), lo, dtype=np.uint8)
    arr[::2, ::2] = hi
    return arr


class _StubVideo:
    """Minimal moviepy-shaped stub; records every requested sample time."""

    def __init__(self, duration, fps=30, frames=None, fail_positive=False):
        self.duration = duration
        self.fps = fps
        self.requests: list[float] = []
        self._frames = frames or {}
        self._fail_positive = fail_positive

    def get_frame(self, t):
        self.requests.append(t)
        if self._fail_positive and t != 0.0:
            raise RuntimeError("decode failure")
        return self._frames.get(t, _uniform(0))


class _FakeDraw:
    """Font-free textlength: one width unit per character."""

    def textlength(self, text, font=None):
        return float(len(text))


def test_frame_contrast_of_a_uniform_frame_is_zero():
    assert thumb._frame_contrast(_uniform(128)) == 0.0


def test_frame_contrast_ignores_the_alpha_channel():
    rgb = _checker()
    rgba = np.dstack([rgb, np.full((6, 6), 7, dtype=np.uint8)])
    assert thumb._frame_contrast(rgba) == pytest.approx(thumb._frame_contrast(rgb))


def test_frame_contrast_grows_with_spatial_variance():
    assert thumb._frame_contrast(_checker()) > thumb._frame_contrast(_uniform(64))


def test_best_frame_selects_the_highest_contrast_sample():
    # dur=10, n=4 -> raw sample times 1.25, 3.75, 6.25, 8.75; all below dur-0.01.
    marked = {6.25: _checker()}
    vid = _StubVideo(10.0, fps=30, frames=marked)
    frame_index, arr = thumb._best_frame(vid, n_samples=4)
    assert vid.requests == pytest.approx([1.25, 3.75, 6.25, 8.75])
    assert np.array_equal(arr, marked[6.25])
    assert frame_index == int(6.25 * 30)


def test_best_frame_clamps_sample_times_before_duration_end():
    # dur=0.02 -> raw times 0.0025/0.0075/0.0125/0.0175, clamped at dur-0.01=0.01.
    vid = _StubVideo(0.02)
    thumb._best_frame(vid, n_samples=4)
    assert vid.requests[0] == pytest.approx(0.0025)
    assert vid.requests[-1] == pytest.approx(0.01)
    assert all(t <= 0.01 + 1e-12 for t in vid.requests)


def test_best_frame_falls_back_to_frame_zero_when_every_sample_fails():
    first = _checker()
    vid = _StubVideo(2.0, fps=24, frames={0.0: first}, fail_positive=True)
    frame_index, arr = thumb._best_frame(vid, n_samples=3)
    assert np.array_equal(arr, first)
    assert frame_index == 0


def test_wrap_joins_words_and_respects_max_width():
    assert thumb._wrap(_FakeDraw(), "a b c d", None, 5) == ["a b c", "d"]


def test_wrap_never_drops_an_oversized_first_word():
    lines = thumb._wrap(_FakeDraw(), "abcdefgh ij", None, 3)
    assert lines[0] == "abcdefgh"
    assert "ij" in lines


def test_wrap_caps_hook_text_at_three_lines():
    lines = thumb._wrap(_FakeDraw(), "w1 w2 w3 w4 w5 w6", None, 2)
    assert lines == ["w1", "w2", "w3"]


def test_wrap_of_empty_text_returns_no_lines():
    assert thumb._wrap(_FakeDraw(), "", None, 10) == []
