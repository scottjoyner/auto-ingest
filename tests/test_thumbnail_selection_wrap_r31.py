"""Hermetic contract tests for the render-only thumbnail helpers (r31).

Subject: ``auto_ingest.shorts.thumbnail`` -- the pure helpers that decide which
video frame becomes a cover and how the hook text is wrapped onto it.

Why this is a new surface: the only existing thumbnail tests monkeypatch
``make_thumbnail`` from ``abtest.py``; none exercises frame selection, contrast
scoring or hook wrapping.  No prior sweep report or agent branch mentions these
helpers.  Everything below is in-memory / fake objects: no MP4, moviepy,
network, graph, service, credential or live-state access.
"""
from __future__ import annotations

import numpy as np
import pytest

from auto_ingest.shorts import thumbnail


class FakeVideo:
    """Minimal stand-in for a VideoFileClip: records requested frame times."""

    def __init__(self, *, duration, frames, fps=None, failing_times=()):
        self.duration = duration
        self._frames = frames
        self.calls = []
        if fps is not None:
            self.fps = fps
        self._failing_times = set(failing_times)

    def get_frame(self, t):
        self.calls.append(t)
        if t in self._failing_times:
            raise RuntimeError("decode failure")
        return self._frames[t]


class FakeDraw:
    """Deterministic text measurement: one unit of width per character."""

    def __init__(self, unit=1):
        self.unit = unit

    def textlength(self, text, font=None):
        return len(text) * self.unit


def _rgb(contrast: float, *, size: int = 4) -> np.ndarray:
    """Return an RGB frame whose luma standard deviation is ``contrast``."""
    low, high = 10.0, 10.0 + contrast
    row = np.array([[low, low, low], [high, high, high]] * (size // 2))
    return np.stack([row] * size, axis=0).astype(np.float64)


# --------------------------------------------------------------------------
# _frame_contrast: variance-based "interestingness" proxy
# --------------------------------------------------------------------------


def test_frame_contrast_is_zero_for_flat_rgb():
    assert thumbnail._frame_contrast(np.full((4, 4, 3), 128.0)) == pytest.approx(0.0)


def test_frame_contrast_ignores_alpha_channel():
    rgb = np.full((4, 4, 4), 64.0)
    rgb[..., 3] = np.tile([0.0, 255.0], 2)[:4]  # alpha-only variation
    assert thumbnail._frame_contrast(rgb) == pytest.approx(0.0)


def test_frame_contrast_tracks_luma_variance():
    arr = _rgb(60.0)
    expected = float(arr[..., :3].mean(axis=2).std())
    assert thumbnail._frame_contrast(arr) == pytest.approx(expected)
    assert thumbnail._frame_contrast(arr) > thumbnail._frame_contrast(_rgb(5.0))


# --------------------------------------------------------------------------
# _best_frame: deterministic midpoint sampling and selection
# --------------------------------------------------------------------------


def test_best_frame_picks_highest_contrast_midpoint():
    times = [0.1, 0.3, 0.5, 0.7]
    frames = {t: _rgb(c) for t, c in zip(times, (1.0, 4.0, 9.0, 2.0))}
    vid = FakeVideo(duration=0.8, frames=frames, fps=25)

    idx, arr = thumbnail._best_frame(vid, n_samples=4)

    assert vid.calls == pytest.approx(times)
    assert arr is frames[0.5]
    assert idx == int(0.5 * 25)


def test_best_frame_ties_resolve_to_earliest_sample():
    times = [0.25, 0.75]
    frames = {t: _rgb(7.0) for t in times}
    vid = FakeVideo(duration=1.0, frames=frames)

    _, arr = thumbnail._best_frame(vid, n_samples=2)

    assert arr is frames[0.25]
    assert vid.calls == pytest.approx(times)


def test_best_frame_zero_duration_falls_back_to_one_second():
    frames = {t: _rgb(3.0) for t in (0.25, 0.75)}
    vid = FakeVideo(duration=0, frames=frames)

    _, arr = thumbnail._best_frame(vid, n_samples=2)

    assert vid.calls == pytest.approx([0.25, 0.75])
    assert arr is frames[0.25]


def test_best_frame_retries_are_not_needed_when_one_sample_fails():
    times = [0.25, 0.75]
    frames = {t: _rgb(c) for t, c in zip(times, (1.0, 8.0))}
    vid = FakeVideo(duration=1.0, frames=frames, failing_times={0.25})

    _, arr = thumbnail._best_frame(vid, n_samples=2)

    assert arr is frames[0.75]


def test_best_frame_falls_back_to_frame_zero_when_all_samples_fail():
    fallback = _rgb(6.0)
    vid = FakeVideo(duration=1.0, frames={0.0: fallback}, failing_times={0.25, 0.75})

    idx, arr = thumbnail._best_frame(vid, n_samples=2)

    assert arr is fallback
    assert idx == 0
    assert 0.0 in vid.calls


def test_best_frame_zero_fps_uses_thirty():
    frames = {0.25: _rgb(1.0), 0.75: _rgb(2.0)}
    vid = FakeVideo(duration=1.0, frames=frames, fps=0)

    idx, arr = thumbnail._best_frame(vid, n_samples=2)

    assert arr is frames[0.75]
    assert idx == int(0.75 * 30)


# --------------------------------------------------------------------------
# _wrap: greedy whitespace-normalized hook wrapping, capped at three lines
# --------------------------------------------------------------------------


def test_wrap_collapses_whitespace_and_packs_greedily():
    assert thumbnail._wrap(FakeDraw(), "  aa   bb  cc\tdd ee  ", None, 10) == [
        "aa bb cc",
        "dd ee",
    ]


def test_wrap_keeps_a_single_overlong_word():
    assert thumbnail._wrap(FakeDraw(), "supercalifragilistic short", None, 3) == [
        "supercalifragilistic",
        "short",
    ]


def test_wrap_truncates_after_three_lines():
    assert thumbnail._wrap(FakeDraw(), "aa bb cc dd ee ff", None, 3) == [
        "aa",
        "bb",
        "cc",
    ]


def test_wrap_empty_text_returns_no_lines():
    draw = FakeDraw()
    assert thumbnail._wrap(draw, "", None, 10) == []
    assert thumbnail._wrap(draw, "   \t ", None, 10) == []


def test_wrap_uses_the_supplied_draw_measurement():
    draw = FakeDraw(unit=20)
    assert thumbnail._wrap(draw, "aa bb", None, 100) == ["aa bb"]
    assert thumbnail._wrap(draw, "aa bb", None, 40) == ["aa", "bb"]


# --------------------------------------------------------------------------
# _load_font: render-side fallback when the bundled font is unavailable
# --------------------------------------------------------------------------


def test_load_font_falls_back_when_truetype_fails(monkeypatch):
    sentinel = object()

    def _boom(*args, **kwargs):
        raise OSError("font missing")

    monkeypatch.setattr(thumbnail.ImageFont, "truetype", _boom)
    monkeypatch.setattr(thumbnail.ImageFont, "load_default", lambda: sentinel)

    assert thumbnail._load_font(72) is sentinel
