"""Hermetic contract tests for the ffprobe rate parser in the YOLO sidecar worker (r34).

The worker derives its source FPS from ffprobe's ``avg_frame_rate`` rational before
it decodes any frames. That parse is pure string/JSON handling: no ffmpeg, GPU,
model, media file or network is involved, so it can be pinned entirely in memory
by replacing the worker's shell shim with a synthetic ffprobe response.

The invariant under test: an unusable rate must never abort a batch with a raw
ZeroDivisionError/ValueError; it degrades to a finite, non-negative 0.0 FPS.
"""
from __future__ import annotations

import json

import pytest

from auto_ingest.dashcam import yolo_batch_worker as ybw


class _ProbeResult:
    """Minimal stand-in for the CompletedProcess the worker's sh() returns."""

    def __init__(self, stdout: str):
        self.stdout = stdout


def _probe(monkeypatch, *, streams=None, raw=None):
    """Call probe_fps_w_h against a synthetic ffprobe payload, capturing the cmd."""
    payload = {} if streams is None else {"streams": streams}
    captured = {}

    def fake_sh(cmd):
        captured["cmd"] = cmd
        return _ProbeResult(raw if raw is not None else json.dumps(payload))

    monkeypatch.setattr(ybw, "sh", fake_sh)
    fps, width, height = ybw.probe_fps_w_h("/synthetic/clip.MP4")
    return fps, width, height, captured["cmd"]


def test_rational_rate_is_parsed_and_dimensions_are_ints(monkeypatch):
    fps, width, height, cmd = _probe(
        monkeypatch,
        streams=[{"avg_frame_rate": "30000/1001", "width": 1920, "height": 1080}],
    )
    assert fps == pytest.approx(30000 / 1001, rel=1e-12)
    assert (width, height) == (1920, 1080)
    assert "ffprobe" in cmd and "-show_streams" in cmd


def test_integer_rate_keeps_historical_value(monkeypatch):
    fps, width, height, _ = _probe(
        monkeypatch,
        streams=[{"avg_frame_rate": "30/1", "width": "1280", "height": "720"}],
    )
    assert fps == 30.0
    assert (width, height) == (1280, 720)


def test_missing_streams_fall_back_to_legacy_default(monkeypatch):
    fps, width, height, _ = _probe(monkeypatch, raw="{}")
    assert (fps, width, height) == (30.0, 0, 0)


def test_zero_numerator_is_finite_zero(monkeypatch):
    fps, _, _, _ = _probe(
        monkeypatch, streams=[{"avg_frame_rate": "0/1", "width": 640, "height": 480}]
    )
    assert fps == 0.0


def test_zero_denominator_fails_closed_instead_of_crashing(monkeypatch):
    fps, width, height, _ = _probe(
        monkeypatch, streams=[{"avg_frame_rate": "0/0", "width": 640, "height": 480}]
    )
    assert fps == 0.0
    assert (width, height) == (640, 480)


def test_non_numeric_rate_fails_closed_instead_of_crashing(monkeypatch):
    fps, _, _, _ = _probe(
        monkeypatch, streams=[{"avg_frame_rate": "variable", "width": 640, "height": 480}]
    )
    assert fps == 0.0


def test_blank_rate_fails_closed_instead_of_crashing(monkeypatch):
    fps, _, _, _ = _probe(
        monkeypatch, streams=[{"avg_frame_rate": "", "width": 640, "height": 480}]
    )
    assert fps == 0.0


def test_non_finite_rate_fails_closed(monkeypatch):
    fps, _, _, _ = _probe(
        monkeypatch, streams=[{"avg_frame_rate": "nan/1", "width": 640, "height": 480}]
    )
    assert fps == 0.0
