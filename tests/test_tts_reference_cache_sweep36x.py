"""Hermetic proof: a cached voice reference must be a non-empty *file*.

`extract_voice_reference()` promises a ~12 s clean WAV and caches it under
`<audio_root>/.cache/tts/owner_voice_reference.wav`.  On the pristine base it
accepts the cache on `exists()` alone, so a zero-byte file (or a directory with
that name) is handed to XTTS as `speaker_wav` and to ffmpeg as a build result.
Every input here is a tiny local fixture; no Neo4j, ffmpeg, TTS model, mount,
network or credential is used.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from auto_ingest.shorts import tts

REF_NAME = "owner_voice_reference.wav"


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    d = tmp_path / "tts-cache"
    d.mkdir()
    monkeypatch.setattr(tts, "_cache_dir", lambda: d)
    return d


@pytest.fixture
def tiny_capture(tmp_path, monkeypatch):
    src = tmp_path / "capture.wav"
    src.write_bytes(b"fake audio bytes")
    monkeypatch.setattr(tts, "owner_audio_captures", lambda: [src])
    monkeypatch.setattr(tts, "owner_local_audio", lambda: [])
    monkeypatch.setattr(tts, "_audio_seconds", lambda _p: 13.0)
    return src


@pytest.fixture
def fake_build(monkeypatch):
    """Stand in for the ffmpeg concat call; writes a few bytes, never executes."""

    def _run(cmd, **kwargs):
        out = Path(cmd[-1])
        if not out.is_dir():
            out.write_bytes(b"RIFFfakewav")
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr(tts.subprocess, "run", _run)
    return _run


# --- defect cases: fail on pristine base -------------------------------------


def test_zero_byte_cached_reference_is_rebuilt_not_reused(cache_dir, tiny_capture, fake_build):
    cache = cache_dir / REF_NAME
    cache.write_bytes(b"")
    got = tts.extract_voice_reference()
    assert got == cache, "the cache path is reused only after rebuilding it"
    assert cache.read_bytes() == b"RIFFfakewav", "an invalid cache must trigger a fresh build"


def test_directory_named_reference_is_not_returned(cache_dir, tiny_capture, fake_build):
    cache = cache_dir / REF_NAME
    cache.mkdir()
    got = tts.extract_voice_reference()
    assert got != cache, "a directory is not a WAV reference"


def test_zero_byte_build_result_is_not_returned(cache_dir, tiny_capture, monkeypatch):
    def _run(cmd, **kwargs):
        Path(cmd[-1]).touch()  # ffmpeg 'succeeded' but produced no data
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr(tts.subprocess, "run", _run)
    got = tts.extract_voice_reference()
    assert got is None, "an empty build result must fail closed"


# --- controls: pass on both sides ---------------------------------------------


def test_nonempty_cached_reference_is_reused(cache_dir, monkeypatch):
    cache = cache_dir / REF_NAME
    cache.write_bytes(b"RIFFvalidwav")

    def _boom(*args, **kwargs):
        raise AssertionError("valid cache must not be rebuilt")

    monkeypatch.setattr(tts.subprocess, "run", _boom)
    assert tts.extract_voice_reference() == cache


def test_force_rebuilds_over_a_valid_cache(cache_dir, tiny_capture, fake_build):
    cache = cache_dir / REF_NAME
    cache.write_bytes(b"stale data")
    got = tts.extract_voice_reference(force=True)
    assert got == cache
    assert cache.read_bytes() == b"RIFFfakewav"


def test_missing_cache_is_built(cache_dir, tiny_capture, fake_build):
    cache = cache_dir / REF_NAME
    got = tts.extract_voice_reference()
    assert got == cache
    assert cache.stat().st_size > 0
