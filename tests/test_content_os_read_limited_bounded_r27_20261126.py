"""Hermetic regression tests for bounded prefix reads in ``content_os`` (r27).

``read_limited`` exists to keep local source ingestion bounded, but it used to
call ``Path.read_bytes()``, which buffers the whole file before slicing off the
preview prefix. These tests use only ``tmp_path`` files and in-process fakes:
no network, credentials, services, model loads, clock or live state.
"""

from __future__ import annotations

from pathlib import Path

from content_os.adapters import MAX_TEXT_BYTES, read_limited


class _SpyFile:
    """Wrap a real binary file and record every ``read`` size requested."""

    def __init__(self, real):
        self._real = real
        self.requests: list[int] = []

    def read(self, size: int = -1) -> bytes:
        self.requests.append(size)
        return self._real.read(size)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._real.close()
        return False


class _SpyPath:
    """Path-like with a real ``open`` but a forbidden unbounded ``read_bytes``."""

    def __init__(self, real: Path):
        self._real = real
        self.spy: _SpyFile | None = None

    def open(self, *args, **kwargs):
        self.spy = _SpyFile(self._real.open(*args, **kwargs))
        return self.spy

    def read_bytes(self) -> bytes:
        raise AssertionError("unbounded Path.read_bytes() called by read_limited")


def test_read_limited_does_not_call_unbounded_read_bytes(tmp_path: Path) -> None:
    """The helper must read a bounded prefix, not buffer the whole file."""
    p = tmp_path / "source.txt"
    p.write_bytes(b"hello world")
    guarded = _SpyPath(p)
    assert read_limited(guarded, limit=32) == ("hello world", False)


def test_read_limited_requests_only_limit_plus_one_byte(tmp_path: Path) -> None:
    """Exactly ``limit + 1`` bytes are requested, so truncation is detectable."""
    p = tmp_path / "source.txt"
    p.write_bytes(b"0123456789" * 20)
    guarded = _SpyPath(p)
    text, truncated = read_limited(guarded, limit=10)
    assert text == "0123456789"
    assert truncated is True
    assert guarded.spy is not None
    assert guarded.spy.requests == [11]


def test_read_limited_small_file_is_exact(tmp_path: Path) -> None:
    p = tmp_path / "source.txt"
    payload = b"A short local note.\n"
    p.write_bytes(payload)
    assert read_limited(p, limit=MAX_TEXT_BYTES) == (payload.decode(), False)


def test_read_limited_truncates_at_explicit_limit(tmp_path: Path) -> None:
    p = tmp_path / "source.txt"
    p.write_bytes(b"x" * (MAX_TEXT_BYTES + 5))
    text, truncated = read_limited(p, limit=MAX_TEXT_BYTES)
    assert truncated is True
    assert len(text) == MAX_TEXT_BYTES
    assert text == "x" * MAX_TEXT_BYTES


def test_read_limited_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "source.txt"
    p.write_bytes(b"")
    assert read_limited(p, limit=MAX_TEXT_BYTES) == ("", False)
