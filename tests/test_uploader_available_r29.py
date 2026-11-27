"""Round 29: the live-upload credential gate must fail closed.

``BaseAdapter.available()`` is the last credential check before
``process_queue(dry_run=False)`` uploads a real short.  Python's ``all([])`` is
``True``, so an adapter that declares no credential variables would report
itself as configured with zero secrets.  Pin the fail-closed contract here.
"""
from __future__ import annotations

from auto_ingest.shorts.uploader import BaseAdapter


class _NoCredAdapter(BaseAdapter):
    name = "noconfig"
    needs = []


class _OneCredAdapter(BaseAdapter):
    name = "onecred"
    needs = ["R29_TEST_TOKEN"]


def test_empty_needs_is_not_available() -> None:
    """all([]) is True; an adapter with no declared secrets must not pass."""
    assert _NoCredAdapter().available() is False


def test_whitespace_only_credential_is_not_available(monkeypatch) -> None:
    """A blank/whitespace env value is not a credential."""
    monkeypatch.setenv("R29_TEST_TOKEN", "   ")
    assert _OneCredAdapter().available() is False


def test_missing_credential_is_not_available(monkeypatch) -> None:
    monkeypatch.delenv("R29_TEST_TOKEN", raising=False)
    assert _OneCredAdapter().available() is False


def test_real_credential_is_available(monkeypatch) -> None:
    monkeypatch.setenv("R29_TEST_TOKEN", "real-token")
    assert _OneCredAdapter().available() is True
