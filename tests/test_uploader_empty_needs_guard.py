"""Vacuous-credential-gate regression for shorts upload adapters (sweep26x).

Hermetic: only the light ``auto_ingest.shorts.uploader`` module and in-process
env edits. No network, DB, mounts, credentials, services or model load.
"""
from __future__ import annotations

from auto_ingest.shorts.uploader import BaseAdapter, YouTubeShortsAdapter


class _NoNeedsAdapter(BaseAdapter):
    """An adapter that forgot to declare its credential requirements."""

    name = "no_needs"

    def upload(self, item, *, dry_run=True):
        return None


def _clear_credentials(monkeypatch):
    for key in (
        "YT_CLIENT_SECRET_JSON", "YT_TOKEN_JSON", "TIKTOK_ACCESS_TOKEN",
        "IG_USER_ID", "IG_ACCESS_TOKEN",
    ):
        monkeypatch.delenv(key, raising=False)


def test_empty_needs_does_not_attest_credentials(monkeypatch):
    """all([]) is True; an empty requirement list must not pass the gate."""
    _clear_credentials(monkeypatch)
    assert BaseAdapter().needs == []
    assert BaseAdapter().available() is False


def test_inherited_empty_needs_does_not_attest_credentials(monkeypatch):
    """A subclass that omits `needs` inherits [] and must be unavailable."""
    _clear_credentials(monkeypatch)
    assert _NoNeedsAdapter().needs == []
    assert _NoNeedsAdapter().available() is False


def test_declared_but_missing_credentials_is_unavailable(monkeypatch):
    """Control: a real adapter with no credentials stays blocked."""
    _clear_credentials(monkeypatch)
    assert YouTubeShortsAdapter().available() is False


def test_declared_credentials_present_is_available(monkeypatch):
    """Control: genuine credentials still pass the gate."""
    _clear_credentials(monkeypatch)
    monkeypatch.setenv("YT_CLIENT_SECRET_JSON", "/tmp/never-read-client-secret.json")
    monkeypatch.setenv("YT_TOKEN_JSON", "/tmp/never-read-token.json")
    assert YouTubeShortsAdapter().available() is True
