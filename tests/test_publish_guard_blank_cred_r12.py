"""Hermetic live-publish guard tests: blank credentials are not credentials.

Round 12 / agent/auto-ingest-r12-20261127. Completely disjoint from the
custody, ingest, outbox, receipt, events-contract, audit-surface and
date-index themes: this pins the ``shorts.publish_guard`` safety fence only.

No real credential is read or printed. Every env value below is a synthetic
string set through pytest's ``monkeypatch`` inside this test process; there is
no network, DB, docker, service or live-upload operation.
"""
from __future__ import annotations

import pytest

from auto_ingest.shorts import publish_guard as pg

_ALL_ENV = [pg.LIVE_ENV] + [
    var for vars_ in pg.PLATFORM_CREDS.values() for var in vars_
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Start each test with a guard-relevant environment that is entirely absent."""
    for var in _ALL_ENV:
        monkeypatch.delenv(var, raising=False)


def test_whitespace_only_credential_is_not_configured(monkeypatch):
    monkeypatch.setenv("TIKTOK_ACCESS_TOKEN", "   \t  ")
    assert pg.configured_platforms() == []


def test_blank_credential_keeps_safe_to_run_false(monkeypatch):
    monkeypatch.setenv(pg.LIVE_ENV, "1")
    for var in pg.PLATFORM_CREDS["instagram"]:
        monkeypatch.setenv(var, "   ")
    assert pg.safe_to_run() is False


def test_whitespace_only_credential_still_blocks_live_mode(monkeypatch):
    monkeypatch.setenv(pg.LIVE_ENV, "1")
    monkeypatch.setenv("TIKTOK_ACCESS_TOKEN", " \t ")
    with pytest.raises(pg.LivePublishForbidden):
        pg.require_live_mode()


def test_empty_credential_is_not_configured(monkeypatch):
    """Control: a truly empty var was already correctly ignored."""
    monkeypatch.setenv("TIKTOK_ACCESS_TOKEN", "")
    assert pg.configured_platforms() == []


def test_real_looking_credential_still_counts(monkeypatch):
    """Control: a non-blank (synthetic) credential keeps the existing behaviour."""
    monkeypatch.setenv(pg.LIVE_ENV, "1")
    monkeypatch.setenv("YT_TOKEN_JSON", "/nonexistent/fake-token.json")
    assert pg.configured_platforms() == ["youtube_shorts"]
    assert pg.safe_to_run() is True
    pg.require_live_mode()  # must not raise


def test_no_opt_in_blocks_even_with_a_valid_credential(monkeypatch):
    """Control: opt-in is still mandatory regardless of credentials."""
    monkeypatch.setenv("IG_ACCESS_TOKEN", "synthetic-token")
    monkeypatch.setenv("IG_USER_ID", "synthetic-user")
    assert pg.configured_platforms() == ["instagram"]
    assert pg.safe_to_run() is False
    with pytest.raises(pg.LivePublishForbidden):
        pg.require_live_mode()
