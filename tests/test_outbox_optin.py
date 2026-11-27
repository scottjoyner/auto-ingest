"""Hermetic tests for the outbox opt-in contract (``get_outbox``).

The outbox must be strictly opt-in: a disable spelling such as ``0``/``false``
must not create a SQLite file in the caller's working directory, and an explicit
sqlite path must be enough to opt in (as the ``get_outbox`` docstring promises).
"""

from __future__ import annotations

import pytest

from auto_ingest import outbox


@pytest.fixture(autouse=True)
def _isolated_outbox(monkeypatch):
    """No process-level singleton or ambient env leaks between these cases."""
    monkeypatch.setattr(outbox, "_OUTBOX", None)
    monkeypatch.delenv("AUTO_INGEST_OUTBOX", raising=False)
    monkeypatch.delenv("AUTO_INGEST_OUTBOX_PATH", raising=False)
    yield
    # Close any outbox the case created so no SQLite handle is left open.
    current = outbox._OUTBOX
    if current is not None:
        current.close()
    monkeypatch.setattr(outbox, "_OUTBOX", None)


@pytest.mark.parametrize("raw", ["0", " 0 ", "false", "FALSE", "no", "off", "disabled"])
def test_disable_spellings_do_not_create_an_outbox(monkeypatch, tmp_path, raw):
    db = tmp_path / f"disabled-{raw.strip() or 'empty'}.sqlite"
    monkeypatch.setenv("AUTO_INGEST_OUTBOX", raw)
    monkeypatch.setenv("AUTO_INGEST_OUTBOX_PATH", str(db))

    assert outbox.get_outbox() is None
    assert not db.exists()


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "on", "enabled"])
def test_enable_spellings_create_the_configured_outbox(monkeypatch, tmp_path, raw):
    db = tmp_path / f"enabled-{raw.lower()}.sqlite"
    monkeypatch.setenv("AUTO_INGEST_OUTBOX", raw)
    monkeypatch.setenv("AUTO_INGEST_OUTBOX_PATH", str(db))

    ob = outbox.get_outbox()
    assert ob is not None
    assert ob.path == str(db)
    assert db.exists()


def test_explicit_path_opts_in_without_env(tmp_path):
    db = tmp_path / "explicit.sqlite"

    ob = outbox.get_outbox(str(db))

    assert ob is not None
    assert ob.path == str(db)
    assert db.exists()


def test_explicit_path_wins_over_env_path(monkeypatch, tmp_path):
    env_db = tmp_path / "from-env.sqlite"
    explicit_db = tmp_path / "from-arg.sqlite"
    monkeypatch.setenv("AUTO_INGEST_OUTBOX", "1")
    monkeypatch.setenv("AUTO_INGEST_OUTBOX_PATH", str(env_db))

    ob = outbox.get_outbox(str(explicit_db))

    assert ob is not None
    assert ob.path == str(explicit_db)
    assert explicit_db.exists()
    assert not env_db.exists()


def test_singleton_is_reused_once_enabled(monkeypatch, tmp_path):
    db = tmp_path / "singleton.sqlite"
    monkeypatch.setenv("AUTO_INGEST_OUTBOX", "1")
    monkeypatch.setenv("AUTO_INGEST_OUTBOX_PATH", str(db))

    first = outbox.get_outbox()
    second = outbox.get_outbox()
    assert first is second
