"""embed_coverage.py must fail closed when a label cannot be audited.

The report is used to decide whether an embedding wave actually finished. A
failed per-label count used to be swallowed with ``except -> continue``: the
label disappeared from the table *and* from the overall totals, the script still
exited 0, and "overall: 100%" was indistinguishable from a complete audit.

Hermetic: the Neo4j driver and password resolver are replaced with in-memory
fakes. No database, network, credentials or filesystem state is involved.
"""
from __future__ import annotations

import logging
import sys

import pytest

import embed_coverage  # noqa: E402  (repo root is on sys.path via tests/conftest.py)


class _Result:
    """Minimal stand-in for a Neo4j result with one scalar record."""

    def __init__(self, value: int):
        self._value = value

    def single(self):
        return self

    def values(self):
        return [self._value]


class _Session:
    """Run every count query, but fail the ones naming ``fail_label``."""

    def __init__(self, fail_label: str | None, value: int = 5):
        self.fail_label = fail_label
        self.value = value

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, query, **_kwargs):
        if self.fail_label and f"(n:{self.fail_label})" in query:
            raise RuntimeError(f"simulated query failure for {self.fail_label}")
        return _Result(self.value)


class _Driver:
    def __init__(self, fail_label: str | None):
        self.fail_label = fail_label
        self.closed = False

    def session(self):
        return _Session(self.fail_label)

    def close(self):
        self.closed = True


@pytest.fixture
def fake_driver(monkeypatch):
    """Install a fake driver + password resolver; return (factory, drivers, run).

    ``embed_coverage.main()`` reads ``sys.argv`` (no argv parameter), so ``run``
    patches it for the duration of one invocation.
    """
    drivers: list[_Driver] = []

    def factory(fail_label: str | None):
        d = _Driver(fail_label)
        drivers.append(d)
        return d

    def run(*argv: str) -> int:
        monkeypatch.setattr(sys, "argv", ["embed_coverage.py", *argv])
        return embed_coverage.main()

    monkeypatch.setattr(embed_coverage.GraphDatabase, "driver",
                        lambda *_a, **_k: drivers[-1])
    monkeypatch.setattr(embed_coverage, "get_neo4j_password", lambda: "not-a-real-password")
    return factory, drivers, run


def test_healthy_run_exits_zero(fake_driver, caplog):
    factory, _, run = fake_driver
    factory(None)
    with caplog.at_level(logging.INFO):
        rc = run("--labels", "Segment", "Utterance")
    assert rc == 0
    assert "overall: 10/10 (100.0%)" in caplog.text
    assert "coverage incomplete" not in caplog.text


def test_failed_label_query_fails_closed(fake_driver, caplog):
    """A skipped label must change the exit status, not just log a warning."""
    factory, drivers, run = fake_driver
    factory("Utterance")
    with caplog.at_level(logging.INFO):
        rc = run("--labels", "Segment", "Utterance")

    # Segment was audited; Utterance was not, and the omission is visible.
    assert "Segment" in caplog.text
    assert "overall: 5/5 (100.0%)" in caplog.text
    assert "coverage incomplete" in caplog.text
    assert "Utterance" in caplog.text
    assert rc == 1, "a label that could not be audited must not yield exit 0"
    assert drivers[0].closed


def test_failed_label_query_is_named_in_the_report(fake_driver, caplog):
    """Every failed label must be listed, not just the first one."""
    factory, _, run = fake_driver
    factory("Utterance")
    with caplog.at_level(logging.INFO):
        rc = run("--labels", "Segment", "Utterance", "Summary")
    assert rc == 1
    summary = [r.message for r in caplog.records if "coverage incomplete" in r.message]
    assert len(summary) == 1
    assert "Utterance" in summary[0]
