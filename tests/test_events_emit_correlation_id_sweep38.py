"""Hermetic regression for ``auto_ingest.events.emit`` under the canonical contract.

The canonical AssistX ``EventEnvelope`` requires ``correlation_id`` to be a UUID
string and ``links`` to be a list. ``emit`` promises to auto-generate the
correlation id, so it must supply both fields even when the caller omits them.
When AssistX is not importable the local mirror happens to do that coercion,
which hides the defect in standalone runs; the canonical (production) path
rejects the ``None`` values.

No live AssistX, pydantic, Neo4j, network, credentials or filesystem state is
used: a tiny in-memory stand-in reproduces the canonical field requirements.
"""
from __future__ import annotations

import importlib
import sys
import types
import uuid

import pytest

import auto_ingest.events as events

_FAKE_NAMES = (
    "assistx",
    "assistx.contracts",
    "assistx.contracts.event_envelope",
    "assistx.contracts.version",
)


class _StrictEventEnvelope:
    """In-memory mirror of the canonical EventEnvelope field requirements."""

    def __init__(self, *, schema_version, source_repo, event_type,
                 correlation_id, actor=None, payload=None, links=None):
        if not isinstance(correlation_id, str):
            raise TypeError("correlation_id must be a valid UUID string")
        uuid.UUID(correlation_id)
        if not isinstance(links, list):
            raise TypeError("links must be a list")
        self.schema_version = schema_version
        self.source_repo = source_repo
        self.event_type = event_type
        self.correlation_id = correlation_id
        self.actor = actor if actor is not None else {}
        self.payload = payload if payload is not None else {}
        self.links = links


def _install_canonical_stub() -> dict:
    pkg = types.ModuleType("assistx")
    pkg.__path__ = []
    contracts = types.ModuleType("assistx.contracts")
    contracts.__path__ = []
    envelope_mod = types.ModuleType("assistx.contracts.event_envelope")
    version_mod = types.ModuleType("assistx.contracts.version")

    envelope_mod.EventEnvelope = _StrictEventEnvelope
    envelope_mod.Actor = type("Actor", (), {})
    envelope_mod.AuthState = type("AuthState", (), {})
    envelope_mod.EventLink = type("EventLink", (), {})
    envelope_mod.TraceEvent = type("TraceEvent", (), {})
    envelope_mod.TraceGroup = type("TraceGroup", (), {})
    version_mod.SCHEMA_VERSION = "2026-06-08.v1"

    contracts.event_envelope = envelope_mod
    contracts.version = version_mod
    pkg.contracts = contracts

    return {
        "assistx": pkg,
        "assistx.contracts": contracts,
        "assistx.contracts.event_envelope": envelope_mod,
        "assistx.contracts.version": version_mod,
    }


@pytest.fixture
def canonical_events():
    """Reload ``auto_ingest.events`` against a canonical-contract stand-in."""
    saved = {name: sys.modules.get(name) for name in _FAKE_NAMES}
    sys.modules.update(_install_canonical_stub())
    importlib.reload(events)
    try:
        assert events._USING_CANONICAL is True
        yield events
    finally:
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod
        importlib.reload(events)


def test_emit_generates_correlation_id_without_caller(canonical_events):
    env = canonical_events.emit("ingest.evidence.linked", {"key": "k1"})
    assert env.event_type == "ingest.evidence.linked"
    assert isinstance(env.correlation_id, str)
    uuid.UUID(env.correlation_id)
    assert env.links == []


def test_emit_allows_explicit_id_without_links(canonical_events):
    cid = str(uuid.uuid4())
    env = canonical_events.emit("ingest.evidence.linked", {}, correlation_id=cid)
    assert env.correlation_id == cid
    assert env.links == []


def test_emit_keeps_explicit_links(canonical_events):
    cid = str(uuid.uuid4())
    links = [{"rel": "FOR_KEY", "target_type": "MediaFile", "target_id": "k1"}]
    env = canonical_events.emit(
        "ingest.evidence.linked", {}, correlation_id=cid, links=links)
    assert env.links == links


def test_fallback_mirror_still_auto_generates(monkeypatch):
    """Control: the standalone fallback path is unchanged by the fix."""
    for name in _FAKE_NAMES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    importlib.reload(events)
    try:
        env = events.emit("ingest.evidence.linked", {})
        uuid.UUID(env.correlation_id)
    finally:
        importlib.reload(events)
