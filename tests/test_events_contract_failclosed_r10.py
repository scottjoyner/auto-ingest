"""Hermetic fail-closed invariants of ``auto_ingest.events`` (r10).

The event shim had no test module at all (a repo-wide grep for
``auto_ingest.events`` / ``emit(`` / ``correlation_id must be`` in ``tests/``
returned zero hits). It is the seam where ingest emits cross-repo fleet events,
so the UUID-correlation invariant and the envelope shape deserve drift
sentinels.

The module prefers the canonical ``assistx.contracts.EventEnvelope`` and falls
back to a local mirror when that package is absent. To make these tests
independent of whether a sibling ``auto-assist`` checkout happens to be on
``sys.path``, we import a *fresh copy* of events.py with the canonical import
actively blocked. Nothing here touches a DB, a network, a service, or the
real process-wide module cache beyond a temporary, restored entry.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import sys
import uuid
from datetime import datetime
from pathlib import Path

import pytest


class _BlockCanonicalEnvelope(importlib.abc.MetaPathFinder):
    """Force the ``except ImportError`` fallback branch of events.py."""

    def find_spec(self, fullname, path=None, target=None):  # noqa: ANN001
        if fullname == "assistx.contracts.event_envelope":
            raise ImportError("blocked for hermetic fallback test (r10)")
        return None


def _load_forced_fallback():
    finder = _BlockCanonicalEnvelope()
    canonical_name = "assistx.contracts.event_envelope"
    saved = sys.modules.pop(canonical_name, None)
    sys.meta_path.insert(0, finder)
    try:
        source = Path(__file__).resolve().parents[1] / "auto_ingest" / "events.py"
        spec = importlib.util.spec_from_file_location("auto_ingest_events_fallback_r10", source)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        sys.meta_path.remove(finder)
        if saved is not None:
            sys.modules[canonical_name] = saved
    return module


events = _load_forced_fallback()


# ---------------------------------------------------------------------------
# Module identity / contract constants
# ---------------------------------------------------------------------------


def test_fallback_branch_is_actually_exercised():
    assert events._USING_CANONICAL is False


def test_contract_identity_constants_are_pinned():
    assert events.SOURCE_REPO == "auto-ingest"
    assert events.SCHEMA_VERSION == "2026-06-08.v1"


def test_all_declared_exports_exist():
    for name in events.__all__:
        assert hasattr(events, name), f"missing export: {name}"


# ---------------------------------------------------------------------------
# correlation_id fail-closed validation
# ---------------------------------------------------------------------------


def test_emit_generates_a_uuid_when_correlation_id_is_omitted():
    envelope = events.emit("ingest.evidence.linked", {"key": "k"})
    assert isinstance(envelope.correlation_id, str)
    assert uuid.UUID(envelope.correlation_id).version == 4


def test_emit_preserves_a_supplied_uuid_verbatim():
    cid = "12345678-1234-5678-1234-567812345678"
    envelope = events.emit("ingest.evidence.linked", {}, correlation_id=cid)
    assert envelope.correlation_id == cid


@pytest.mark.parametrize(
    "bad",
    ["not-a-uuid", "", "1234", "x" * 64, 12345, [], {"id": 1}],
    ids=["words", "empty", "short", "hexwords", "int", "list", "dict"],
)
def test_emit_rejects_non_uuid_correlation_id(bad):
    with pytest.raises(ValueError, match="correlation_id must be a valid UUID"):
        events.emit("ingest.evidence.linked", {}, correlation_id=bad)


@pytest.mark.parametrize(
    "alternate",
    [
        "12345678-1234-5678-1234-567812345678",
        "{12345678-1234-5678-1234-567812345678}",
        "urn:uuid:12345678-1234-5678-1234-567812345678",
        "12345678123456781234567812345678",
    ],
    ids=["canonical", "braced", "urn", "bare_hex"],
)
def test_uuid_equivalent_representations_are_accepted_verbatim(alternate):
    """The mirror uses ``uuid.UUID`` (like the canonical validator), so all
    representations it accepts are stored unchanged - not normalised."""
    envelope = events.emit("ingest.evidence.linked", {}, correlation_id=alternate)
    assert envelope.correlation_id == alternate


def test_local_mirror_envelope_enforces_uuid_directly():
    with pytest.raises(ValueError, match="correlation_id must be a valid UUID"):
        events.EventEnvelope(
            schema_version=events.SCHEMA_VERSION,
            source_repo=events.SOURCE_REPO,
            event_type="ingest.evidence.linked",
            correlation_id="garbage",
        )


# ---------------------------------------------------------------------------
# Envelope shape / isolation
# ---------------------------------------------------------------------------


def test_emit_sets_schema_version_and_source_repo():
    envelope = events.emit("ingest.evidence.linked", {"k": "v"})
    assert envelope.schema_version == events.SCHEMA_VERSION
    assert envelope.source_repo == events.SOURCE_REPO
    assert envelope.event_type == "ingest.evidence.linked"


def test_emit_preserves_payload_and_links():
    payload = {"t_id": "t-1", "key": "dashcam/2026-07-14/CLT_1423_F"}
    links = [{"rel": "FOR_TASK", "target_type": "Task", "target_id": "t-1"}]
    envelope = events.emit("ingest.evidence.linked", payload, links=links)
    assert envelope.payload == payload
    assert envelope.links == links


def test_default_payload_links_and_actor_are_isolated_per_envelope():
    first = events.emit("ingest.evidence.linked", {})
    second = events.emit("ingest.evidence.linked", {})
    first.payload["leak"] = True
    first.links.append({"rel": "x"})
    first.to_dict()["actor"]["leak"] = True
    assert second.payload == {}
    assert second.links == []
    assert second.to_dict()["actor"] == {}


def test_to_dict_shape_is_exact():
    envelope = events.emit("ingest.evidence.linked", {"k": "v"})
    assert set(envelope.to_dict()) == {
        "schema_version",
        "source_repo",
        "event_type",
        "correlation_id",
        "actor",
        "ts",
        "payload",
        "links",
    }


def test_to_dict_timestamp_is_utc_iso8601():
    envelope = events.emit("ingest.evidence.linked", {})
    ts = datetime.fromisoformat(envelope.to_dict()["ts"])
    assert ts.tzinfo is not None
    assert ts.utcoffset().total_seconds() == 0
