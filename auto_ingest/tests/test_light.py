"""Lightweight, ML-free tests for auto_ingest shared helpers (W-49/W-48/W-42).

These intentionally avoid importing the heavy subpackages (ingest/diarize/
dashcam/content) which require torch/transformers/ultralytics. They cover the
dependency-free surface: the durable outbox and the local event-envelope shim
(mirroring the shared contract, correlation_id required).
"""

from __future__ import annotations

import os
import tempfile
import uuid

from auto_ingest.outbox import GraphOutbox, get_outbox
from auto_ingest.events import emit, EventEnvelope, SCHEMA_VERSION, SOURCE_REPO


def test_outbox_roundtrip():
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    try:
        ob = GraphOutbox(path)
        cid = str(uuid.uuid4())
        op_id = ob.append("ingest.transcription.header", {"t_id": "T1"}, correlation_id=cid)
        pending = ob.pending()
        assert len(pending) == 1
        assert pending[0][0] == op_id
        assert pending[0][2] == cid
        ob.mark_done(op_id)
        assert ob.pending() == []
    finally:
        os.remove(path)


def test_outbox_failure_marks_retry():
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    try:
        ob = GraphOutbox(path)
        op_id = ob.append("x", {"a": 1})
        ob.mark_failed(op_id, "boom")
        pending = ob.pending()
        assert pending[0][4] == 1  # retry_count incremented
    finally:
        os.remove(path)


def test_event_envelope_requires_uuid_correlation_id():
    e = emit("ingest.evidence.linked", {"k": "v"})
    assert e.source_repo == SOURCE_REPO
    assert e.schema_version == SCHEMA_VERSION
    # correlation_id is auto-generated and a valid UUID
    uuid.UUID(e.correlation_id)
    assert e.to_dict()["event_type"] == "ingest.evidence.linked"


def test_event_envelope_rejects_non_uuid():
    try:
        EventEnvelope(
            schema_version=SCHEMA_VERSION,
            source_repo=SOURCE_REPO,
            event_type="x",
            correlation_id="not-a-uuid",
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError for non-UUID correlation_id")
