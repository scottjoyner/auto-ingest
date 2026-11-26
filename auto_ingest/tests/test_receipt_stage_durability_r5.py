"""Focused Round-5 tests: receipt identity, stage order, publication durability.

All tests are hermetic: publication-durability cases use a *temporary* SQLite
outbox (never a live/production DB), and stage-order cases use a fake Neo4j
driver. No Neo4j, no service, no network, no ML imports.

These complement ``test_light.py`` (outbox round-trip / retry increment) and
``tests/test_ingest_intermediate.py`` (single-key ``import_all`` order) with
the identity, cross-key ordering and replay/durability edge cases that were
not previously asserted.
"""

from __future__ import annotations

from auto_ingest import ingest_import, ingest_write
from auto_ingest.outbox import GraphOutbox

# ---------------------------------------------------------------------------
# Minimal fake driver (no live Neo4j) — records Cypher calls
# ---------------------------------------------------------------------------


class _FakeResult:
    def data(self):
        return []

    def single(self):
        return None


class _FakeSession:
    def __init__(self, log):
        self._log = log

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, query, **params):
        self._log.append((query, dict(params)))
        return _FakeResult()


class _FakeDriver:
    def __init__(self):
        self.calls = []

    def session(self):
        return _FakeSession(self.calls)


# ---------------------------------------------------------------------------
# Receipt identity — correlation_id is the durable identity of an op
# ---------------------------------------------------------------------------


def test_correlation_ids_are_distinct_and_order_preserved(tmp_path):
    ob = GraphOutbox(str(tmp_path / "identity.sqlite"))
    try:
        first = ob.append("ingest.header", {"t_id": "T1"}, correlation_id="receipt-A")
        second = ob.append("ingest.segments", {"t_id": "T2"}, correlation_id="receipt-B")
        assert first < second
        pending = ob.pending()
        assert [row[2] for row in pending] == ["receipt-A", "receipt-B"]
        assert [row[1] for row in pending] == ["ingest.header", "ingest.segments"]
    finally:
        ob.close()


def test_correlation_id_survives_reopen(tmp_path):
    """The receipt identity is durable, not just an in-memory field."""
    path = str(tmp_path / "identity-reopen.sqlite")
    ob = GraphOutbox(path)
    op_id = ob.append("ingest.header", {"t_id": "T1"}, correlation_id="receipt-C")
    ob.close()

    reopened = GraphOutbox(path)
    try:
        pending = reopened.pending()
        assert len(pending) == 1
        assert pending[0][0] == op_id
        assert pending[0][2] == "receipt-C"
    finally:
        reopened.close()


def test_empty_correlation_id_is_allowed(tmp_path):
    ob = GraphOutbox(str(tmp_path / "identity-empty.sqlite"))
    try:
        ob.append("ingest.header", {"t_id": "T1"})
        assert ob.pending()[0][2] == ""
    finally:
        ob.close()


# ---------------------------------------------------------------------------
# Stage order — canonical parent-before-child import ordering
# ---------------------------------------------------------------------------


def test_sort_stages_uses_canonical_order():
    shuffled = ["summary", "segments", "transcription", "metadata"]
    assert ingest_import._sort_stages(shuffled) == [
        "transcription",
        "segments",
        "metadata",
        "summary",
    ]


def test_sort_stages_unknown_goes_last_and_is_stable():
    ordered = ingest_import._sort_stages(["zeta", "summary", "alpha", "segments", "omega"])
    # Known stages first in canonical order; unknown stages retain their
    # relative (input) order at the tail (Python's sorted is stable).
    assert ordered == ["segments", "summary", "zeta", "alpha", "omega"]


def test_sort_stages_is_idempotent():
    once = ingest_import._sort_stages(["utterances", "transcription", "entities", "speakers"])
    assert ingest_import._sort_stages(once) == once


def test_parent_stage_precedes_every_child():
    order = ingest_import.STAGE_ORDER
    parent = order.index("transcription")
    assert parent == 0
    for child in ingest_import._STAGES_WITH_PARENT:
        assert order.index(child) > parent
    assert set(ingest_import._STAGES_WITH_PARENT) == set(ingest_import._STAGE_EDGE)


def test_import_all_orders_across_multiple_keys(tmp_path):
    ingest_write.write_intermediate(str(tmp_path), "b", "summary", [{"id": "s1"}], fmt="json")
    ingest_write.write_intermediate(
        str(tmp_path), "a", "segments", [{"id": "g1", "parent": "t1"}], fmt="json"
    )
    ingest_write.write_intermediate(
        str(tmp_path), "a", "transcription", [{"id": "t1", "text": "hi"}], fmt="json"
    )

    drv = _FakeDriver()
    summary = ingest_import.import_all(str(tmp_path), drv, batch=500)
    # Grouped by key, then by canonical stage order within each key.
    assert list(summary.keys()) == [
        "a/transcription",
        "a/segments",
        "b/summary",
    ]
    assert len(drv.calls) == 3


# ---------------------------------------------------------------------------
# Publication durability — replay drains only verified intents
# ---------------------------------------------------------------------------


def _outbox(tmp_path, name="durable.sqlite"):
    return GraphOutbox(str(tmp_path / name))


def test_replay_drains_verified_and_keeps_unverified(tmp_path):
    ob = _outbox(tmp_path)
    try:
        ok_id = ob.append("x", {"id": "landed"}, correlation_id="r1")
        gap_id = ob.append("x", {"id": "missing"}, correlation_id="r2")

        def verify(op_type, payload):
            return payload["id"] == "landed"

        summary = ob.replay(verify=verify)
        assert summary["seen"] == 2
        assert summary["drained"] == 1
        assert summary["kept"] == 1
        assert summary["failed"] == 0
        remaining = ob.pending()
        assert [row[0] for row in remaining] == [gap_id]
        assert ok_id not in [row[0] for row in remaining]
    finally:
        ob.close()


def test_replay_dry_run_keeps_everything_unchanged(tmp_path):
    ob = _outbox(tmp_path, name="dry.sqlite")
    try:
        op_id = ob.append("x", {"id": "T1"}, correlation_id="r3")
        before = ob.pending()
        summary = ob.replay(verify=None)  # dry-run: count only
        assert summary["kept"] == 1
        assert summary["drained"] == 0
        assert ob.pending() == before
        assert ob.pending()[0][0] == op_id
    finally:
        ob.close()


def test_replay_marks_failed_over_max_retries(tmp_path):
    ob = _outbox(tmp_path, name="retries.sqlite")
    try:
        op_id = ob.append("x", {"id": "T1"}, correlation_id="r4")
        for _ in range(4):
            ob.mark_failed(op_id, "boom")
        called = []

        def verify(op_type, payload):
            called.append(payload)
            return True

        summary = ob.replay(verify=verify, max_retries=3)
        assert summary["failed"] == 1
        assert summary["drained"] == 0
        assert called == []  # retry cap checked before verify
        assert [row[0] for row in ob.pending()] == [op_id]
        assert ob.pending()[0][4] > 3  # retry_count incremented by mark_failed
    finally:
        ob.close()


def test_replay_bad_payload_is_failed_and_kept(tmp_path):
    ob = _outbox(tmp_path, name="bad.sqlite")
    try:
        ob.conn.execute(
            "INSERT INTO graph_outbox(op_type, correlation_id, payload, created_at) "
            "VALUES(?,?,?,?)",
            ("x", "r5", "{not-json", "2026-11-26T00:00:00+00:00"),
        )
        ob.conn.commit()
        summary = ob.replay(verify=lambda *_: True)
        assert summary["failed"] == 1
        assert summary["drained"] == 0
        assert len(ob.pending()) == 1
    finally:
        ob.close()


def test_replay_verify_exception_keeps_intent_pending(tmp_path):
    ob = _outbox(tmp_path, name="exc.sqlite")
    try:
        op_id = ob.append("x", {"id": "T1"}, correlation_id="r6")

        def verify(op_type, payload):
            raise RuntimeError("neo4j down")

        summary = ob.replay(verify=verify)
        assert summary["failed"] == 1
        assert summary["drained"] == 0
        assert [row[0] for row in ob.pending()] == [op_id]
    finally:
        ob.close()


def test_mark_done_survives_reopen(tmp_path):
    """A drained receipt must not come back after the process restarts."""
    path = str(tmp_path / "drain.sqlite")
    ob = GraphOutbox(path)
    ob.append("x", {"id": "T1"}, correlation_id="r7")
    summary = ob.replay(verify=lambda *_: True)
    assert summary["drained"] == 1
    ob.close()

    reopened = GraphOutbox(path)
    try:
        assert reopened.pending() == []
        assert reopened.count_pending() == 0
    finally:
        reopened.close()
