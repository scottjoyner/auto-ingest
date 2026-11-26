"""Round-8 tests: durable-outbox lifecycle, idempotency and opt-in gating.

``auto_ingest/outbox.py`` is a dependency-free SQLite store. R5-4
(``auto_ingest/tests/test_receipt_stage_durability_r5.py``) pinned the *replay*
contract (drain/keep, max-retries, bad payload, verify exception, reopen) and
``auto_ingest/tests/test_light.py`` pinned a basic round-trip and retry bump.
This file covers the rest of the surface: payload fidelity, idempotent/targeted
deletion, durable failure state, pending-limit ordering, the replay status
callback and batch cap, and the ``get_outbox`` opt-in singleton.

Every test uses a pytest ``tmp_path`` SQLite file (never a live/production DB),
no Neo4j, no network, no service, no ML imports.
"""

from __future__ import annotations

import json

from auto_ingest import outbox as outbox_mod
from auto_ingest.outbox import GraphOutbox, get_outbox


def _new(tmp_path, name="lifecycle.sqlite"):
    return GraphOutbox(str(tmp_path / name))


# ---------------------------------------------------------------------------
# Append / pending fidelity
# ---------------------------------------------------------------------------


def test_append_preserves_payload_types_and_defaults(tmp_path):
    ob = _new(tmp_path)
    try:
        payload = {"t_id": "T1", "n": 3, "nested": {"flags": [True, None]}}
        op_id = ob.append("ingest.header", payload)
        row = ob.pending()
        assert len(row) == 1
        assert row[0][0] == op_id
        assert row[0][1] == "ingest.header"
        assert row[0][2] == ""  # default correlation id
        assert json.loads(row[0][3]) == payload  # types survive the JSON hop
        assert row[0][4] == 0  # retry_count starts at zero
        stored = ob.conn.execute(
            "SELECT last_error FROM graph_outbox WHERE id=?", (op_id,)
        ).fetchone()
        assert stored[0] == ""
    finally:
        ob.close()


def test_pending_is_ordered_by_id_and_respects_limit(tmp_path):
    ob = _new(tmp_path)
    try:
        ids = [ob.append("x", {"i": i}) for i in range(3)]
        assert [row[0] for row in ob.pending()] == ids
        assert [row[0] for row in ob.pending(limit=2)] == ids[:2]
        assert ob.pending(limit=0) == []
    finally:
        ob.close()


# ---------------------------------------------------------------------------
# Idempotent / targeted state transitions
# ---------------------------------------------------------------------------


def test_mark_done_is_targeted_and_idempotent(tmp_path):
    ob = _new(tmp_path)
    try:
        first = ob.append("x", {"id": "a"})
        second = ob.append("x", {"id": "b"})
        ob.mark_done(first)
        assert [row[0] for row in ob.pending()] == [second]
        assert ob.count_pending() == 1
        # Re-marking an already drained id is a no-op; it must not delete the
        # other pending op.
        ob.mark_done(first)
        assert [row[0] for row in ob.pending()] == [second]
        ob.mark_done(second)
        assert ob.pending() == []
        assert ob.count_pending() == 0
    finally:
        ob.close()


def test_mark_failed_accumulates_retries_and_persists_last_error(tmp_path):
    path = str(tmp_path / "failed.sqlite")
    ob = GraphOutbox(path)
    op_id = ob.append("x", {"id": "T1"}, correlation_id="r8")
    ob.mark_failed(op_id, "first error")
    ob.mark_failed(op_id, "second error")
    assert ob.pending()[0][4] == 2
    ob.close()

    reopened = GraphOutbox(path)
    try:
        row = reopened.conn.execute(
            "SELECT retry_count, last_error FROM graph_outbox WHERE id=?", (op_id,)
        ).fetchone()
        assert row == (2, "second error")
        assert reopened.count_pending() == 1
    finally:
        reopened.close()


def test_mark_failed_keeps_the_op_pending(tmp_path):
    ob = _new(tmp_path)
    try:
        op_id = ob.append("x", {"id": "T1"})
        ob.mark_failed(op_id, "boom")
        assert ob.count_pending() == 1
        assert [row[0] for row in ob.pending()] == [op_id]
    finally:
        ob.close()


def test_count_pending_tracks_the_full_lifecycle(tmp_path):
    ob = _new(tmp_path)
    try:
        assert ob.count_pending() == 0
        first = ob.append("x", {"id": "a"})
        second = ob.append("x", {"id": "b"})
        assert ob.count_pending() == 2
        ob.mark_failed(first, "transient")
        assert ob.count_pending() == 2
        ob.mark_done(first)
        assert ob.count_pending() == 1
        ob.mark_done(second)
        assert ob.count_pending() == 0
    finally:
        ob.close()


# ---------------------------------------------------------------------------
# Replay control surface (distinct from R5-4's drain/keep cases)
# ---------------------------------------------------------------------------


def test_replay_on_status_reports_each_processed_op(tmp_path):
    ob = _new(tmp_path, name="status.sqlite")
    try:
        landed = ob.append("x", {"id": "landed"})
        missing = ob.append("x", {"id": "missing"})
        seen = []
        summary = ob.replay(
            verify=lambda _op_type, payload: payload["id"] == "landed",
            on_status=lambda op_id, op_type, ok: seen.append((op_id, op_type, ok)),
        )
        assert seen == [(landed, "x", True), (missing, "x", False)]
        assert (summary["drained"], summary["kept"]) == (1, 1)
    finally:
        ob.close()


def test_dry_run_replay_emits_no_status_callbacks(tmp_path):
    ob = _new(tmp_path, name="dry-status.sqlite")
    try:
        ob.append("x", {"id": "T1"})
        seen = []
        summary = ob.replay(verify=None, on_status=lambda *a: seen.append(a))
        assert seen == []
        assert summary["kept"] == 1
    finally:
        ob.close()


def test_replay_max_ops_bounds_the_batch_and_keeps_the_rest(tmp_path):
    ob = _new(tmp_path, name="maxops.sqlite")
    try:
        ids = [ob.append("x", {"i": i}) for i in range(5)]
        summary = ob.replay(verify=None, max_ops=2)
        assert summary["seen"] == 2
        assert summary["kept"] == 2
        assert [row[0] for row in ob.pending()] == ids  # nothing mutated
    finally:
        ob.close()


def test_replay_summary_reports_the_retry_ceiling(tmp_path):
    ob = _new(tmp_path, name="ceiling.sqlite")
    try:
        summary = ob.replay(verify=None, max_retries=7)
        assert summary["max_retries"] == 7
        assert set(summary) == {"drained", "kept", "failed", "seen", "max_retries"}
    finally:
        ob.close()


# ---------------------------------------------------------------------------
# get_outbox opt-in singleton
# ---------------------------------------------------------------------------


def test_get_outbox_is_none_when_opt_in_is_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("AUTO_INGEST_OUTBOX", raising=False)
    monkeypatch.setattr(outbox_mod, "_OUTBOX", None)
    assert get_outbox() is None


def test_get_outbox_uses_env_path_when_enabled(tmp_path, monkeypatch):
    target = str(tmp_path / "env-outbox.sqlite")
    monkeypatch.setenv("AUTO_INGEST_OUTBOX", "1")
    monkeypatch.setenv("AUTO_INGEST_OUTBOX_PATH", target)
    monkeypatch.setattr(outbox_mod, "_OUTBOX", None)
    ob = get_outbox()
    try:
        assert isinstance(ob, GraphOutbox)
        assert ob.path == target
        assert (tmp_path / "env-outbox.sqlite").exists()
    finally:
        ob.close()


def test_explicit_path_overrides_env_path(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTO_INGEST_OUTBOX", "1")
    monkeypatch.setenv("AUTO_INGEST_OUTBOX_PATH", str(tmp_path / "env.sqlite"))
    monkeypatch.setattr(outbox_mod, "_OUTBOX", None)
    explicit = str(tmp_path / "explicit.sqlite")
    ob = get_outbox(explicit)
    try:
        assert ob.path == explicit
        assert not (tmp_path / "env.sqlite").exists()
    finally:
        ob.close()


def test_blank_explicit_path_falls_back_to_env_path(tmp_path, monkeypatch):
    target = str(tmp_path / "fallback.sqlite")
    monkeypatch.setenv("AUTO_INGEST_OUTBOX", "1")
    monkeypatch.setenv("AUTO_INGEST_OUTBOX_PATH", target)
    monkeypatch.setattr(outbox_mod, "_OUTBOX", None)
    ob = get_outbox("   ")
    try:
        assert ob.path == target
    finally:
        ob.close()


def test_get_outbox_returns_the_same_singleton(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTO_INGEST_OUTBOX", "1")
    monkeypatch.setenv("AUTO_INGEST_OUTBOX_PATH", str(tmp_path / "single.sqlite"))
    monkeypatch.setattr(outbox_mod, "_OUTBOX", None)
    first = get_outbox()
    second = get_outbox()
    try:
        assert first is second
        assert outbox_mod._OUTBOX is first
    finally:
        first.close()
