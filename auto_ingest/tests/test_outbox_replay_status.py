"""Deterministic tests for ``GraphOutbox.replay`` per-op status reporting.

Regression: ``replay()`` accepts ``on_status(op_id, op_type, ok)`` so a caller
can render one status line per examined op. The callback was only invoked on
the final drained/kept path; ops that are only *counted* (dry-run, ``verify is
None``) and ops marked failed (verify error / over-retry / bad payload)
``continue`` before reaching it. Those ops -- exactly the ones a report needs
to surface -- were silently omitted from ``on_status`` even though they are
included in the summary's ``seen`` count.

Hermetic: uses only the dependency-free SQLite outbox and pytest's ``tmp_path``
(redirected to the private SSD scratch dir via TMPDIR). No Neo4j, no ML stack.
"""

from __future__ import annotations

from auto_ingest.outbox import GraphOutbox


def _outbox(tmp_path):
    return GraphOutbox(str(tmp_path / "outbox.sqlite"))


def test_dry_run_reports_every_op(tmp_path):
    """Dry-run is a *reporting* mode: on_status must see every pending op."""
    ob = _outbox(tmp_path)
    try:
        a = ob.append("ingest.transcription.header", {"t_id": "T1"})
        b = ob.append("ingest.transcription.header", {"t_id": "T2"})
        seen = []
        summary = ob.replay(verify=None, on_status=lambda i, t, ok: seen.append((i, t, ok)))
        # Summary counts both ops ...
        assert summary["seen"] == 2
        assert summary["kept"] == 2
        # ... so the per-op reporter must not silently omit them.
        assert sorted(s[0] for s in seen) == sorted([a, b]), (
            "dry-run on_status silently omitted pending ops"
        )
        assert all(s[2] is False for s in seen), "dry-run ops are kept, not drained"
    finally:
        ob.close()


def test_verify_error_op_is_reported(tmp_path):
    """An op whose verifier raises is marked failed; on_status must see it."""
    ob = _outbox(tmp_path)
    try:
        op_id = ob.append("ingest.transcription.header", {"t_id": "T3"})
        seen = []

        def boom(op_type, payload):
            raise RuntimeError("verifier down")

        summary = ob.replay(verify=boom, on_status=lambda i, t, ok: seen.append((i, t, ok)))
        assert summary["failed"] == 1
        assert [s[0] for s in seen] == [op_id], (
            "failed op was silently omitted from on_status"
        )
        assert seen[0][2] is False, "a failed op is not drained"
    finally:
        ob.close()


def test_drained_and_kept_still_reported(tmp_path):
    """Guard the existing behavior: normal verify outcomes still call on_status."""
    ob = _outbox(tmp_path)
    try:
        good = ob.append("x", {"t_id": "keep"})
        bad = ob.append("x", {"t_id": "drop"})
        seen = []
        summary = ob.replay(
            verify=lambda _t, p: p["t_id"] == "keep",
            on_status=lambda i, t, ok: seen.append((i, t, ok)),
        )
        assert summary["drained"] == 1 and summary["kept"] == 1
        by_id = {s[0]: s[2] for s in seen}
        assert by_id == {good: True, bad: False}
    finally:
        ob.close()
