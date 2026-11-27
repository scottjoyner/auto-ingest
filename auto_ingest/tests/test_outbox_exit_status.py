"""Hermetic contract tests for the outbox CLI exit status (sweep20).

``python -m auto_ingest.outbox`` is used by drain/cron callers that key off the
process exit code. A drain that hit failed operations (corrupt payload, verify
error, or retries exhausted) must not report success: that is a fail-open exit
status and hides ops that still need operator attention. These tests seed a
synthetic SQLite outbox under pytest's tmp_path, so no Neo4j, credentials, or
live outbox is touched.
"""
from __future__ import annotations

from auto_ingest import outbox

# Deliberately not valid JSON. ``GraphOutbox.replay`` takes the json.loads branch
# before ever calling the verifier, so this fails deterministically without Neo4j.
_BAD_PAYLOAD = "{not-json"


def _seed_op(db_path: str, payload: str, correlation_id: str) -> None:
    ob = outbox.GraphOutbox(db_path)
    ob.conn.execute(
        "INSERT INTO graph_outbox(op_type, correlation_id, payload, created_at, last_error) "
        "VALUES(?,?,?,?,?)",
        ("ingest.transcription.header", correlation_id, payload,
         "2026-01-01T00:00:00+00:00", ""),
    )
    ob.conn.commit()
    ob.close()


def test_main_nonzero_when_replay_has_failed_ops(tmp_path, monkeypatch):
    """A corrupt-payload op makes replay fail; the CLI must not exit 0."""
    db = tmp_path / "outbox.sqlite"
    _seed_op(str(db), _BAD_PAYLOAD, "cid-bad")
    monkeypatch.setenv("AUTO_INGEST_OUTBOX_PATH", str(db))

    rc = outbox.main([])

    assert rc != 0, "drain with failed ops reported success (fail-open)"


def test_main_zero_when_drain_has_no_failures(tmp_path, monkeypatch):
    """A clean dry-run (ops kept, none failed) still exits 0."""
    db = tmp_path / "outbox.sqlite"
    ob = outbox.GraphOutbox(str(db))
    ob.append("ingest.transcription.header", {"t_id": "T1"}, correlation_id="cid-ok")
    ob.close()
    monkeypatch.setenv("AUTO_INGEST_OUTBOX_PATH", str(db))

    rc = outbox.main(["--dry-run"])

    assert rc == 0, "a drain with no failed ops must still report success"
