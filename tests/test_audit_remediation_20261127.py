"""Regression tests for the 2026-11-27 audit remediation slice.

Covers: partial-failure ingest, traversal attempts, identifier validation,
and the job-trigger default fail-closed behavior.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import types
import unittest.mock as mock
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers for importing transcripts with heavy ML deps stubbed out.
# ---------------------------------------------------------------------------

_TRANSCRIPT_STUBS = {
    "torch": mock.MagicMock(),
    "torch.nn": mock.MagicMock(),
    "torch.nn.functional": mock.MagicMock(),
    "transformers": mock.MagicMock(),
    "neo4j": mock.MagicMock(),
    "neo4j.exceptions": mock.MagicMock(),
}


@pytest.fixture(scope="module")
def transcripts_mod():
    with mock.patch.dict(sys.modules, _TRANSCRIPT_STUBS):
        mod = importlib.import_module("auto_ingest.ingest.transcripts")
        yield mod


class _FakeResult:
    def __init__(self, row):
        self._row = row

    def single(self):
        return self._row


class _FakeSession:
    def __init__(self, rows):
        # rows is a list consumed in order per run() call
        self._rows = list(rows)
        self.calls = []

    def run(self, cy, **kwargs):
        self.calls.append((cy, kwargs))
        row = self._rows.pop(0) if self._rows else {"written": 0}
        return _FakeResult(row)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeDriver:
    def __init__(self, session_rows):
        self._session_rows = list(session_rows)
        self.sessions = []

    def session(self, database=None):
        rows = self._session_rows.pop(0) if self._session_rows else []
        sess = _FakeSession(rows)
        self.sessions.append(sess)
        return sess


# ---------------------------------------------------------------------------
# 1) Partial-failure ingest: parent must exist; report rows actually written
# ---------------------------------------------------------------------------

def test_chunked_child_write_fails_loud_when_parent_missing(transcripts_mod):
    T = transcripts_mod
    # first session: parent existence check returns 0
    driver = _FakeDriver([[{"n": 0}]])
    with pytest.raises(RuntimeError) as ei:
        T.ingest_segments_chunked(driver, "missing-tid", [{"id": "s1"}], batch=10)
    assert "parent Transcription missing" in str(ei.value)
    assert "missing-tid" in str(ei.value)


def test_chunked_child_write_reports_actual_rows_written(transcripts_mod):
    T = transcripts_mod
    segs = [{"id": "s1", "idx": 0, "start": 0.0, "end": 1.0, "text": "a",
             "tokens_count": 1, "embedding": [0.0], "abs_start": None,
             "abs_end": None, "speaker_label": "S1", "speaker_id": "sp1"}]
    # first session: parent exists; second session: batch write returns written=1
    driver = _FakeDriver([[{"n": 1}], [{"written": 1}]])
    written = T.ingest_segments_chunked(driver, "tid1", segs, batch=10)
    assert written == 1


def test_chunked_child_write_fails_if_batch_underwrites(transcripts_mod):
    T = transcripts_mod
    segs = [{"id": "s1", "idx": 0, "start": 0.0, "end": 1.0, "text": "a",
             "tokens_count": 1, "embedding": [0.0], "abs_start": None,
             "abs_end": None, "speaker_label": "S1", "speaker_id": "sp1"},
            {"id": "s2", "idx": 1, "start": 1.0, "end": 2.0, "text": "b",
             "tokens_count": 1, "embedding": [0.0], "abs_start": None,
             "abs_end": None, "speaker_label": "S1", "speaker_id": "sp1"}]
    # parent exists, but write reports 1 row when 2 were attempted
    driver = _FakeDriver([[{"n": 1}], [{"written": 1}]])
    with pytest.raises(RuntimeError) as ei:
        T.ingest_segments_chunked(driver, "tid1", segs, batch=10)
    assert "missing during write" in str(ei.value)
    assert "expected 2 rows written, got 1" in str(ei.value)


# ---------------------------------------------------------------------------
# 2) Traversal attempts: custody verify/execute must reject escaping keys
# ---------------------------------------------------------------------------

def test_verify_destination_rejects_escaping_ledger_key(tmp_path):
    from auto_ingest.custody.verify import verify_destination
    from auto_ingest.custody.ledger import ledger_dir, HASH_LEDGER

    bundle = tmp_path / "bundle"
    dest = tmp_path / "dest"
    dest.mkdir()
    # create hash ledger with an escaping key
    ldir = ledger_dir(bundle)
    ldir.mkdir(parents=True, exist_ok=True)
    (ldir / HASH_LEDGER).write_text(
        json.dumps({"key": "../escape.txt", "status": "verified", "digest": "abc", "size": 1}) + "\n",
        encoding="utf-8",
    )

    result = verify_destination(bundle, dest)
    assert result.failed == 1
    assert result.verified == 0
    assert any("key_escapes_destination" in e for e in result.errors)


def test_executor_safe_join_returns_resolved_and_rejects_escape(tmp_path):
    from auto_ingest.custody.executor import _safe_join

    root = tmp_path / "dest"
    root.mkdir()
    safe = _safe_join(root, "a/b.txt")
    assert safe is not None
    assert safe.is_absolute()
    assert safe == (root / "a/b.txt").resolve()

    assert _safe_join(root, "../escape.txt") is None
    assert _safe_join(root, "/etc/passwd") is None


def test_stream_copy_uses_unique_partial_name(tmp_path):
    from auto_ingest.custody.executor import stream_copy, TEMP_DIRNAME

    src = tmp_path / "src.bin"
    src.write_bytes(b"payload")
    dest = tmp_path / "dest" / "out.bin"
    digest, written = stream_copy(src, dest)
    assert written == len(b"payload")
    assert dest.read_bytes() == b"payload"

    tmp_dir = dest.parent / TEMP_DIRNAME
    # No deterministic .partial left behind; if any temp remains, it must be unique-named
    leftovers = list(tmp_dir.glob("*.partial")) if tmp_dir.exists() else []
    assert leftovers == []


# ---------------------------------------------------------------------------
# 3) Identifier validation
# ---------------------------------------------------------------------------

def test_ingest_import_rejects_invalid_identifiers(tmp_path):
    from auto_ingest.ingest_import import import_intermediate

    # write a minimal intermediate file
    p = tmp_path / "k.metadata.json"
    p.write_text(json.dumps([{"id": "1", "v": "x"}]), encoding="utf-8")

    class _NoopSess:
        def run(self, *a, **k):
            return None
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    class _NoopDriver:
        def session(self):
            return _NoopSess()

    with pytest.raises(ValueError) as ei:
        import_intermediate(_NoopDriver(), str(p), label="Chunk;DROP", key_field="id")
    assert "Invalid Cypher identifier for label" in str(ei.value)

    with pytest.raises(ValueError):
        import_intermediate(_NoopDriver(), str(p), label="Chunk", key_field="id;bad")

    with pytest.raises(ValueError):
        import_intermediate(
            _NoopDriver(), str(p), label="Chunk", key_field="id",
            parent_label="Transcription", parent_field="id", edge_label="HAS:BAD",
        )


def test_transcripts_embed_extra_prop_validation(transcripts_mod, monkeypatch):
    T = transcripts_mod
    with pytest.raises(ValueError) as ei:
        T._validate_cypher_identifier("bad-name", name="EMBED_EXTRA_PROP")
    assert "EMBED_EXTRA_PROP" in str(ei.value)
    # valid identifier passes
    assert T._validate_cypher_identifier("emb_gte_small", name="EMBED_EXTRA_PROP") == "emb_gte_small"


# ---------------------------------------------------------------------------
# 4) Job-trigger default fail-closed and scan root validation
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def job_trigger_api():
    import importlib.util

    here = Path(__file__).resolve().parents[1]
    mod_path = here / "deploy" / "job_trigger_api.py"
    spec = importlib.util.spec_from_file_location("job_trigger_api_under_test", mod_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def test_job_trigger_main_fails_closed_without_auth(monkeypatch, job_trigger_api):
    api = job_trigger_api
    monkeypatch.delenv("JOB_TRIGGER_API_TOKEN", raising=False)
    monkeypatch.delenv("JOB_TRIGGER_API_ALLOW_INSECURE", raising=False)
    with pytest.raises(SystemExit) as ei:
        api.main()
    assert "Refusing to start job trigger API without auth" in str(ei.value)


def test_job_trigger_scan_roots_require_allowlist(monkeypatch, job_trigger_api):
    api = job_trigger_api
    monkeypatch.delenv("JOB_TRIGGER_ALLOWED_ROOTS", raising=False)
    handler = object.__new__(api.JobTriggerHandler)
    with pytest.raises(ValueError) as ei:
        handler._safe_scan_roots("/tmp/data")
    assert "JOB_TRIGGER_ALLOWED_ROOTS is not configured" in str(ei.value)


def test_job_trigger_scan_roots_reject_outside_allowlist(tmp_path, monkeypatch, job_trigger_api):
    api = job_trigger_api
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    monkeypatch.setenv("JOB_TRIGGER_ALLOWED_ROOTS", str(allowed))
    handler = object.__new__(api.JobTriggerHandler)

    ok = handler._safe_scan_roots(str(allowed / "subdir"))
    assert ok == str((allowed / "subdir").resolve())

    with pytest.raises(ValueError) as ei:
        handler._safe_scan_roots(str(tmp_path / "outside"))
    assert "outside allowed roots" in str(ei.value)


def test_job_trigger_auth_required(monkeypatch, job_trigger_api):
    api = job_trigger_api
    monkeypatch.setenv("JOB_TRIGGER_API_TOKEN", "secret-token")
    monkeypatch.delenv("JOB_TRIGGER_API_ALLOW_INSECURE", raising=False)

    handler = object.__new__(api.JobTriggerHandler)
    handler.headers = {"Authorization": "Bearer wrong"}
    responses = []
    handler._send_json = lambda status, data: responses.append((status, data))

    assert handler._require_auth() is False
    assert responses[0][0] == 401

    handler.headers = {"Authorization": "Bearer secret-token"}
    responses.clear()
    assert handler._require_auth() is True
    assert responses == []
