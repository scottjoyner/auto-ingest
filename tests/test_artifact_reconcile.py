from __future__ import annotations

import json
from pathlib import Path

from auto_ingest.artifact_reconcile import classify_record, reconcile
from auto_ingest.artifacts import sha256_bytes


class _Record:
    def __init__(self, n):
        self.n = n

    def get(self, key):
        return self.n if key == "n" else None


class _Result:
    def __init__(self, n):
        self.n = n

    def single(self):
        return _Record(self.n)


class _Session:
    def __init__(self, present):
        self.present = present

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, _query, **params):
        return _Result(1 if params["id"] in self.present else 0)


class _Driver:
    def __init__(self, present=()):
        self.present = set(present)

    def session(self):
        return _Session(self.present)


def _record(path: Path, artifact_id="aid", state="COMMITTED"):
    data = path.read_bytes() if path.exists() else b"missing"
    return {
        "artifact_id": artifact_id,
        "state": state,
        "artifact_path": str(path),
        "artifact_sha256": sha256_bytes(data),
    }


def test_reconciler_classifies_healthy(tmp_path: Path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"ok")
    finding = classify_record(_Driver({"aid"}), _record(artifact))
    assert finding.classification == "HEALTHY"


def test_reconciler_classifies_orphan_file(tmp_path: Path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"ok")
    finding = classify_record(_Driver(), _record(artifact, state="ARTIFACT_COMMITTED"))
    assert finding.classification == "ORPHAN_FILE"


def test_reconciler_classifies_dangling_graph(tmp_path: Path):
    missing = tmp_path / "missing.bin"
    finding = classify_record(_Driver({"aid"}), _record(missing))
    assert finding.classification == "GRAPH_DANGLING"


def test_reconcile_scans_commit_journal(tmp_path: Path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"ok")
    journal = tmp_path / ".commit-journal" / "aa" / "aid.json"
    journal.parent.mkdir(parents=True)
    journal.write_text(json.dumps(_record(artifact)), encoding="utf-8")
    findings = reconcile(tmp_path, _Driver({"aid"}))
    assert [f.classification for f in findings] == ["HEALTHY"]
