from __future__ import annotations

from pathlib import Path

import pytest

from auto_ingest.artifacts import build_identity, sha256_bytes
from auto_ingest.commit_protocol import ArtifactCommitError, atomic_commit_bytes
from auto_ingest.stage_executor import recover_artifact


def test_artifact_identity_is_deterministic_and_config_sensitive():
    a = build_identity(
        source_hash="abc", stage="transcribed", stage_version="v2",
        config={"language": "en", "beam": 5}, model={"name": "whisper", "rev": "1"},
    )
    b = build_identity(
        source_hash="abc", stage="transcribed", stage_version="v2",
        config={"beam": 5, "language": "en"}, model={"rev": "1", "name": "whisper"},
    )
    c = build_identity(
        source_hash="abc", stage="transcribed", stage_version="v2",
        config={"language": "en", "beam": 6}, model={"name": "whisper", "rev": "1"},
    )
    assert a.artifact_id == b.artifact_id
    assert a.artifact_id != c.artifact_id


def test_atomic_commit_never_exposes_partial_final_file(tmp_path: Path):
    final = tmp_path / "artifact.bin"

    def crash(_temp):
        raise RuntimeError("simulated power loss before rename")

    with pytest.raises(RuntimeError):
        atomic_commit_bytes(final, b"complete-payload", before_replace=crash)
    assert not final.exists()
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob(".*.tmp"))


def test_atomic_commit_reuses_identical_artifact(tmp_path: Path):
    final = tmp_path / "artifact.bin"
    digest = sha256_bytes(b"payload")
    first = atomic_commit_bytes(final, b"payload", expected_sha256=digest)
    second = atomic_commit_bytes(final, b"payload", expected_sha256=digest)
    assert first.reused is False
    assert second.reused is True
    assert final.read_bytes() == b"payload"


def test_atomic_commit_rejects_conflicting_existing_artifact(tmp_path: Path):
    final = tmp_path / "artifact.bin"
    final.write_bytes(b"wrong")
    with pytest.raises(ArtifactCommitError):
        atomic_commit_bytes(final, b"right", expected_sha256=sha256_bytes(b"right"))


def test_recovery_without_journal_is_missing(tmp_path: Path):
    assert recover_artifact(tmp_path, "0" * 64) == "MISSING"
