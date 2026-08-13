from __future__ import annotations

import json
from pathlib import Path

import pytest

from auto_ingest.file_queue import QueueJobError, claim_one, enqueue, load_job, process_claimed


def test_enqueue_is_typed_and_atomic(tmp_path: Path):
    path = enqueue(tmp_path, "sync", job_id="abc", now=123)
    assert path.name == "abc.job.json"
    job = load_job(path)
    assert job.job_id == "abc"
    assert job.profile == "sync"
    assert job.created_at == 123
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob(".*.tmp"))


def test_unknown_profile_and_unknown_fields_rejected(tmp_path: Path):
    with pytest.raises(QueueJobError):
        enqueue(tmp_path, "shell", job_id="bad", now=123)

    path = tmp_path / "evil.job.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "job_id": "evil",
                "profile": "sync",
                "created_at": 123,
                "metadata": {},
                "command": "rm -rf /",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(QueueJobError):
        load_job(path)


def test_legacy_executable_job_is_rejected_not_executed(tmp_path: Path):
    legacy = tmp_path / "danger.job"
    legacy.write_text("touch /tmp/should-never-run\n", encoding="utf-8")
    assert claim_one(tmp_path, owner="ci") is None
    assert not legacy.exists()
    rejected = list((tmp_path / "rejected").glob("*.legacy-shell-rejected"))
    assert len(rejected) == 1
    assert "touch /tmp/should-never-run" in rejected[0].read_text(encoding="utf-8")


def test_failed_typed_job_returns_to_queue_for_retry(tmp_path: Path):
    enqueue(tmp_path, "sync", job_id="retry-me", now=123)
    claimed = claim_one(tmp_path, owner="ci")
    assert claimed is not None

    def runner(_driver, profile, **kwargs):
        assert profile == "sync"
        assert kwargs["job_key"] == "filequeue:retry-me"
        return 1

    rc = process_claimed(None, claimed, root=tmp_path, owner="ci", runner=runner)
    assert rc == 1
    assert (tmp_path / "retry-me.job.json").exists()


def test_successful_typed_job_moves_to_done(tmp_path: Path):
    enqueue(tmp_path, "sync", job_id="done-me", now=123)
    claimed = claim_one(tmp_path, owner="ci")
    assert claimed is not None

    rc = process_claimed(
        None,
        claimed,
        root=tmp_path,
        owner="ci",
        runner=lambda *_args, **_kwargs: 0,
    )
    assert rc == 0
    assert (tmp_path / "done" / "done-me.done.json").exists()
