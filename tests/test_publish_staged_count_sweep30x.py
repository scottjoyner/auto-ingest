"""Hermetic contract: queue_all_unpublished() reports entries actually appended.

Pre-fix the helper counts every short it *iterates*, including idempotent
re-queues that append nothing, so a CLI rerun prints "Queued N short(s)" while
the queue file does not grow. All fakes are in-process: no Neo4j, network,
credentials, platform API or rendered media.
"""
from __future__ import annotations

import json

import pytest

from auto_ingest.shorts import publish


def _items(*keys: str) -> list:
    return [
        publish.Publishable(
            key=k, topic="t", title=f"Title {k}", out_path=f"/rendered/{k}.mp4"
        )
        for k in keys
    ]


def _lines(queue_path) -> list:
    if not queue_path.exists():
        return []
    return [line for line in queue_path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture
def staged(monkeypatch, tmp_path):
    queue = tmp_path / "publish_queue.jsonl"
    monkeypatch.setattr(publish, "QUEUE_PATH", queue)
    return queue


def test_first_stage_counts_every_appended_entry(staged, monkeypatch):
    monkeypatch.setattr(publish, "list_unpublished", lambda limit=50: _items("s1", "s2"))
    assert publish.queue_all_unpublished(platform="youtube_shorts") == 2
    assert len(_lines(staged)) == 2


def test_idempotent_rerun_reports_zero_not_iterated_count(staged, monkeypatch):
    """The defect: a no-op rerun must not claim shorts were staged."""
    monkeypatch.setattr(publish, "list_unpublished", lambda limit=50: _items("s1", "s2"))
    assert publish.queue_all_unpublished(platform="youtube_shorts") == 2
    before = _lines(staged)

    reported = publish.queue_all_unpublished(platform="youtube_shorts")

    assert reported == 0, "re-run reported staged shorts but appended nothing"
    assert _lines(staged) == before


def test_new_platform_entries_are_still_counted(staged, monkeypatch):
    monkeypatch.setattr(publish, "list_unpublished", lambda limit=50: _items("s1", "s2"))
    publish.queue_all_unpublished(platform="youtube_shorts")
    before = _lines(staged)

    reported = publish.queue_all_unpublished(platform="tiktok")

    assert reported == 2
    assert len(_lines(staged)) - len(before) == 2
    assert {json.loads(line)["platform"] for line in _lines(staged)} == {
        "youtube_shorts", "tiktok"
    }


def test_missing_out_path_is_never_counted(staged, monkeypatch):
    monkeypatch.setattr(publish, "list_unpublished", lambda limit=50: [
        publish.Publishable(key="s3", topic="t", title="T3", out_path=""),
    ])
    assert publish.queue_all_unpublished(platform="youtube_shorts") == 0
    assert _lines(staged) == []
