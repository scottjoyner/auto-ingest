"""Per-(key, platform) identity of the published marker (sweep39, offline).

No network, no credentials, no Neo4j, no upload adapter, no real media: the
marker is a tiny JSON file next to a one-byte placeholder in pytest's tmp_path.
"""
from __future__ import annotations

import json
from pathlib import Path

from auto_ingest.shorts.uploader import (
    QueueItem,
    _published_marker,
    _record_published,
    already_published,
)


def _item(tmp_path: Path, platform: str) -> QueueItem:
    out = tmp_path / "short.mp4"
    out.write_bytes(b"x")  # a byte is enough; only the marker next to it matters
    return QueueItem(
        key="short", topic="topic", title="Short",
        out_path=str(out), platform=platform,
    )


def test_second_platform_is_not_blocked_by_first_platform_marker(tmp_path):
    """A YouTube post must not silently suppress the TikTok queue entry."""
    yt = _item(tmp_path, "youtube_shorts")
    tiktok = _item(tmp_path, "tiktok")

    assert already_published(yt) is False
    assert already_published(tiktok) is False

    _record_published(yt, url="https://yt.example/s")

    assert already_published(yt) is True
    # DEFECT on the base: the marker's mere existence answers for every platform.
    assert already_published(tiktok) is False

    _record_published(tiktok, url="https://tt.example/s")

    assert already_published(yt) is True
    assert already_published(tiktok) is True


def test_legacy_single_platform_marker_is_honoured(tmp_path):
    """An existing single-platform marker still means published for that one."""
    out = tmp_path / "short.mp4"
    out.write_bytes(b"x")
    _published_marker(out).write_text(json.dumps({
        "key": "short", "platform": "youtube_shorts", "url": "u", "at": 0,
    }))

    yt = _item(tmp_path, "youtube_shorts")
    tiktok = _item(tmp_path, "tiktok")

    assert already_published(yt) is True
    assert already_published(tiktok) is False


def test_marker_accumulates_every_published_platform(tmp_path):
    """Both posts stay recorded, so neither platform can post twice on a re-run."""
    yt = _item(tmp_path, "youtube_shorts")
    tiktok = _item(tmp_path, "tiktok")

    _record_published(yt, url="u1")
    _record_published(tiktok, url="u2")

    rec = json.loads(_published_marker(Path(yt.out_path)).read_text())
    assert rec["key"] == "short"
    assert set(rec["platforms"]) == {"youtube_shorts", "tiktok"}


def test_unreadable_marker_fails_closed(tmp_path):
    """A marker we cannot read still means 'do not post this again'."""
    item = _item(tmp_path, "tiktok")
    _published_marker(Path(item.out_path)).write_text("not json")

    assert already_published(item) is True


def test_absent_marker_means_unpublished(tmp_path):
    item = _item(tmp_path, "tiktok")
    assert already_published(item) is False
