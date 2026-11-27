"""Regression tests: DB-independent publish staging must find rendered MP4s
regardless of the case of the extension (round 23, session 4).

``publish.scan_rendered`` is the resilience path used by ``queue_all_on_disk``
when Neo4j is saturated. It used a case-sensitive ``rglob("*.mp4")``, so a
rendered short saved as ``s1.MP4`` --- the same uppercase convention staging and
the rest of the media contract preserve --- was silently never queued. It also
accepted a *directory* named ``*.mp4`` as a publishable item.

All tests are hermetic: tmp_path only, no Neo4j, no network, no credentials, no
model load, and the module-level queue path is always monkeypatched away from
the configured NAS location before any write.
"""
from pathlib import Path

from auto_ingest.shorts import publish


def _topic(tmp_path: Path, name: str = "llm") -> Path:
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_scan_rendered_finds_uppercase_mp4(tmp_path):
    d = _topic(tmp_path)
    (d / "s1.MP4").write_bytes(b"x")

    items = publish.scan_rendered(tmp_path)

    assert [i.key for i in items] == ["s1"]


def test_scan_rendered_finds_mixed_case_mp4(tmp_path):
    d = _topic(tmp_path)
    (d / "s2.Mp4").write_bytes(b"x")

    items = publish.scan_rendered(tmp_path)

    assert [i.key for i in items] == ["s2"]


def test_scan_rendered_lowercase_still_found(tmp_path):
    """Control: the historical lowercase behaviour is unchanged."""
    d = _topic(tmp_path)
    (d / "s3.mp4").write_bytes(b"x")

    items = publish.scan_rendered(tmp_path)

    assert [i.key for i in items] == ["s3"]
    assert all(i.out_path.endswith("s3.mp4") for i in items)


def test_scan_rendered_uppercase_topic_and_key(tmp_path):
    d = _topic(tmp_path, name="Diffusion_Models")
    (d / "episode_007.MP4").write_bytes(b"x")

    items = publish.scan_rendered(tmp_path)

    assert len(items) == 1
    assert items[0].key == "episode_007"
    assert items[0].topic == "Diffusion_Models"
    assert items[0].title == "episode 007"


def test_scan_rendered_ignores_non_mp4_and_directory_named_mp4(tmp_path):
    d = _topic(tmp_path)
    (d / "good.mp4").write_bytes(b"x")
    (d / "notes.txt").write_text("ignore")
    (d / "clip.mov").write_bytes(b"x")
    (d / "folder.mp4").mkdir()  # a directory whose name ends in .mp4

    items = publish.scan_rendered(tmp_path)

    assert [i.key for i in items] == ["good"]


def test_queue_all_on_disk_stages_uppercase_render_idempotently(tmp_path, monkeypatch):
    """End-to-end: an uppercase render reaches the queue exactly once."""
    queue = tmp_path / "queue.jsonl"
    monkeypatch.setattr(publish, "QUEUE_PATH", queue)
    d = _topic(tmp_path)
    (d / "s4.MP4").write_bytes(b"x")

    first = publish.queue_all_on_disk(tmp_path, ["youtube_shorts"])
    second = publish.queue_all_on_disk(tmp_path, ["youtube_shorts"])

    assert first == 1
    assert second == 0
    lines = [ln for ln in queue.read_text().splitlines() if ln.strip()]
    assert len(lines) == 1
    assert '"key": "s4"' in lines[0]
