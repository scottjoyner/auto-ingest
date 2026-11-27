"""
Tests for the posting-scheduler activation: scheduling.consume_calendar_for_day
plus the `publish schedule` CLI action (dry-run, report-only).

Pure: no network, no credentials, no real queue writes.
"""

from __future__ import annotations

import json
from pathlib import Path

from auto_ingest.shorts import publish, scheduling, uploader
from auto_ingest.shorts.cli import build_parser


def _rendered_short(out_root: Path, topic: str = "large_language_models",
                    key: str = "short_a") -> Path:
    """Create one on-disk rendered MP4 in the layout scan_rendered expects."""
    d = out_root / topic
    d.mkdir(parents=True, exist_ok=True)
    mp4 = d / f"{key}.mp4"
    mp4.write_bytes(b"fake mp4")
    return mp4


def test_consume_calendar_for_day_returns_day1_slots():
    avail = [{"short_id": f"s{i}", "topic": t} for i, t in enumerate(
        ["large_language_models", "diffusion_models", "graph_neural_networks",
         "reinforcement_learning", "computer_vision",
         "retrieval_augmented_generation", "robotics"] * 3)]
    # Day 1 is in the warm-up 1-3 window: only youtube_shorts, 1 slot.
    slots = scheduling.consume_calendar_for_day(1, available=avail)
    assert len(slots) == 1
    assert slots[0].platform == "youtube_shorts"
    assert slots[0].day == 1

    # Day 7 is in the 4-7 window: 3 platforms, 3 slots total.
    slots7 = scheduling.consume_calendar_for_day(7, available=avail)
    assert len(slots7) == 3
    assert {s.platform for s in slots7} == {
        "youtube_shorts", "tiktok", "instagram"}


def test_consume_calendar_for_day_returns_requested_day_only():
    # consume_calendar_for_day(day) builds a calendar of length `day`, so it
    # always returns the slots for exactly that day (never empty for day>=1).
    avail = [{"short_id": f"s{i}", "topic": "x"} for i in range(20)]
    d1 = scheduling.consume_calendar_for_day(1, available=avail)
    d3 = scheduling.consume_calendar_for_day(3, available=avail)
    assert all(s.day == 1 for s in d1)
    assert all(s.day == 3 for s in d3)
    assert len(d1) == 1  # warm-up 1-3: 1 slot
    assert len(d3) == 1


def test_calendar_to_queue_slots_flattens():
    avail = [{"short_id": f"s{i}", "topic": "x"} for i in range(20)]
    plans = scheduling.build_calendar(days=3, available=avail)
    flat = scheduling.calendar_to_queue_slots(plans)
    assert len(flat) == sum(len(dp.slots) for dp in plans)
    assert all(hasattr(s, "short_id") for s in flat)


def test_cli_publish_schedule_dry_run_no_queuing(capsys, monkeypatch, tmp_path):
    # Point the real queue at a temp file so --apply (if accidentally set)
    # can't touch the real NAS queue.
    monkeypatch.setattr("auto_ingest.shorts.publish.QUEUE_PATH", tmp_path / "queue.jsonl")
    monkeypatch.setattr("auto_ingest.shorts.uploader.QUEUE_PATH", tmp_path / "queue.jsonl")

    args = build_parser().parse_args([
        "publish", "schedule", "--days", "3",
        "--out-root", str(tmp_path / "rendered"),
    ])
    assert args.action == "schedule"
    rc = args.func(args)
    out = capsys.readouterr().out
    assert rc == 0
    assert "DRY-RUN" in out
    assert "youtube_shorts" in out
    # No queue file should be written in dry-run.
    assert not (tmp_path / "queue.jsonl").exists()


def test_slot_carries_persona_source_and_needs_render(monkeypatch):
    monkeypatch.delenv("PERSONA_SOURCE", raising=False)
    avail = [{"short_id": "s0", "topic": "large_language_models"}]
    # Day 1 has 1 slot; a matching short is available -> needs_render False.
    slots = scheduling.consume_calendar_for_day(1, available=avail)
    assert slots[0].persona_source == "stylized"
    assert slots[0].needs_render is False
    d = slots[0].to_dict()
    assert d["persona_source"] == "stylized"
    assert d["needs_render"] is False


def test_persona_source_env_override(monkeypatch):
    monkeypatch.setenv("PERSONA_SOURCE", "photo")
    slots = scheduling.consume_calendar_for_day(1, available=[])
    assert slots[0].persona_source == "photo"
    # No available short -> needs_render True.
    assert slots[0].needs_render is True


def test_persona_source_invalid_env_falls_back(monkeypatch):
    monkeypatch.setenv("PERSONA_SOURCE", "bogus")
    slots = scheduling.consume_calendar_for_day(1, available=[])
    assert slots[0].persona_source == "stylized"


def test_persona_source_explicit_arg_wins(monkeypatch):
    monkeypatch.setenv("PERSONA_SOURCE", "photo")
    plans = scheduling.build_calendar(days=1, available=[], persona_source="video")
    slot = plans[0].slots[0]
    assert slot.persona_source == "video"


def test_calendar_to_json_needs_render_and_persona(monkeypatch):
    monkeypatch.delenv("PERSONA_SOURCE", raising=False)
    plans = scheduling.build_calendar(days=7, available=[])
    out = scheduling.calendar_to_json(plans)
    # Every slot lacks a rendered short -> all need render.
    assert out["needs_render"] == out["total_slots"]
    for dp in out["days"]:
        for s in dp["slots"]:
            assert s["persona_source"] == "stylized"
            assert s["needs_render"] is True


def test_schedule_apply_enqueues_rendered_short(capsys, monkeypatch, tmp_path):
    """--apply must enqueue a short that is actually rendered on disk.

    Regression: the schedule action built its calendar with no ``available``
    shorts, so every slot was NEEDS_RENDER and --apply enqueued 0 items while
    still exiting 0 (silent fail-open).
    """
    out_root = tmp_path / "rendered"
    mp4 = _rendered_short(out_root)
    queue = tmp_path / "queue.jsonl"
    monkeypatch.setattr(uploader, "QUEUE_PATH", queue)
    monkeypatch.setattr(publish, "QUEUE_PATH", queue)

    args = build_parser().parse_args([
        "publish", "schedule", "--days", "1", "--apply",
        "--out-root", str(out_root),
    ])
    assert args.func(args) == 0
    out = capsys.readouterr().out
    assert "READY" in out
    assert "1 enqueued" in out
    rows = [json.loads(line) for line in queue.read_text().splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["key"] == "short_a"
    assert rows[0]["platform"] == "youtube_shorts"
    assert rows[0]["out_path"] == str(mp4)


def test_schedule_reports_missing_rendered_file_without_enqueue(capsys, monkeypatch, tmp_path):
    """A slot backed by a vanished render must not be labelled READY or queued."""
    out_root = tmp_path / "rendered"
    _rendered_short(out_root)
    queue = tmp_path / "queue.jsonl"
    monkeypatch.setattr(uploader, "QUEUE_PATH", queue)
    monkeypatch.setattr(publish, "QUEUE_PATH", queue)
    monkeypatch.setattr(
        publish, "scan_rendered",
        lambda root: [publish.Publishable(
            key="short_a", topic="large_language_models", title="short a",
            out_path=str(tmp_path / "gone.mp4"))],
    )

    args = build_parser().parse_args([
        "publish", "schedule", "--days", "1", "--apply",
        "--out-root", str(out_root),
    ])
    assert args.func(args) == 0
    out = capsys.readouterr().out
    assert "MISSING_FILE" in out
    assert "0 enqueued" in out
    assert not queue.exists()
