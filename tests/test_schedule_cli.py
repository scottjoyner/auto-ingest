"""
Tests for the posting-scheduler activation: scheduling.consume_calendar_for_day
plus the `publish schedule` CLI action (dry-run, report-only).

Pure: no network, no credentials, no real queue writes.
"""

from __future__ import annotations

from auto_ingest.shorts import scheduling
from auto_ingest.shorts.cli import build_parser


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

    args = build_parser().parse_args(["publish", "schedule", "--days", "3"])
    assert args.action == "schedule"
    rc = args.func(args)
    out = capsys.readouterr().out
    assert rc == 0
    assert "DRY-RUN" in out
    assert "youtube_shorts" in out
    # No queue file should be written in dry-run.
    assert not (tmp_path / "queue.jsonl").exists()
