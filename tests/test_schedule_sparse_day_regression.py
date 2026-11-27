"""Regression: consume_calendar_for_day must look a day up by its day number,
not by its position in the (possibly sparse) list of emitted DayPlans.

build_calendar() only appends a DayPlan when that day actually has slots. A
schedule can legitimately declare posting-free days (e.g. a warm-up bucket with
no active platforms). When an earlier day is skipped, ``plans[day - 1]`` no
longer refers to ``day`` -- it points at a later day, or runs off the end.

Pure and hermetic: a hand-built schedule dict, no filesystem/network/credentials.
"""

from __future__ import annotations

from auto_ingest.shorts import scheduling


def _sparse_schedule(empty_buckets: set[str]) -> dict:
    """A minimal schedule where the named warm-up buckets post nothing."""
    def bucket(posts: int, platforms: list[str]) -> dict:
        return {"posts_per_day_total": posts, "platforms": platforms}

    return {
        "platforms": ["youtube_shorts", "tiktok"],
        "cadence": {
            "windows_local": {
                "youtube_shorts": ["12:00"],
                "tiktok": ["11:00"],
            }
        },
        "warmup_days": {
            name: bucket(0 if name in empty_buckets else 1,
                         [] if name in empty_buckets else ["youtube_shorts"])
            for name in ("1-3", "4-7", "8-14", "15+")
        },
        "topic_rotation_7day": ["t0"],
        "filler_topics": ["f0"],
    }


def _available():
    return [{"short_id": f"s{i}", "topic": "t0"} for i in range(10)]


def test_requested_day_after_empty_earlier_days_returns_its_slots():
    # Days 1-3 post nothing, so build_calendar emits only the day-4 plan. The
    # caller asking for day 4 must get that plan, not plans[3] (which is OOB).
    schedule = _sparse_schedule({"1-3"})
    avail = _available()
    emitted = scheduling.build_calendar(days=4, schedule=schedule, available=avail)
    assert [p.day for p in emitted] == [4]  # list is sparse: no day 1/2/3 entries

    slots = scheduling.consume_calendar_for_day(
        4, available=avail, schedule=schedule)

    assert len(slots) == 1
    assert slots[0].day == 4
    assert slots[0].platform == "youtube_shorts"


def test_requested_day_after_a_middle_gap_returns_its_slots():
    # Days 1-3 post, days 4-7 are empty, day 8 posts. plans has four entries
    # ([day1, day2, day3, day8]); plans[day-1] = plans[7] used to raise
    # IndexError even though day 8 genuinely has a slot.
    schedule = _sparse_schedule({"4-7"})
    avail = _available()

    slots = scheduling.consume_calendar_for_day(
        8, available=avail, schedule=schedule)

    assert len(slots) == 1
    assert slots[0].day == 8
    assert slots[0].platform == "youtube_shorts"
