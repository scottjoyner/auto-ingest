"""Sweep31x: build_calendar must honour the declared warm-up posts_per_day_total.

Hermetic: in-process schedule dicts, no Neo4j, no schedule file, no credentials,
no rendered media. Each slot a planner emits is a post an operator is told to make,
so a total that is not honoured is an over- or under-scheduling defect.
"""
from auto_ingest.shorts.scheduling import build_calendar

PLATFORMS = ["youtube_shorts", "tiktok", "instagram"]


def _schedule(total, platforms=PLATFORMS):
    return {
        "platforms": PLATFORMS,
        "cadence": {
            "windows_local": {p: ["12:00", "18:00", "21:00"] for p in PLATFORMS}
        },
        "topic_rotation_7day": [f"t{i}" for i in range(7)],
        "filler_topics": ["montage", "highlights", "trip_story"],
        "warmup_days": {
            "1-3": {"posts_per_day_total": total, "platforms": list(platforms)}
        },
    }


def _slot_count(total, platforms=PLATFORMS):
    return sum(len(day.slots) for day in build_calendar(days=1, schedule=_schedule(total, platforms)))


def test_divisible_total_is_unchanged_control():
    # Control: passes before and after the fix; the default 9-post warm-up day
    # must keep producing exactly nine slots.
    assert _slot_count(9) == 9


def test_single_platform_control():
    # Control: the declared total across one platform already worked.
    assert _slot_count(1, ["youtube_shorts"]) == 1


def test_total_below_platform_count_is_honoured():
    # Defect: a gentler 1-post warm-up across three platforms used to emit
    # three slots because of max(1, round(total / n)).
    assert _slot_count(1) == 1


def test_non_divisible_total_is_honoured():
    # Defect: 4 declared posts across three platforms used to emit only three.
    assert _slot_count(4) == 4


def test_zero_declared_posts_emits_no_slots():
    # Defect: a zero-post day is a "no posting today" instruction, not a
    # licence to post once per platform.
    assert _slot_count(0) == 0
