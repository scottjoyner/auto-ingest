"""Round 26 — pin the posting-calendar planner's purity contract.

``auto_ingest.shorts.scheduling.build_calendar`` is described in its own
docstring as a *pure planning helper*: given a strategy schedule and the list
of rendered shorts currently on disk, it emits a day-by-day posting plan. It
is the function that drives the (future, opt-in) upload work, so callers are
entitled to treat it as a pure function of ``(schedule, available,
persona_source)``: asking for the same plan twice must give the same answer,
and asking for a plan must not consume the caller's inventory.

The existing suites (``tests/test_publish.py``, ``tests/test_schedule_cli.py``)
pin the warm-up ramp, topic rotation, ``needs_render`` stamping and the CLI
wrapper, but none pins purity: input immutability, repeat-call stability, or
the no-double-booking / scarcity consequences of the round-robin cursor. This
module pins exactly that, with a fully synthetic schedule and in-memory dicts
(no schedule JSON, env, filesystem, Neo4j, network, credential or model).
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List

from auto_ingest.shorts import scheduling

# Synthetic strategy schedule — the exact shape build_calendar consumes. Two
# platforms, one time window each, one slot per platform per day. Kept local so
# the tests never read docs/publish_schedule.json or depend on PERSONA_SOURCE.
SCHEDULE: Dict[str, Any] = {
    "platforms": ["youtube_shorts", "tiktok"],
    "cadence": {
        "windows_local": {
            "youtube_shorts": ["12:00"],
            "tiktok": ["18:00"],
        }
    },
    "warmup_days": {
        "1-3": {"posts_per_day_total": 2, "platforms": ["youtube_shorts", "tiktok"]},
        "4-7": {"posts_per_day_total": 2, "platforms": ["youtube_shorts", "tiktok"]},
        "8-14": {"posts_per_day_total": 2, "platforms": ["youtube_shorts", "tiktok"]},
        "15+": {"posts_per_day_total": 2, "platforms": ["youtube_shorts", "tiktok"]},
    },
    "topic_rotation_7day": [
        "lead0", "lead1", "lead2", "lead3", "lead4", "lead5", "lead6",
    ],
    "filler_topics": ["filler0", "filler1"],
}


def _avail(n: int, topic: str = "match0") -> List[Dict[str, str]]:
    """Build ``n`` distinct available-short dicts, all of one topic."""
    return [{"short_id": f"short-{i}", "topic": topic} for i in range(n)]


def _build(days: int, available: Any = None, persona: str = "stylized"):
    """Call build_calendar with an explicit schedule and persona (no env)."""
    return scheduling.build_calendar(
        days=days,
        schedule=SCHEDULE,
        available=available,
        persona_source=persona,
    )


def _flat(plans):
    return [slot for day in plans for slot in day.slots]


def test_available_list_and_its_dicts_are_not_mutated():
    """Planning must not consume or rewrite the caller's inventory."""
    a = _avail(5)
    before = copy.deepcopy(a)
    ids_before = [id(row) for row in a]

    _build(days=4, available=a)

    assert a == before
    # The caller's list still holds the very same row objects, in order —
    # build_calendar works on its own shallow copy.
    assert [id(row) for row in a] == ids_before


def test_repeated_calls_sharing_one_available_list_are_identical():
    """The scheduler asks for 'today' more than once; the answer must not drift."""
    a = _avail(4)
    first = scheduling.calendar_to_json(_build(days=4, available=a))
    second = scheduling.calendar_to_json(_build(days=4, available=a))
    assert first == second


def test_equal_but_distinct_inputs_produce_identical_output():
    """Output is a function of input *values*, not of object identity."""
    left = scheduling.calendar_to_json(_build(days=4, available=_avail(4)))
    right = scheduling.calendar_to_json(_build(days=4, available=_avail(4)))
    assert left == right


def test_none_and_empty_available_are_equivalent():
    assert scheduling.calendar_to_json(_build(days=3, available=None)) == \
        scheduling.calendar_to_json(_build(days=3, available=[]))


def test_consume_calendar_for_day_is_stable_across_repeated_calls():
    """The public per-day wrapper inherits the planner's purity."""
    a = _avail(3)
    first = scheduling.consume_calendar_for_day(
        1, available=a, schedule=SCHEDULE, persona_source="stylized")
    second = scheduling.consume_calendar_for_day(
        1, available=a, schedule=SCHEDULE, persona_source="stylized")
    assert [s.to_dict() for s in first] == [s.to_dict() for s in second]


def test_persona_source_is_applied_to_every_slot():
    slots = _flat(_build(days=3, available=_avail(3), persona="video"))
    assert slots
    assert all(s.persona_source == "video" for s in slots)


def test_needs_render_and_note_stay_consistent():
    slots = _flat(_build(days=3, available=_avail(2)))
    assert slots
    for s in slots:
        assert s.needs_render == (s.short_id is None)
        assert (s.note == "NEEDS_RENDER") is s.needs_render


def test_a_rendered_short_is_never_booked_twice():
    """The round-robin cursor pops each available short at most once."""
    slots = _flat(_build(days=4, available=_avail(4)))
    used = [s.short_id for s in slots if s.short_id is not None]
    assert len(used) == 4
    assert len(set(used)) == 4


def test_scarcity_fills_only_one_slot_per_available_short():
    slots = _flat(_build(days=2, available=_avail(1)))
    filled = [s for s in slots if s.short_id is not None]
    assert len(filled) == 1
    assert (filled[0].day, filled[0].platform) == (1, "youtube_shorts")
    assert all(s.needs_render for s in slots if s is not filled[0])


def test_no_matching_topic_falls_back_without_leaving_a_gap():
    """When no short matches the lead topic, any short fills the slot."""
    a = [
        {"short_id": "wrong-0", "topic": "unrelated"},
        {"short_id": "wrong-1", "topic": "unrelated"},
    ]
    slots = _flat(_build(days=2, available=a))
    assert [s.short_id for s in slots[:2]] == ["wrong-0", "wrong-1"]
    assert all(s.short_id is not None for s in slots[:2])
    # With the two shorts spent, day 2 degrades to placeholders.
    assert all(s.needs_render for s in slots[2:])
