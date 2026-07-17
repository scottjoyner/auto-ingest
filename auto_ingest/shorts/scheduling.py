"""Posting calendar generator from the publish schedule + available shorts.

Pure planning helper (no network, no credentials): given the strategy in
``docs/publish_schedule.json`` and the list of rendered shorts currently on disk,
emit a concrete day-by-day, platform-by-platform posting plan with local times
and which short to post. This is what drives future upload work AFTER accounts
are created and OAuth is authorized — it does not publish anything itself.

The warm-up curve (ramp 1 -> 3 -> 6 -> 9 posts/day) and the 7-day topic
rotation are both encoded in the schedule JSON and applied here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_SCHEDULE = Path(__file__).resolve().parent.parent.parent / "docs" / "publish_schedule.json"


@dataclass
class PostSlot:
    day: int
    platform: str
    time_local: str
    topic: str
    short_id: Optional[str] = None
    note: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "day": self.day, "platform": self.platform, "time_local": self.time_local,
            "topic": self.topic, "short_id": self.short_id, "note": self.note,
        }


@dataclass
class DayPlan:
    day: int
    slots: List[PostSlot] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {"day": self.day, "slots": [s.to_dict() for s in self.slots]}


def load_schedule(path: Optional[Path] = None) -> Dict:
    p = Path(path) if path else DEFAULT_SCHEDULE
    return json.loads(p.read_text(encoding="utf-8"))


def _warmup_posts(day: int, schedule: Dict) -> Dict[str, object]:
    """Return the warm-up spec (posts/day total + which platforms) for a day."""
    w = schedule["warmup_days"]
    if day <= 3:
        return w["1-3"]
    if day <= 7:
        return w["4-7"]
    if day <= 14:
        return w["8-14"]
    return w["15+"]


def _topic_for_day(day: int, schedule: Dict) -> str:
    rot = schedule["topic_rotation_7day"]
    return rot[(day - 1) % len(rot)]


def build_calendar(days: int = 30, schedule: Optional[Dict] = None,
                   available: Optional[List[Dict]] = None) -> List[DayPlan]:
    """Build a ``days``-long posting calendar.

    ``available`` is a list of dicts like ``{"short_id": str, "topic": str}``
    representing rendered shorts ready to post. Slots are filled round-robin
    from available shorts of the day's lead topic, falling back to any
    available short, then to a placeholder (note="NEEDS_RENDER") so gaps are
    visible in the plan.
    """
    schedule = schedule or load_schedule()
    platforms = schedule["platforms"]
    windows = schedule["cadence"]["windows_local"]
    avail = list(available or [])

    plans: List[DayPlan] = []
    ring = 0  # round-robin cursor over available shorts
    for d in range(1, days + 1):
        wu = _warmup_posts(d, schedule)
        lead_topic = _topic_for_day(d, schedule)
        active_platforms = [p for p in platforms if p in wu["platforms"]]
        per_platform_target = max(1, round(wu["posts_per_day_total"] / len(active_platforms))) \
            if active_platforms else 0

        day_plan = DayPlan(day=d)
        for p in active_platforms:
            times = windows.get(p, ["12:00"])
            for i in range(per_platform_target):
                t = times[i % len(times)]
                # First slot of the day for a platform carries the lead topic;
                # any extra (3rd) slot uses a filler topic for variety.
                topic = lead_topic if i == 0 else schedule["filler_topics"][(d + i) % len(schedule["filler_topics"])]
                # pick a short: prefer matching topic, else any, else placeholder
                chosen = None
                for off in range(len(avail)):
                    idx = (ring + off) % len(avail) if avail else 0
                    cand = avail[idx] if avail else None
                    if cand is None:
                        break
                    if cand.get("topic") == topic:
                        chosen = avail.pop(idx)
                        break
                if chosen is None and avail:
                    chosen = avail.pop(0)
                slot = PostSlot(
                    day=d, platform=p, time_local=t, topic=topic,
                    short_id=chosen["short_id"] if chosen else None,
                    note="" if chosen else "NEEDS_RENDER",
                )
                ring = (ring + 1) % max(len(avail), 1) if avail else 0
                day_plan.slots.append(slot)
        if day_plan.slots:
            plans.append(day_plan)
    return plans


def consume_calendar_for_day(day: int, available: Optional[List[Dict]] = None,
                              schedule: Optional[Dict] = None) -> List[PostSlot]:
    """Return the posting slots scheduled for a given ``day`` (1-indexed).

    Pure (no network/creds): wraps :func:`build_calendar` with ``days=day``
    and returns ``plans[day-1].slots``. Useful for the scheduler to know what a
    specific day should post (e.g. "today" = day 1).
    """
    if day < 1:
        raise ValueError("day must be >= 1")
    plans = build_calendar(days=day, schedule=schedule, available=available)
    if not plans or day > plans[-1].day:
        return []
    return plans[day - 1].slots


def calendar_to_queue_slots(plans: List[DayPlan]) -> List[PostSlot]:
    """Flatten a calendar into a single list of every slot (for batch enqueue)."""
    slots: List[PostSlot] = []
    for dp in plans:
        slots.extend(dp.slots)
    return slots


def calendar_to_json(plans: List[DayPlan]) -> Dict[str, object]:
    return {
        "days": [dp.to_dict() for dp in plans],
        "total_slots": sum(len(dp.slots) for dp in plans),
        "needs_render": sum(1 for dp in plans for s in dp.slots if s.note == "NEEDS_RENDER"),
    }


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    out = calendar_to_json(build_calendar(days=n))
    print(json.dumps(out, indent=2))
