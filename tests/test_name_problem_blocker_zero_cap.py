"""Bounded-cap contract for custody planner name-problem blockers.

Pure/offline: ``name_problem_blockers`` reads only its arguments. No campaign
ledger, Neo4j, filesystem fixture or network is needed.

``policy.max_summary_entries`` is an operator-settable cap and may legitimately
be set to 0 (or a negative value from a bad config). The cap is a *maximum*,
so a zero cap must yield no blockers at all - not one blocker from the first
sample, which is what the old post-append break allowed.
"""
from __future__ import annotations

from auto_ingest.custody.planner import name_problem_blockers

_SAMPLES = tuple(
    f"case_collision: 'clip_{i:05d}.mp4' collides with 'CLIP_{i:05d}.mp4'"
    for i in range(5)
)


def test_zero_cap_returns_no_blockers() -> None:
    """A cap of zero means zero blockers, even when samples are present."""
    assert name_problem_blockers(_SAMPLES, 0) == ()


def test_negative_cap_returns_no_blockers() -> None:
    """A negative cap is still a cap, not an off-by-one invitation."""
    assert name_problem_blockers(_SAMPLES, -1) == ()


def test_positive_cap_still_works() -> None:
    """Guard against 'fixing' zero by breaking the normal bounded path."""
    assert len(name_problem_blockers(_SAMPLES, 2)) == 2
