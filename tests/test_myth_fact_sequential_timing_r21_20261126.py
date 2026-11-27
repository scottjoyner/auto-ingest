"""R21 regression: the myth/fact cue pair must be timed where it is appended.

``hook_bank.myth_fact_pair`` emits a fixed 0.4s start.  When
``planner._distribute_cues`` appended that pair after the hook/points/reveal,
the myth cue collided with the opening hook (0.4-3.2) and the fact cue collided
with the first points (3.4-6.4).  The pair now takes an explicit ``start`` and
``_distribute_cues`` schedules it after the content built so far, still inside
the runtime.

All tests are hermetic: pure functions over Brief/Cue dataclasses, no I/O, no
Neo4j, no clock, no filesystem writes.
"""
from __future__ import annotations

from auto_ingest.shorts import hook_bank
from auto_ingest.shorts.models import Brief
from auto_ingest.shorts.planner import _distribute_cues

_TOPIC = "large_language_models"
_TITLE = "Scaling limits"
_POINTS = ["point one", "point two", "point three"]


def _brief() -> Brief:
    return Brief(topic=_TOPIC, title=_TITLE, hook="hook", points=list(_POINTS))


def _cues(duration: float = 30.0, **kwargs):
    return _distribute_cues(
        _TITLE, "hook", list(_POINTS), [], duration=duration,
        brief=_brief(), **kwargs,
    )


def test_myth_fact_pair_default_timing_unchanged():
    """Standalone callers keep the original 0.4/3.2/3.4/6.4 opening timing."""
    pair = hook_bank.myth_fact_pair("claim", "correction")
    assert [c.kind for c in pair] == ["myth", "fact"]
    assert (pair[0].start, pair[0].end) == (0.4, 3.2)
    assert (pair[1].start, pair[1].end) == (3.4, 6.4)


def test_myth_fact_pair_accepts_explicit_start():
    pair = hook_bank.myth_fact_pair("claim", "correction", start=20.0)
    assert pair[0].start == 20.0
    assert pair[0].end == 22.8
    assert pair[1].start == 23.0
    assert pair[1].end == 26.0


def test_myth_fact_no_longer_collides_with_hook():
    cues = _cues(myth_fact=("bigger is always better", "diminishing returns"))
    hook = next(c for c in cues if c.kind == "hook")
    myth = next(c for c in cues if c.kind == "myth")
    fact = next(c for c in cues if c.kind == "fact")
    assert myth.start >= hook.end, "myth cue must not overlap the opening hook"
    assert fact.start >= myth.end, "fact cue must follow the myth cue"


def test_myth_fact_after_all_points():
    cues = _cues(myth_fact=("m", "f"))
    point_end = max(c.end for c in cues if c.kind == "point")
    myth = next(c for c in cues if c.kind == "myth")
    assert myth.start >= point_end, "myth cue must not overlap the point cues"


def test_myth_fact_after_reveal_when_present():
    cues = _cues(reveal_text="surprising answer", myth_fact=("m", "f"))
    reveal_end = max(c.end for c in cues if c.kind in ("question", "reveal"))
    myth = next(c for c in cues if c.kind == "myth")
    assert myth.start >= reveal_end, "myth cue must follow the curiosity reveal"


def test_myth_fact_pair_within_runtime():
    cues = _cues(reveal_text="answer", myth_fact=("m", "f"))
    pair = [c for c in cues if c.kind in ("myth", "fact")]
    assert pair and max(c.end for c in pair) <= 30.0 + 1e-9


def test_myth_fact_pair_order_in_cue_list():
    cues = _cues(myth_fact=("m", "f"))
    kinds = [c.kind for c in cues]
    mi, fi = kinds.index("myth"), kinds.index("fact")
    assert fi == mi + 1
    assert cues[mi].start <= cues[fi].start


def test_short_overfull_still_emits_pair_inside_runtime():
    cues = _cues(duration=12.0, myth_fact=("m", "f"))
    pair = [c for c in cues if c.kind in ("myth", "fact")]
    assert [c.kind for c in pair] == ["myth", "fact"]
    assert all(0 <= c.start < c.end for c in pair)
    assert max(c.end for c in pair) <= 12.0 + 1e-9


def test_no_pair_when_not_requested():
    assert not [c for c in _cues() if c.kind in ("myth", "fact")]


def test_output_is_deterministic():
    def sig(cues):
        return [(c.start, c.end, c.text, c.kind) for c in cues]
    assert sig(_cues(myth_fact=("m", "f"))) == sig(_cues(myth_fact=("m", "f")))
