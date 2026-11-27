"""Hermetic regression: speed-event construction must survive non-numeric mph.

``_mine_speed_events`` turns graph rows into highlight events. A Neo4j ``mph``
property is not schema-enforced: it can be null, a string, or a non-finite
float. The label was formatted with ``:.0f`` directly on the raw value, so a
single such row aborted ``plan_highlights`` before any short was planned.

All inputs here are in-memory literals and a fake driver. No Neo4j, network,
media, clock, credentials or fixture files are involved.
"""

from __future__ import annotations

import math

import pytest

from auto_ingest.shorts.planner import _highlight_cues, _mine_speed_events

UNKNOWN_LABEL = "🚀 speed unknown"


class _RowsSession:
    def __init__(self, rows):
        self._rows = rows

    def run(self, cypher, **params):
        self.cypher = cypher
        self.params = params
        return self

    def data(self):
        return [dict(row) for row in self._rows]


class _FakeDriver:
    def __init__(self, rows):
        self.rows = rows

    def session(self):
        return _RowsSession(self.rows)


def _event(mph, *, frame=300, fps=30.0, clip_key="clip-A"):
    row = {"clip_key": clip_key, "frame": frame, "fps": fps}
    if mph is not ...:  # explicit "missing key" case passes ... sentinel
        row["mph"] = mph
    return _mine_speed_events(_FakeDriver([row]), per_kind=2, limit=10)


def test_valid_float_mph_is_preserved_and_formatted():
    events = _event(92.4)
    assert len(events) == 1
    assert events[0]["mph"] == 92.4
    assert events[0]["label"] == "🚀 92 mph"
    assert events[0]["t_sec"] == pytest.approx(10.0)


def test_valid_int_mph_is_preserved_and_formatted():
    events = _event(85)
    assert events[0]["mph"] == 85
    assert events[0]["label"] == "🚀 85 mph"


@pytest.mark.parametrize("bad", [None, ..., "92", "", True, math.nan, math.inf, -math.inf])
def test_non_numeric_mph_never_aborts_event_construction(bad):
    events = _event(bad)
    assert len(events) == 1
    assert events[0]["kind"] == "speed"
    assert events[0]["mph"] is None
    assert events[0]["label"] == UNKNOWN_LABEL


def test_missing_mph_key_never_aborts_event_construction():
    events = _mine_speed_events(
        _FakeDriver([{"clip_key": "clip-Z", "frame": 60, "fps": 30.0}]),
        per_kind=2,
        limit=10,
    )
    assert events[0]["mph"] is None
    assert events[0]["label"] == UNKNOWN_LABEL


def test_one_bad_row_does_not_drop_other_rows_or_change_order():
    rows = [
        {"clip_key": "a", "mph": None, "frame": 0, "fps": 30.0},
        {"clip_key": "b", "mph": 70.0, "frame": 300, "fps": 30.0},
        {"clip_key": "c", "mph": "80", "frame": 600, "fps": 30.0},
    ]
    events = _mine_speed_events(_FakeDriver(rows), per_kind=3, limit=10)
    assert [e["clip_key"] for e in events] == ["a", "b", "c"]
    assert [e["mph"] for e in events] == [None, 70.0, None]
    assert [e["label"] for e in events] == [UNKNOWN_LABEL, "🚀 70 mph", UNKNOWN_LABEL]


def test_unknown_mph_row_keeps_clip_key_index_and_empty_text():
    event = _event(None, frame=17, clip_key="clip-X")[0]
    assert event["clip_key"] == "clip-X"
    assert event["idx"] == 17
    assert event["text"] == ""
    assert event["kind"] == "speed"


def test_zero_fps_keeps_existing_default_rate_fallback():
    # Control: fps=0 is falsy and already falls back to 30 fps; unchanged here.
    event = _event(95.0, frame=300, fps=0)[0]
    assert event["t_sec"] == pytest.approx(10.0)
    assert event["label"] == "🚀 95 mph"


def test_empty_rows_yield_no_events():
    assert _mine_speed_events(_FakeDriver([]), per_kind=2, limit=10) == []


def test_unknown_mph_label_renders_as_a_highlight_cue():
    event = _event(None)[0]
    cues = _highlight_cues(event, 20.0)
    assert len(cues) == 1
    assert cues[0].text == UNKNOWN_LABEL
    assert cues[0].kind == "speed"
