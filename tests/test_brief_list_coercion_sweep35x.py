"""Brief.from_dict must not coerce a string into a list of characters (sweep35x).

Hermetic: pure dataclass/dict inputs only; no Neo4j, network, clock, model or
filesystem fixtures are used.
"""
from __future__ import annotations

import pytest

from auto_ingest.shorts.models import Brief, Plan


POINT = "One coherent research point about translucent concrete"
TAG = "materials-science"


def test_string_points_are_rejected_not_split_into_characters():
    with pytest.raises(ValueError, match="points"):
        Brief.from_dict({"topic": "t", "points": POINT})


def test_string_tags_are_rejected_not_split_into_characters():
    with pytest.raises(ValueError, match="tags"):
        Brief.from_dict({"topic": "t", "tags": TAG})


def test_non_string_point_element_is_rejected():
    with pytest.raises(ValueError, match="points"):
        Brief.from_dict({"topic": "t", "points": [1, 2]})


def test_plan_from_dict_rejects_string_points_in_brief():
    with pytest.raises(ValueError, match="points"):
        Plan.from_dict({"topic": "t", "brief": {"topic": "t", "points": POINT}})


def test_list_fields_roundtrip_unchanged():
    brief = Brief.from_dict(
        {"topic": "t", "points": [POINT, "second"], "tags": [TAG]}
    )
    assert brief.points == [POINT, "second"]
    assert brief.tags == [TAG]
    assert Brief.from_dict(brief.to_dict()).points == [POINT, "second"]


def test_missing_and_empty_list_fields_default_empty():
    brief = Brief.from_dict({"topic": "t"})
    assert brief.points == []
    assert brief.tags == []
