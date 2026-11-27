"""R22: a Brief built from discussion clips must tag only clips it actually used.

Pure/offline: dataclasses + pure helpers (curator.brief_from_discussions and
hook_bank.topic_type_of/hook_for). No filesystem, DB, Neo4j, LLM, network,
clock or model access.

Regression: a clip rejected as degenerate short text (or as a duplicate, or
past the six-point trim) used to leak its ``concept`` into ``brief.tags``.
Because ``brief_from_discussions`` derives ``topic_type`` from those tags, a
single dropped "math." clip tagged ``debate`` could flip a discussion brief to
the debate hook template even though no surviving utterance supports it.
"""
from __future__ import annotations

import pytest

from auto_ingest.shorts import curator
from auto_ingest.shorts.hook_bank import hook_for

DC = curator.DiscussionClip
LONG = "I think large language models will change everything over the next decade"


def _brief(clips):
    return curator.brief_from_discussions("large_language_models", clips)


# --- fail-before / pass-after -------------------------------------------------

def test_degenerate_short_clip_concept_does_not_leak_into_tags():
    clips = [
        DC("u1", LONG, "k1", "utterance", 0.91),
        DC("u2", "math.", "k2", "debate", 0.99),  # rejected: len < 20
    ]
    brief = _brief(clips)
    assert brief.tags == ["utterance"]
    assert "debate" not in brief.tags


def test_derived_topic_type_not_flipped_by_dropped_clip():
    clips = [
        DC("u1", LONG, "k1", "utterance", 0.91),
        DC("u2", "math.", "k2", "debate", 0.99),
    ]
    brief = _brief(clips)
    assert brief.topic_type == "utterance"


def test_hook_template_follows_surviving_utterance_not_dropped_concept():
    clips = [
        DC("u1", LONG, "k1", "utterance", 0.91),
        DC("u2", "math.", "k2", "debate", 0.99),
    ]
    brief = _brief(clips)
    assert hook_for(brief, variant=0).startswith("In my own words:")


def test_duplicate_clip_foreign_concept_does_not_leak_into_tags():
    clips = [
        DC("u1", LONG, "k1", "utterance", 0.91),
        DC("u1", LONG, "k1x", "paper", 0.91),  # duplicate utterance_id + text
    ]
    brief = _brief(clips)
    assert brief.points == [LONG]
    assert brief.tags == ["utterance"]


def test_concepts_beyond_six_point_trim_do_not_leak_into_tags():
    clips = [
        DC(f"u{i}", f"{LONG} point number {i}", f"k{i}", f"topic_{i}", 0.5)
        for i in range(8)
    ]
    brief = _brief(clips)
    assert len(brief.points) == 6
    assert len(brief.sources) == 6
    assert brief.tags == [f"topic_{i}" for i in range(6)]
    assert "topic_6" not in brief.tags and "topic_7" not in brief.tags


# --- invariants that must hold before and after -------------------------------

def test_all_surviving_clip_concepts_are_present_in_tags():
    clips = [
        DC("u1", LONG, "k1", "utterance", 0.91),
        DC("u2", "agents can call tools to finish a task", "k2", "agent", 0.72),
    ]
    brief = _brief(clips)
    assert brief.tags == ["agent", "utterance"]
    assert [s.kind for s in brief.sources] == ["utterance", "utterance"]
    assert [s.ref_id for s in brief.sources] == ["u1", "u2"]


def test_empty_usable_clips_still_raises():
    with pytest.raises(ValueError):
        _brief([])
    with pytest.raises(ValueError):
        _brief([DC("u1", "math.", "k1", "debate", 0.99)])


def test_tags_are_sorted_and_deterministic():
    clips = [
        DC("u2", "agents can call tools to finish a task", "k2", "zeta", 0.72),
        DC("u1", LONG, "k1", "alpha", 0.91),
    ]
    a = _brief(clips)
    b = _brief(list(reversed(clips)))
    assert a.tags == b.tags == ["alpha", "zeta"]
