"""Hermetic contract tests for the pure text heuristics in ``content_os.core`` (r32).

These pin the deterministic, in-memory surface that drives route selection and
draft verification: ``slugify`` / ``run_id_for``, ``classify_route``,
``avoid_slop`` (including its banned-pattern vocabulary) and ``score_draft``.

Every test calls pure functions on literal strings. There is no filesystem,
network, graph, database, service, credential or fixture I/O, and the module's
clock is monkeypatched where a timestamp is observed.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

import pytest

from content_os import core

# ---------------------------------------------------------------------------
# slugify / run_id_for
# ---------------------------------------------------------------------------


def test_slugify_normalizes_case_separators_and_whitespace():
    assert core.slugify("Hello, World!") == "hello-world"
    assert core.slugify("  Mixed_CASE  123  ") == "mixed-case-123"
    assert core.slugify("a---b___c") == "a-b-c"


def test_slugify_falls_back_to_untitled_when_no_alnum_survives():
    assert core.slugify("!!!") == "untitled"
    assert core.slugify("---") == "untitled"
    assert core.slugify("") == "untitled"


def test_slugify_drops_non_ascii_letters():
    # 'é' is not in [a-z0-9], so it is a separator, not a kept character.
    assert core.slugify("Café") == "caf"


def test_slugify_truncates_to_64_chars_then_strips_trailing_separator():
    assert core.slugify("A" * 70) == "a" * 64
    assert len(core.slugify("A" * 70)) == 64
    # Truncation can land exactly on a separator; it is stripped afterwards.
    assert core.slugify("a" * 63 + " b") == "a" * 63


def test_run_id_for_is_utc_year_month_dash_slug(monkeypatch):
    class _FixedDateTime:
        @staticmethod
        def now(tz):
            assert tz is timezone.utc
            return datetime(2026, 11, 27, 12, 34, tzinfo=timezone.utc)

    monkeypatch.setattr(core, "datetime", _FixedDateTime)
    assert core.run_id_for("Research Ideas!") == "2026-11-research-ideas"


# ---------------------------------------------------------------------------
# classify_route
# ---------------------------------------------------------------------------


def test_classify_route_research_triggers_and_exact_reason():
    for text in ("research", "EXPLORE", "ideas about X", "candidate ideas"):
        assert core.classify_route(text) == (
            "RESEARCH_IDEATE",
            "Heuristic saw research/ideation language.",
        )


def test_classify_route_rewrite_triggers_and_exact_reason():
    for text in ("rewrite this", "EXTERNAL source", "source: x", "attribution", "link: http"):
        assert core.classify_route(text) == (
            "REWRITE",
            "Heuristic saw external source or rewrite language.",
        )


def test_classify_route_repurpose_triggers_and_exact_reason():
    for text in ("repurpose", "transcript", "article", "newsletter", "old post", "convert"):
        assert core.classify_route(text) == (
            "REPURPOSE",
            "Heuristic saw user-owned source material conversion language.",
        )


def test_classify_route_defaults_to_original_with_manual_edit_hint():
    assert core.classify_route("just my own view") == (
        "ORIGINAL",
        "Heuristic defaulted to original user viewpoint. Edit content-object.yaml if wrong.",
    )


def test_classify_route_precedence_is_research_then_rewrite_then_repurpose():
    assert core.classify_route("research this rewrite of the transcript")[0] == "RESEARCH_IDEATE"
    assert core.classify_route("rewrite the transcript")[0] == "REWRITE"
    assert core.classify_route("my transcript")[0] == "REPURPOSE"


# ---------------------------------------------------------------------------
# avoid_slop / BANNED_PATTERNS
# ---------------------------------------------------------------------------


def test_banned_pattern_vocabulary_is_compilable_and_total():
    assert len(core.BANNED_PATTERNS) == 20
    for pattern in core.BANNED_PATTERNS:
        re.compile(pattern)


def test_avoid_slop_returns_empty_for_clean_text():
    assert core.avoid_slop("") == []
    assert core.avoid_slop("a perfectly ordinary sentence") == []


def test_avoid_slop_is_case_insensitive_and_returns_vocabulary_order():
    # 'groundbreaking' precedes 'literally' in BANNED_PATTERNS regardless of
    # the order in which they appear in the text.
    assert core.avoid_slop("literally GROUNDBREAKING") == ["groundbreaking", "literally"]


def test_avoid_slop_flags_em_dash_overuse_only_above_three():
    assert core.avoid_slop("a—b—c—d") == []  # three em dashes: allowed
    assert core.avoid_slop("a—b—c—d—e") == ["em dash overuse"]  # four: flagged


def test_avoid_slop_three_no_sentence_regex_boundary():
    pattern = r"\bNo [^.]+\. No [^.]+\. No [^.]+\."
    assert core.avoid_slop("No one. No two. No three.") == [pattern]
    assert core.avoid_slop("No one. No two.") == []


def test_avoid_slop_optional_the_and_either_variant_patterns():
    assert core.avoid_slop("unlock power") == ["unlock (?:the )?power"]
    assert core.avoid_slop("unlock the power") == ["unlock (?:the )?power"]
    assert core.avoid_slop("in today's digital world") == ["in today's (?:fast-paced|digital)"]
    assert core.avoid_slop("in today's fast-paced world") == ["in today's (?:fast-paced|digital)"]


def test_avoid_slop_plain_substring_pattern_is_literal():
    assert core.avoid_slop("best") == ["best"]
    assert core.avoid_slop("better") == []  # 'best' is not a substring of 'better'


# ---------------------------------------------------------------------------
# score_draft / SCORE_ITEMS
# ---------------------------------------------------------------------------


def test_score_rubric_is_six_ordered_items_with_missing_notes():
    assert len(core.SCORE_ITEMS) == 6
    rows = core.score_draft("")
    assert [r[0] for r in rows] == core.SCORE_ITEMS
    assert [r[1] for r in rows] == [0] * 6
    assert all(r[2] == "Missing or too vague." for r in rows)


def test_score_draft_all_signals_yield_max_twelve():
    text = (
        "save checklist workflow template reuse proof example screenshot number "
        "metric case framework mental model founder operator creator "
        "job-to-be-done apply copy use this without artifact diagram"
    )
    rows = core.score_draft(text)
    assert sum(r[1] for r in rows) == 12
    assert all(r[1] == 2 for r in rows)
    assert all(r[2] == "Heuristic evidence found." for r in rows)


@pytest.mark.parametrize(
    "text,expected_indices",
    [
        ("save", [0]),
        ("checklist", [0, 2]),  # present in both the future-work and takeaway sets
        ("proof", [1]),
        ("framework", [2]),
        ("founder", [3]),
        ("step", [4]),
        ("artifact", [5]),
        ("PROOF", [1]),  # matching is case-insensitive
    ],
)
def test_score_draft_single_keyword_lights_only_its_rows(text, expected_indices):
    rows = core.score_draft(text)
    lit = [i for i, r in enumerate(rows) if r[1] == 2]
    assert lit == expected_indices
    assert sum(r[1] for r in rows) == 2 * len(expected_indices)
