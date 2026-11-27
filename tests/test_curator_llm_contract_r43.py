"""Hermetic contract tests for the curator's LLM boundary (round 43).

``_build_prompt`` turns curated Neo4j facts into a bounded, deterministic prompt
and ``_extract_json`` unwraps a JSON object from noisy LLM output. Both are pure
string/data transforms: no Neo4j, Ollama, filesystem or network is touched, so
the whole module is a genuinely new hermetic surface (prior rounds only covered
``brief_from_discussions`` tags in this module).

Every expectation below was first confirmed against the pristine base at
``f24b26465f56bab0eadce977ed67b91a410125b7`` with bounded probe scripts.
"""

from __future__ import annotations

import copy
import json

import pytest

from auto_ingest.shorts.curator import CuratedFacts, _build_prompt, _extract_json


@pytest.fixture
def base_facts():
    return CuratedFacts(topic="transformer_attention", topic_title="Transformer Attention")


# --- _build_prompt -------------------------------------------------------


def test_prompt_header_and_terminal_line(base_facts):
    lines = _build_prompt(base_facts).splitlines()
    assert lines[0] == "Topic: Transformer Attention (transformer_attention)"
    assert lines[1] == ""
    assert "Sources (indexed):" in lines
    assert lines[-2] == ""
    assert lines[-1] == "Write the JSON brief now."


def test_empty_facts_yield_minimal_prompt(base_facts):
    assert _build_prompt(base_facts) == (
        "Topic: Transformer Attention (transformer_attention)\n"
        "\n"
        "Sources (indexed):\n"
        "\n"
        "Write the JSON brief now."
    )


def test_papers_are_indexed_from_one_with_defaults(base_facts):
    base_facts.papers = [
        {"title": "Attention Is All You Need", "year": 2017},
        {"year": 0},
        {},
        {"title": None, "year": 2020},
    ]
    lines = _build_prompt(base_facts).splitlines()
    assert "  [1] Attention Is All You Need (2017)" in lines
    assert "  [2] Untitled" in lines
    assert not any(line.endswith("(0)") for line in lines)
    assert "  [3] Untitled" in lines
    # A present-but-None title is not replaced by the missing-key default.
    assert "  [4] None (2020)" in lines


def test_concepts_are_capped_at_twelve(base_facts):
    base_facts.concepts = [f"c{i}" for i in range(18)]
    lines = _build_prompt(base_facts).splitlines()
    concept_line = next(line for line in lines if line.startswith("Key concepts: "))
    assert concept_line == "Key concepts: " + ", ".join(f"c{i}" for i in range(12))
    assert "c12" not in concept_line
    assert not any(
        line.startswith("Key concepts:")
        for line in _build_prompt(CuratedFacts(topic="t", topic_title="T")).splitlines()
    )


def test_chunks_are_capped_at_four(base_facts):
    base_facts.chunks = [{"text": f"excerpt-{i}"} for i in range(6)]
    lines = _build_prompt(base_facts).splitlines()
    excerpts = [line for line in lines if line.startswith("  - ")]
    assert excerpts == [f"  - excerpt-{i}" for i in range(4)]


def test_excerpt_collapses_newlines_but_keeps_internal_spaces(base_facts):
    base_facts.chunks = [{"text": "  first\nsecond   third  "}]
    lines = _build_prompt(base_facts).splitlines()
    assert "  - first second   third" in lines


def test_excerpt_truncates_to_280_chars_without_ellipsis(base_facts):
    base_facts.chunks = [{"text": "Z" * 500}]
    excerpt = [line for line in _build_prompt(base_facts).splitlines() if line.startswith("  - ")][
        0
    ]
    assert len(excerpt) == 4 + 280
    assert excerpt[4:] == "Z" * 280


def test_null_excerpt_text_still_emits_an_empty_line(base_facts):
    base_facts.chunks = [{"text": None}]
    assert "  - " in _build_prompt(base_facts).splitlines()


def test_build_prompt_does_not_mutate_facts(base_facts):
    base_facts.papers = [{"title": "P", "year": 2020}]
    base_facts.concepts = ["a", "b"]
    base_facts.chunks = [{"text": "line\nnext"}]
    before = copy.deepcopy(base_facts)
    _build_prompt(base_facts)
    assert base_facts == before


# --- _extract_json -------------------------------------------------------


def test_plain_object_is_parsed():
    assert _extract_json('{"title": "A", "points": [1]}') == {"title": "A", "points": [1]}


def test_surrounding_prose_and_markdown_fence_are_stripped():
    raw = 'Sure! Here it is:\n```json\n{"title": "B"}\n```\nDone.'
    assert _extract_json(raw) == {"title": "B"}


def test_outermost_braces_select_nested_object():
    raw = '{"a": {"b": 1}, "c": [2, 3]}'
    assert _extract_json(raw) == {"a": {"b": 1}, "c": [2, 3]}


def test_object_inside_array_is_unwrapped():
    assert _extract_json('[{"a": 1}]') == {"a": 1}


@pytest.mark.parametrize("text", ["", "plain text", "[1, 2, 3]"])
def test_no_object_raises_value_error(text):
    with pytest.raises(ValueError, match="No JSON object in LLM output"):
        _extract_json(text)


def test_reversed_braces_yield_an_empty_slice_and_decode_error():
    # With `}` before `{`, start > end so the slice is empty: the function
    # reaches json.loads("") instead of the no-object ValueError branch.
    with pytest.raises(json.JSONDecodeError):
        _extract_json("}{")


def test_error_message_shows_only_the_first_400_chars():
    text = "x" * 1000
    with pytest.raises(ValueError) as exc:
        _extract_json(text)
    msg = str(exc.value)
    assert msg.startswith("No JSON object in LLM output:\n")
    assert "x" * 400 in msg
    assert "x" * 401 not in msg


def test_malformed_object_propagates_json_decode_error():
    with pytest.raises(json.JSONDecodeError):
        _extract_json('{"a": }')


def test_first_open_brace_is_authoritative_boundary():
    # A brace in the surrounding prose is part of the slice, so malformed
    # prose makes the extraction fail loudly rather than silently pick a later
    # object. This pins the documented first-{/last-} contract.
    with pytest.raises(json.JSONDecodeError):
        _extract_json('Use {title} then {"ok": 1}')
