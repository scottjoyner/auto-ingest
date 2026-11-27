"""Hermetic contract tests for community-summary LLM output cleanup (r39).

``community_summaries.clean_llm`` is the last gate before an LLM-generated
topic description is written back to ``Community.summary`` in the knowledge
graph.  Reasoning models emit their chain of thought inside
``<think>...</think>``; if that markup is not stripped, private reasoning is
persisted as a human-readable graph label.  Despite being a pure string
function with a security-adjacent purpose, this surface had no test coverage
before round 39.

Every case below is a pure in-process string/constant check: no Neo4j, no LM
Studio, no network, no media, and no filesystem fixtures.
"""

from __future__ import annotations

from community_summaries import DEFAULT_TEXT_PROP, THINK_RE, clean_llm


def test_single_think_block_is_removed_and_answer_trimmed() -> None:
    raw = "<think>internal reasoning</think>  Topic summary.  "
    assert clean_llm(raw) == "Topic summary."


def test_multiple_think_blocks_are_all_removed() -> None:
    raw = "<think>one</think>kept middle<think>two</think>final"
    assert clean_llm(raw) == "kept middlefinal"


def test_think_block_spans_newlines() -> None:
    raw = "<think>line one\nline two\nline three</think>\nAnswer"
    assert clean_llm(raw) == "Answer"


def test_empty_think_block_leaves_the_answer() -> None:
    assert clean_llm("<think></think>  Answer  ") == "Answer"


def test_plain_text_is_only_stripped() -> None:
    assert clean_llm("  plain topic description \n") == "plain topic description"


def test_empty_and_whitespace_inputs_return_empty_string() -> None:
    assert clean_llm("") == ""
    assert clean_llm("   \n\t ") == ""


def test_think_re_is_non_greedy_and_dotall() -> None:
    # A greedy expression would also swallow the kept text between the blocks.
    raw = "<think>a</think>KEEP<think>b</think>"
    assert THINK_RE.sub("", raw) == "KEEP"
    assert THINK_RE.sub("", "<think>a\nb</think>TAIL") == "TAIL"


def test_unclosed_think_tag_is_preserved() -> None:
    # Characterization of current behaviour: only a closed pair is stripped.
    raw = "<think>unfinished reasoning and the answer"
    assert clean_llm(raw) == raw.strip()


def test_think_tags_are_case_sensitive() -> None:
    # Characterization of current behaviour: uppercase markers are not treated
    # as reasoning (a possible future hardening point, deliberately not changed
    # by this test-only round).
    raw = "<THINK>upper reasoning</THINK>Answer"
    assert clean_llm(raw) == raw


def test_default_text_prop_covers_core_text_labels() -> None:
    for label in ("Chunk", "Summary", "Segment", "Utterance", "Transcription"):
        assert DEFAULT_TEXT_PROP[label] == "text"


def test_default_text_prop_entries_are_nonempty_strings() -> None:
    assert DEFAULT_TEXT_PROP
    for label, prop in DEFAULT_TEXT_PROP.items():
        assert isinstance(label, str) and label, label
        assert isinstance(prop, str) and prop.strip(), (label, prop)
