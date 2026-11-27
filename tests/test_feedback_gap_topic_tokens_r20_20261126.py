"""R20: gap topics in feedback advice must be WORD matches, not substrings.

Hermetic: in-memory ``MetricsRecord`` objects and pure string predicates only.
No filesystem, DB, graph, network, clock, credential, service or model access.
"""
from __future__ import annotations

import pytest

from auto_ingest.shorts import feedback
from auto_ingest.shorts.feedback import suggest_next
from auto_ingest.shorts.metrics import MetricsRecord


def _has_gap_word(text: str) -> bool:
    """Use the production predicate; on pristine source emulate the old rule.

    This keeps the whole file importable (and behaviorally failing, not merely
    a collection ImportError) when run against the unmodified base source.
    """
    helper = getattr(feedback, "_has_gap_word", None)
    if helper is not None:
        return helper(text)
    return any(w in text.lower() for w in feedback._GAP_WORDS)


def _rec(topic: str, views: float, *, short_id: str = "s1",
         platform: str = "yt") -> MetricsRecord:
    return MetricsRecord(
        short_id=short_id, platform=platform, topic=topic,
        published_at="2026-01-01", views=views,
    )


def _gap_advice(suggestions: list[str]) -> list[str]:
    return [s for s in suggestions if s.startswith("Curiosity-gap topics")]


# --------------------------------------------------------------------------- #
# Pure predicate
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("topic", [
    "The client's view of reality",   # 'lie' in client, 'real' in reality
    "Beliefs about the world",        # 'lie' in Beliefs
    "A truthful account",             # 'truth' in truthful
    "A catchy title",                 # 'catch' in catchy
    "Everyday observations",          # 'everyone' must be a whole word
    "",
])
def test_substring_hits_are_not_gap_topics(topic):
    assert _has_gap_word(topic) is False


@pytest.mark.parametrize("topic", [
    "Why this matters",
    "THE MYTH, Explained",            # case + punctuation boundary
    "debunked claims",                # inflection
    "secrets nobody tells you",       # plural + whole word
    "a mystery worth solving",        # whole-word gap noun
])
def test_real_gap_words_are_recognized(topic):
    assert _has_gap_word(topic) is True


def test_predicate_is_deterministic():
    text = "The client's reality vs the myth"
    assert [_has_gap_word(text) for _ in range(4)] == [True] * 4


# --------------------------------------------------------------------------- #
# Human-facing advice
# --------------------------------------------------------------------------- #

def test_plain_topic_with_substrings_does_not_produce_gap_advice():
    """A high-view plain topic must not be called a curiosity-gap win."""
    rows = [
        _rec("The client's view of reality", 1000, short_id="plain-hi"),
        _rec("A quiet weekly review", 100, short_id="plain-lo"),
    ]
    advice = _gap_advice(suggest_next(rows))
    assert advice == []


def test_plain_topic_still_wins_the_topic_series_advice():
    rows = [
        _rec("The client's view of reality", 1000, short_id="plain-hi"),
        _rec("A quiet weekly review", 100, short_id="plain-lo"),
    ]
    out = suggest_next(rows)
    assert any("strongest performer" in s and "client's view" in s for s in out)


def test_genuine_gap_topic_still_produces_gap_advice():
    rows = [
        _rec("Why the myth persists", 1000, short_id="gap-hi"),
        _rec("A quiet weekly review", 100, short_id="plain-lo"),
    ]
    advice = _gap_advice(suggest_next(rows))
    assert len(advice) == 1
    assert "10.0x" in advice[0]


def test_gap_advice_absent_when_gap_topics_do_not_outperform():
    rows = [
        _rec("Why the myth persists", 100, short_id="gap-lo"),
        _rec("A quiet weekly review", 1000, short_id="plain-hi"),
    ]
    assert _gap_advice(suggest_next(rows)) == []


def test_suggestions_are_byte_identical_across_repeated_calls():
    rows = [
        _rec("The client's view of reality", 1000, short_id="a"),
        _rec("Why the myth persists", 300, short_id="b"),
        _rec("A quiet weekly review", 100, short_id="c"),
    ]
    runs = [suggest_next(rows) for _ in range(4)]
    assert all(r == runs[0] for r in runs)
    assert all(isinstance(s, str) for s in runs[0])


def test_empty_input_is_unchanged():
    assert suggest_next([]) == [
        "Not enough data yet — ingest metrics after a few posts."
    ]
