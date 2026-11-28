"""Regression: topic_type_of must match keyword *tokens*, not substrings.

``hook_bank.topic_type_of`` decides which of five hook-template families a
brief opens with. It used substring containment, so an ordinary word that merely
*contains* a keyword hijacked the classification: "mistakes" contains "take"
and was filed as an opinion, and "vscode" contains "vs" and was filed as a
debate. A paper brief titled "Paper about mistakes" therefore rendered with the
opinion hot-take hook instead of the paper hook.

Pure and hermetic: no Neo4j, no model, no network, no filesystem writes.
"""
from __future__ import annotations

import pytest

from auto_ingest.shorts.hook_bank import topic_type_of
from auto_ingest.shorts.models import Brief


def _brief(topic: str, tags=()) -> Brief:
    return Brief(topic=topic, title=topic, hook="", tags=list(tags))


@pytest.mark.parametrize("topic,tags,expected", [
    # "take" inside "mistakes" must not trigger the opinion branch.
    ("Paper about mistakes", [], "paper"),
    ("mistakes in diffusion models", [], "concept"),
    ("concept overview", ["mistakes"], "concept"),
    # "vs" inside "vscode" must not trigger the debate branch.
    ("vscode tips", [], "concept"),
    # Real tokens still classify as before.
    ("a paper on arxiv", [], "paper"),
    ("debate: X vs Y", [], "debate"),
    ("my opinion", [], "opinion"),
    ("hot take on models", [], "opinion"),
    ("", [], "utterance"),
    ("plain topic", ["utterance"], "utterance"),
    ("concept", ["concept"], "concept"),
])
def test_topic_type_of_matches_word_tokens(topic, tags, expected):
    assert topic_type_of(_brief(topic, tags)) == expected
