"""Regression: ``stable_id`` must not collapse distinct part tuples.

``auto_ingest.ingest.discovery.stable_id`` mints IDs for transcript entities that
have no usable name (utterances, segments, speakers, locations). Its callers pass
an *ordered tuple* of parts; distinct tuples must produce distinct IDs, or a
Neo4j MERGE silently fuses two different entities.

The original implementation concatenated ``part + "|"`` without escaping the
delimiter, so ``("utt", "A|B", "1.000", "2.000")`` and
``("utt", "A", "B|1.000", "2.000")`` produced the same MD5 input and the same ID.

Pure, hermetic tests: stdlib imports only (the discovery module is the shared
dependency-free naming contract), no fixtures, DB, network or credentials.
"""
from __future__ import annotations

import pytest

from auto_ingest.ingest.discovery import stable_id

# (left parts, right parts) -- both distinct tuples, both with the same part count.
COLLIDING_TUPLES = [
    # The 4-part shape used by transcripts.py for utterance IDs.
    (("utt", "A|B", "1.000", "2.000"), ("utt", "A", "B|1.000", "2.000")),
    # The 2-part shape used by transcripts.py for speaker IDs.
    (("a|b", "c"), ("a", "b|c")),
    # Leading separator.
    (("", "a"), ("|a",)),
    # Trailing separator.
    (("a", ""), ("a|",)),
]


@pytest.mark.parametrize("left,right", COLLIDING_TUPLES)
def test_delimiter_in_a_part_cannot_mimic_a_part_boundary(left, right):
    assert left != right
    assert stable_id(*left) != stable_id(*right), (
        f"distinct part tuples {left!r} and {right!r} minted the same entity ID"
    )


# Digests of parts that contain neither the delimiter nor the escape character.
# The fix must leave legacy IDs byte-identical (no graph-ID migration).
LEGACY_DIGESTS = [
    (("utt", "SPEAKER_00", "1.000", "2.000"), "079c416710bd86431f05dd61d9e6c5b8"),
    (("a", "b"), "66b69f6e54562b52ab37ccecbea18e40"),
    (("loc", "key", "12", "1.5", "2.5"), "41fbc3d9e59d3044920a5fd24861d3e1"),
    (("a", "", "b"), "b04dd462b8900f3586be14a6f536508a"),
]


@pytest.mark.parametrize("parts,digest", LEGACY_DIGESTS)
def test_ordinary_parts_keep_their_legacy_digest(parts, digest):
    assert stable_id(*parts) == digest


def test_ids_are_stable_and_distinct_tuples_stay_distinct_without_delimiters():
    assert stable_id("utt", "S1", "1", "2") == stable_id("utt", "S1", "1", "2")
    assert stable_id("utt", "S1", "1", "2") != stable_id("utt", "S2", "1", "2")
    # An empty part is still a distinct tuple from a missing part.
    assert stable_id("a", "", "b") != stable_id("a", "b")
