"""The declared source read-only flag must not be parsed by truthiness.

``campaign.source.read_only`` is the release gate's precondition for destroying
source bytes. A JSON bundle may carry the string ``"false"``; every non-empty
string is truthy in Python, so a naive ``bool()`` promoted an explicit "writable"
declaration into the one state that opens the gate.
"""
from __future__ import annotations

import pytest
from custody_helpers import fully_copied_campaign, strict_policy

from auto_ingest.custody.campaign import Campaign, SourceRef
from auto_ingest.custody.release import evaluate_release


def _blockers(decision):
    return {b.code for b in decision.blockers}


@pytest.mark.parametrize("raw", ["false", "off", "no", "0", " FALSE "])
def test_false_spellings_never_declare_a_read_only_source(raw):
    assert SourceRef.from_dict({"read_only": raw}).read_only is False


@pytest.mark.parametrize("raw", ["maybe", 2, -1, [], {}, object()])
def test_unparseable_values_fail_closed(raw):
    assert SourceRef.from_dict({"read_only": raw}).read_only is False


@pytest.mark.parametrize("raw,expected", [
    (True, True), (False, False), (1, True), (0, False),
    ("true", True), ("YES", True), (" on ", True), ("off", False),
])
def test_real_booleans_and_canonical_spellings_are_preserved(raw, expected):
    assert SourceRef.from_dict({"read_only": raw}).read_only is expected


def test_a_string_false_in_campaign_json_keeps_the_release_gate_closed():
    """The full gate: a bundle that says 'false' must still block deletion."""
    camp, ev = fully_copied_campaign()
    raw = camp.to_dict()
    raw["source"]["read_only"] = "false"
    parsed = Campaign.from_dict(raw)

    decision = evaluate_release(parsed, ev, strict_policy())
    assert parsed.source.read_only is False
    assert "source_not_read_only" in _blockers(decision)
    assert decision.allowed is False


def test_a_true_declaration_still_opens_the_gate():
    """Control: strict parsing must not close a genuinely read-only campaign."""
    camp, ev = fully_copied_campaign()
    decision = evaluate_release(camp, ev, strict_policy())
    assert camp.source.read_only is True
    assert decision.allowed is True
    assert _blockers(decision) == set()
