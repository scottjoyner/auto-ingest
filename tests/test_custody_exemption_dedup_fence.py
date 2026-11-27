"""A repeated exemption pattern must not count as multiple exempted objects.

`CampaignEvidence.hash_coverage_under` added one to coverage for every
occurrence of an honoured exemption in the evidence document. A malformed or
tampered document repeating ``*.tmp`` therefore inflated coverage by the number
of repeats, so the release gate could see full coverage while objects were still
unverified.
"""
from __future__ import annotations

from custody_helpers import evidence, hashing, fully_copied_campaign, strict_policy
from auto_ingest.custody import evaluate_release


def test_duplicate_exemption_patterns_count_once():
    ev = evidence(hsh=hashing(8, complete=False,
                                  exemptions=("*.tmp", "*.tmp")))
    policy = strict_policy(declared_hash_exemptions=("*.tmp",))
    assert ev.hash_coverage_under(policy) == 9
    assert policy.undeclared_exemptions(ev.hashing.exemptions) == ()


def test_distinct_exemption_patterns_still_count_separately():
    ev = evidence(hsh=hashing(7, complete=False,
                                  exemptions=("*.tmp", "*.part")))
    policy = strict_policy(
        declared_hash_exemptions=("*.tmp", "*.part"))
    assert ev.hash_coverage_under(policy) == 9


def test_duplicate_exemption_does_not_open_the_release_gate():
    camp, ev = fully_copied_campaign(total_files=10, total_bytes=100)
    from dataclasses import replace
    ev = replace(ev, hashing=replace(ev.hashing, verified_files=8, complete=True,
                                     exemptions=("*.tmp", "*.tmp")))
    ev = replace(ev, copy=replace(ev.copy,
                                  planned=replace(ev.copy.planned, files=10)))
    policy = strict_policy(declared_hash_exemptions=("*.tmp",))
    decision = evaluate_release(camp, ev, policy)
    assert decision.allowed is False
    assert "hash_evidence_incomplete" in {b.code for b in decision.blockers}
