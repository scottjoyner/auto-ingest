"""Policy knobs must be parsed, not coerced by Python truthiness.

A JSON policy override is a plain document: every scalar arrives as a string if
the operator quoted it. ``allow_hash_exemptions`` is the one relaxation knob in
the custody contract, and treating the *string* ``"false"`` as a boolean True
lets a policy that was explicitly told not to honour exemptions honour them.
That is a fail-open in the release gate, so the parse has to reject values it
does not understand instead of trusting their truthiness.
"""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from custody_helpers import fully_copied_campaign
from auto_ingest.custody import evaluate_release
from auto_ingest.custody.cli import _policy_from_file
from auto_ingest.custody.policy import CustodyPolicy

STRICT_RAW = {
    "required_scope": "all_inventory",
    "allow_hash_exemptions": "false",
    "declared_hash_exemptions": ["*.tmp"],
    "require_operator_witness": False,
    "strict_destination_scope": False,
    "require_destination_identity": True,
    "require_read_only_source": True,
    "require_mounted_destination": True,
}


def test_a_quoted_false_is_false_not_a_truthy_string():
    policy = CustodyPolicy.from_dict(STRICT_RAW)
    assert policy.allow_hash_exemptions is False
    assert policy.hash_exemptions_allowed is False


def test_canonical_boolean_spellings_are_understood_both_ways():
    for false_value in (False, "false", "False", "no", "off", "0", 0):
        policy = CustodyPolicy.from_dict({"allow_hash_exemptions": false_value})
        assert policy.allow_hash_exemptions is False, false_value
    for true_value in (True, "true", "True", "yes", "on", "1", 1):
        policy = CustodyPolicy.from_dict({"allow_hash_exemptions": true_value})
        assert policy.allow_hash_exemptions is True, true_value


def test_an_unknown_boolean_is_refused_not_guessed():
    with pytest.raises(ValueError):
        CustodyPolicy.from_dict({"allow_hash_exemptions": "perhaps"})
    with pytest.raises(ValueError):
        CustodyPolicy.from_dict({"require_read_only_source": 7})


def test_policy_file_override_cannot_reopen_the_exemption_path(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({"policy": STRICT_RAW}), encoding="utf-8")
    assert _policy_from_file(str(path)).hash_exemptions_allowed is False


def test_quoted_false_does_not_open_the_release_gate():
    """One unverified object + an exemption the policy says is disabled."""
    camp, ev = fully_copied_campaign(total_files=9, total_bytes=90)
    ev = replace(ev, hashing=replace(ev.hashing, verified_files=8, complete=True,
                                     exemptions=("*.tmp",)))
    ev = replace(ev, copy=replace(ev.copy, planned=replace(ev.copy.planned, files=9)))
    policy = CustodyPolicy.from_dict(STRICT_RAW)
    decision = evaluate_release(camp, ev, policy)
    assert decision.allowed is False
    codes = {b.code for b in decision.blockers}
    assert {"hash_exemptions_not_declared", "hash_evidence_incomplete"} <= codes


def test_boolean_round_trip_stays_boolean():
    policy = CustodyPolicy.from_dict(STRICT_RAW)
    assert CustodyPolicy.from_dict(policy.to_dict()).allow_hash_exemptions is False
