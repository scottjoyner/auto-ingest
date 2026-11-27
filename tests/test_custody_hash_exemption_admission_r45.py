"""Round-45 hermetic contract: custody hash-exemption admission.

`auto_ingest.custody.policy.CustodyPolicy` declares the only knobs that may
relax the fail-closed release gate. The hash-exemption helpers below are the
pure admission decisions used by the state machine and the release gate:

* ``hash_exemptions_allowed`` -- is this policy permitted to honour any
  exemption at all? Requires BOTH the allow flag and a non-empty declaration;
  an operator who flips the flag but forgets to declare a pattern admits
  nothing, and a declaration under a policy that disallows exemptions is inert.
* ``honour_exemption(pattern)`` -- an exemption present in evidence is only
  real when the policy honours it; names are exact members of the declared set.
* ``undeclared_exemptions(observed)`` -- evidence-claimed exemptions the policy
  does NOT honour, reported so the gate can surface a typo instead of silently
  shrinking the campaign.

These tests are pure: the module itself guarantees no clock/network/filesystem
reads, and every value here is an in-memory literal. No fixture, DB, service or
credential is involved.

This is a genuinely new surface: no existing test module or review report
references these three helpers (``excludes_source`` is covered separately in
``tests/test_custody_hash.py``; that is a different pattern-matching method).
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from auto_ingest.custody.policy import CustodyPolicy


def test_default_policy_honours_nothing():
    p = CustodyPolicy()
    assert p.hash_exemptions_allowed is False
    assert p.honour_exemption(".DS_Store") is False
    assert p.honour_exemption("Thumbs.db") is False
    assert p.undeclared_exemptions((".DS_Store", "Thumbs.db")) == (
        ".DS_Store",
        "Thumbs.db",
    )


def test_allow_flag_without_a_declaration_is_inert():
    p = CustodyPolicy(allow_hash_exemptions=True)
    assert p.hash_exemptions_allowed is False
    assert p.honour_exemption(".DS_Store") is False


def test_declaration_alone_enables_admission():
    p = CustodyPolicy(declared_hash_exemptions=(".DS_Store",))
    assert p.allow_hash_exemptions is True
    assert p.hash_exemptions_allowed is True
    assert p.honour_exemption(".DS_Store") is True


def test_declared_patterns_admitted_but_unlisted_names_rejected():
    p = CustodyPolicy(
        allow_hash_exemptions=True,
        declared_hash_exemptions=(".DS_Store", "Thumbs.db"),
    )
    assert p.hash_exemptions_allowed is True
    assert p.honour_exemption(".DS_Store") is True
    assert p.honour_exemption("Thumbs.db") is True
    assert p.honour_exemption("desktop.ini") is False
    assert p.honour_exemption("") is False


def test_disallow_flag_vetoes_even_declared_patterns():
    p = CustodyPolicy(
        allow_hash_exemptions=False,
        declared_hash_exemptions=(".DS_Store",),
    )
    assert p.hash_exemptions_allowed is False
    assert p.honour_exemption(".DS_Store") is False
    assert p.undeclared_exemptions((".DS_Store",)) == (".DS_Store",)


def test_undeclared_exemptions_reports_only_unhonoured_claim_in_order():
    p = CustodyPolicy(declared_hash_exemptions=(".DS_Store",))
    claimed = (".DS_Store", "Thumbs.db", ".DS_Store", "z")
    assert p.undeclared_exemptions(claimed) == ("Thumbs.db", "z")


def test_undeclared_exemptions_empty_when_all_claims_honoured():
    p = CustodyPolicy(declared_hash_exemptions=("a", "b"))
    assert p.undeclared_exemptions(()) == ()
    assert p.undeclared_exemptions(("a", "b", "a")) == ()


def test_from_dict_coerces_string_and_list_declarations_to_tuple():
    single = CustodyPolicy.from_dict({"declared_hash_exemptions": ".DS_Store"})
    assert single.declared_hash_exemptions == (".DS_Store",)
    assert single.hash_exemptions_allowed is True

    multi = CustodyPolicy.from_dict({"declared_hash_exemptions": ["a", "b", "a"]})
    assert multi.declared_hash_exemptions == ("a", "b", "a")
    assert multi.to_dict()["declared_hash_exemptions"] == ["a", "b", "a"]


def test_from_dict_ignores_null_unknown_keys_and_defaults_to_empty():
    assert CustodyPolicy.from_dict(None) == CustodyPolicy()
    p = CustodyPolicy.from_dict({"declared_hash_exemptions": None, "not_a_real_knob": "x"})
    assert p == CustodyPolicy()


def test_policy_is_frozen():
    p = CustodyPolicy(declared_hash_exemptions=("a",))
    with pytest.raises(FrozenInstanceError):
        p.allow_hash_exemptions = False
    with pytest.raises(FrozenInstanceError):
        p.declared_hash_exemptions = ("b",)
