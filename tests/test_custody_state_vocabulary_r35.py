"""Round-35 hermetic contract for the custody campaign-state vocabulary.

``auto_ingest/custody/states.py`` is the single source of truth for the SD-card
campaign state vocabulary.  The derived state machine (``machine.py``), the
planner (``planner.py``) and the store (``store.py``) consume its constants, but
no test pins the vocabulary module itself: neither the enum's exact membership
and string values, nor the three classification sets, nor the two *total*
state->phase / state->action tables.  This file pins that surface directly.

Every assertion is an in-memory operation on module-level constants.  No mount
table, campaign lock directory, graph, DB, network, backup or fixture file is
touched; the tests are safe under ``PYTHONDONTWRITEBYTECODE=1`` and a scratch
``TMPDIR``.
"""
from __future__ import annotations

import pytest

from auto_ingest.custody import states as vocab
from auto_ingest.custody.states import (
    CUSTODY_PROVEN,
    NEEDS_HUMAN,
    NEXT_SAFE_ACTION,
    RESUMABLE,
    STATE_PHASE,
    CampaignState,
)

# Expected vocabulary, transcribed from the module's documented contract. The
# comment block at the top of ``states.py`` is the authority for these values.
EXPECTED_STATES = (
    "DISCOVERED",
    "SOURCE_VERIFIED",
    "HASHING",
    "HASH_COMPLETE",
    "COPY_PENDING",
    "COPYING",
    "COPY_COMPLETE",
    "VERIFYING",
    "VERIFIED",
    "RECONCILE_REQUIRED",
    "BLOCKED",
    "SAFE_TO_RELEASE",
)
EXPECTED_PHASES = {
    "DISCOVERED": "source_inventory",
    "SOURCE_VERIFIED": "hashing",
    "HASHING": "hashing",
    "HASH_COMPLETE": "copy_planning",
    "COPY_PENDING": "copy",
    "COPYING": "copy",
    "COPY_COMPLETE": "destination_verification",
    "VERIFYING": "destination_verification",
    "VERIFIED": "operator_release_review",
    "RECONCILE_REQUIRED": "destination_reconciliation",
    "BLOCKED": "operator_intervention",
    "SAFE_TO_RELEASE": "complete",
}
EXPECTED_ACTIONS = {
    "DISCOVERED": "inventory_source",
    "SOURCE_VERIFIED": "hash_source",
    "HASHING": "hash_source",
    "HASH_COMPLETE": "plan_copy",
    "COPY_PENDING": "copy_objects",
    "COPYING": "await_copy",
    "COPY_COMPLETE": "verify_destination",
    "VERIFYING": "verify_destination",
    "VERIFIED": "review_for_release",
    "RECONCILE_REQUIRED": "reconcile_destination",
    "BLOCKED": "resolve_contradiction",
    "SAFE_TO_RELEASE": "none",
}
EXPECTED_RESUMABLE = frozenset({
    "HASHING",
    "COPY_PENDING",
    "COPYING",
    "COPY_COMPLETE",
    "VERIFYING",
    "RECONCILE_REQUIRED",
    "VERIFIED",
})
EXPECTED_PHASE_VOCAB = {
    "source_inventory",
    "hashing",
    "copy_planning",
    "copy",
    "destination_verification",
    "operator_release_review",
    "destination_reconciliation",
    "operator_intervention",
    "complete",
}
EXPECTED_ACTION_VOCAB = {
    "inventory_source",
    "hash_source",
    "plan_copy",
    "copy_objects",
    "await_copy",
    "verify_destination",
    "review_for_release",
    "reconcile_destination",
    "resolve_contradiction",
    "none",
}


def test_enum_is_exact_total_vocabulary():
    assert [s.value for s in CampaignState] == list(EXPECTED_STATES)
    assert len(CampaignState.__members__) == 12
    # __members__ preserves declaration order; equality here also proves there
    # are no aliases silently collapsing two names onto one member.
    assert list(CampaignState.__members__) == list(EXPECTED_STATES)


def test_enum_is_str_enum_with_stable_round_trip():
    assert issubclass(CampaignState, str)
    for state in CampaignState:
        assert str(state) == state.value
        assert state == state.value
        assert CampaignState(state.value) is state


def test_enum_values_are_hashable_distinct_keys():
    assert len({CampaignState(name) for name in EXPECTED_STATES}) == 12


def test_custody_proven_set_is_exact():
    assert isinstance(CUSTODY_PROVEN, frozenset)
    assert CUSTODY_PROVEN == frozenset({
        CampaignState.VERIFIED,
        CampaignState.SAFE_TO_RELEASE,
    })


def test_needs_human_set_is_exact():
    assert isinstance(NEEDS_HUMAN, frozenset)
    assert NEEDS_HUMAN == frozenset({CampaignState.BLOCKED})


def test_resumable_set_is_exact():
    assert isinstance(RESUMABLE, frozenset)
    assert {s.value for s in RESUMABLE} == EXPECTED_RESUMABLE


def test_resumable_excludes_terminal_and_pre_hash_states():
    assert {s.value for s in RESUMABLE}.isdisjoint({
        "DISCOVERED",
        "SOURCE_VERIFIED",
        "HASH_COMPLETE",
        "BLOCKED",
        "SAFE_TO_RELEASE",
    })


@pytest.mark.parametrize("name,phase", sorted(EXPECTED_PHASES.items()))
def test_state_phase_mapping(name, phase):
    assert STATE_PHASE[CampaignState(name)] == phase


@pytest.mark.parametrize("name,action", sorted(EXPECTED_ACTIONS.items()))
def test_next_safe_action_mapping(name, action):
    assert NEXT_SAFE_ACTION[CampaignState(name)] == action


def test_phase_table_is_total_with_closed_vocabulary():
    assert set(STATE_PHASE) == set(CampaignState)
    assert len(STATE_PHASE) == 12
    assert set(STATE_PHASE.values()) == EXPECTED_PHASE_VOCAB


def test_action_table_is_total_with_closed_vocabulary():
    assert set(NEXT_SAFE_ACTION) == set(CampaignState)
    assert len(NEXT_SAFE_ACTION) == 12
    assert set(NEXT_SAFE_ACTION.values()) == EXPECTED_ACTION_VOCAB


def test_both_tables_share_exactly_the_enum_keys():
    assert set(STATE_PHASE) == set(NEXT_SAFE_ACTION) == set(CampaignState)


def test_terminal_state_is_complete_with_no_next_action():
    assert STATE_PHASE[CampaignState.SAFE_TO_RELEASE] == "complete"
    assert NEXT_SAFE_ACTION[CampaignState.SAFE_TO_RELEASE] == "none"


def test_only_terminal_state_has_none_action():
    none_actions = {s for s in CampaignState if NEXT_SAFE_ACTION[s] == "none"}
    assert none_actions == {CampaignState.SAFE_TO_RELEASE}


def test_custody_proven_states_sit_in_release_phases():
    for state in CUSTODY_PROVEN:
        assert STATE_PHASE[state] in {"operator_release_review", "complete"}


def test_needs_human_state_alignment():
    assert STATE_PHASE[CampaignState.BLOCKED] == "operator_intervention"
    assert NEXT_SAFE_ACTION[CampaignState.BLOCKED] == "resolve_contradiction"
    assert CampaignState.BLOCKED not in RESUMABLE


def test_shared_phase_groups_preserve_enum_order():
    assert [s.value for s in CampaignState if STATE_PHASE[s] == "hashing"] == [
        "SOURCE_VERIFIED", "HASHING",
    ]
    assert [s.value for s in CampaignState if STATE_PHASE[s] == "copy"] == [
        "COPY_PENDING", "COPYING",
    ]
    assert [
        s.value for s in CampaignState
        if STATE_PHASE[s] == "destination_verification"
    ] == ["COPY_COMPLETE", "VERIFYING"]


def test_shared_action_groups_preserve_enum_order():
    assert [s.value for s in CampaignState if NEXT_SAFE_ACTION[s] == "hash_source"] == [
        "SOURCE_VERIFIED", "HASHING",
    ]
    assert [
        s.value for s in CampaignState
        if NEXT_SAFE_ACTION[s] == "verify_destination"
    ] == ["COPY_COMPLETE", "VERIFYING"]


def test_public_exports_are_exact():
    assert vocab.__all__ == [
        "CampaignState",
        "CUSTODY_PROVEN",
        "NEEDS_HUMAN",
        "RESUMABLE",
        "STATE_PHASE",
        "NEXT_SAFE_ACTION",
    ]
