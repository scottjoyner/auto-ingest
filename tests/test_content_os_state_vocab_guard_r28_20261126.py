"""Hermetic regression tests for the content_os state-vocabulary guard (r28).

``ContentObject.transition(..., force=True)`` bypasses the one-step ordering
constraint, but it must still reject states outside the documented total
``State`` vocabulary. These tests are pure in-process unit tests: no network,
LLM, credentials, Neo4j, service, clock control or filesystem fixtures.
"""
from __future__ import annotations

import pytest

from content_os.models import STATE_ORDER, ContentObject


def _obj() -> ContentObject:
    return ContentObject(id="x", title="X", slug="x")


def test_force_rejects_unknown_state() -> None:
    """The pristine defect: force=True silently stored an arbitrary state."""
    obj = _obj()
    assert "bogus_state" not in STATE_ORDER
    with pytest.raises(ValueError, match="Unknown state"):
        obj.transition("bogus_state", force=True)


def test_unknown_state_fails_closed_without_mutating() -> None:
    """A rejected transition leaves state and timestamp untouched."""
    obj = _obj()
    before_state = obj.state
    before_updated = obj.updated_at
    with pytest.raises(ValueError):
        obj.transition("not-a-real-state", force=True)
    assert obj.state == before_state
    assert obj.updated_at == before_updated


def test_force_accepts_every_known_state() -> None:
    """The guard must not reject any legitimate member of the vocabulary."""
    for state in STATE_ORDER:
        start = "approved" if state != "approved" else "idea_review"
        obj = ContentObject(id="x", title="X", slug="x", state=start)
        assert state in STATE_ORDER
        assert obj.state != state  # avoid the same-state no-op
        assert obj.transition(state, force=True) is True
        assert obj.state == state


def test_force_valid_transition_still_updates_state() -> None:
    obj = _obj()
    assert obj.transition("verification", force=True) is True
    assert obj.state == "verification"


def test_non_force_valid_next_state_still_works() -> None:
    """The original (ordering) contract is unchanged."""
    obj = _obj()
    assert obj.transition("brief_ready") is True
    assert obj.state == "brief_ready"


def test_non_force_backward_transition_still_rejected() -> None:
    obj = _obj()
    obj.transition("brief_ready")
    with pytest.raises(ValueError, match="Invalid state transition"):
        obj.transition("idea_review")


def test_same_state_is_a_noop_even_when_forced() -> None:
    obj = _obj()
    before_updated = obj.updated_at
    assert obj.transition("idea_review", force=True) is False
    assert obj.state == "idea_review"
    assert obj.updated_at == before_updated
