"""Hermetic pins for the ``content_os.models`` ContentObject state machine (r30).

Round 30 parent sweep, session 4.  The only prior coverage of this contract is
the two-assertion ``test_state_transition_validation`` in ``test_content_os.py``;
``git log --all -- content_os/models.py`` shows no agent/sweep branch ever
touched the module.  These tests pin the *whole* contract in memory: the
contiguous transition chain, self-transition no-op semantics, rejection
atomicity, ``force`` escape hatch, ``updated_at`` refresh, and error text.

No filesystem, service, network, DB, or live-state access: every test uses
in-memory ``ContentObject`` instances and a monkeypatched clock.
"""
from __future__ import annotations

import re

import pytest

from content_os import models
from content_os.models import ALLOWED_TRANSITIONS, STATE_ORDER, ContentObject, now_iso

FIXED_CREATED = "2026-01-01T00:00:00+00:00"
FIXED_UPDATED = "2026-01-01T00:00:05+00:00"


def make_obj(state: str = "idea_review") -> ContentObject:
    """Build an object with explicit timestamps so default factories never tick."""
    return ContentObject(
        id="obj-1",
        title="Title",
        slug="title",
        state=state,
        created_at=FIXED_CREATED,
        updated_at=FIXED_UPDATED,
    )


@pytest.fixture
def clock(monkeypatch):
    """Deterministic, unbounded replacement for ``models.now_iso``."""
    counter = {"n": 10}

    def fake_now():
        n = counter["n"]
        counter["n"] += 1
        return f"2026-02-01T00:00:{n:02d}+00:00"

    monkeypatch.setattr(models, "now_iso", fake_now)


def test_allowed_transitions_is_contiguous_chain():
    assert len(STATE_ORDER) == 14
    assert set(ALLOWED_TRANSITIONS) == set(STATE_ORDER[:-1])
    for old, new in ALLOWED_TRANSITIONS.items():
        assert STATE_ORDER.index(new) == STATE_ORDER.index(old) + 1, (old, new)
    assert "archived" not in ALLOWED_TRANSITIONS


def test_full_happy_path_walk_from_captured_to_archived(clock):
    obj = make_obj("captured")
    seen = [obj.state]
    for expected in STATE_ORDER[1:]:
        assert obj.transition(expected) is True
        seen.append(obj.state)
    assert seen == STATE_ORDER
    assert obj.state == "archived"


def test_self_transition_is_noop_even_with_force(clock):
    obj = make_obj("idea_review")
    assert obj.transition("idea_review") is False
    assert obj.state == "idea_review"
    assert obj.updated_at == FIXED_UPDATED
    assert obj.transition("idea_review", force=True) is False
    assert obj.updated_at == FIXED_UPDATED


def test_valid_transition_refreshes_updated_at_and_only_state(clock):
    obj = make_obj("captured")
    before = obj.model_dump()
    assert obj.transition("idea_review") is True
    after = obj.model_dump()
    assert after["state"] == "idea_review"
    assert after["updated_at"] == "2026-02-01T00:00:10+00:00"
    for key in before:
        if key not in ("state", "updated_at"):
            assert after[key] == before[key], key


def test_invalid_jump_raises_and_leaves_object_unchanged(clock):
    obj = make_obj("captured")
    before = obj.model_dump()
    with pytest.raises(ValueError) as exc:
        obj.transition("verification")  # skips several states
    assert obj.model_dump() == before
    msg = str(exc.value)
    assert "captured" in msg and "verification" in msg


def test_unknown_target_rejected_and_atomic(clock):
    obj = make_obj("brief_ready")
    before = obj.model_dump()
    with pytest.raises(ValueError):
        obj.transition("does_not_exist")
    assert obj.model_dump() == before


def test_force_allows_backwards_transition_from_terminal_state(clock):
    obj = make_obj("archived")  # terminal: no expected successor
    assert obj.transition("archived") is False
    with pytest.raises(ValueError):
        obj.transition("idea_review")
    assert obj.state == "archived"
    assert obj.transition("idea_review", force=True) is True
    assert obj.state == "idea_review"
    assert obj.updated_at != FIXED_UPDATED


def test_force_allows_unknown_target_state(clock):
    obj = make_obj("captured")
    assert obj.transition("custom_state", force=True) is True
    assert obj.state == "custom_state"
    assert obj.updated_at != FIXED_UPDATED


def test_error_message_reports_expected_successor(clock):
    obj = make_obj("captured")
    with pytest.raises(ValueError) as exc:
        obj.transition("published")
    assert "expected 'idea_review'" in str(exc.value)


def test_second_transition_to_same_state_is_noop(clock):
    obj = make_obj("captured")
    assert obj.transition("idea_review") is True
    stamped = obj.updated_at
    assert obj.transition("idea_review") is False
    assert obj.updated_at == stamped


def test_transition_never_touches_created_at(clock):
    obj = make_obj("drafting")
    created = obj.created_at
    assert obj.transition("verification") is True
    assert obj.created_at == created


def test_now_iso_is_utc_seconds_precision():
    value = now_iso()
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00", value), value
    assert "." not in value
