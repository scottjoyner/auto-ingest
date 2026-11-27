"""Fatal evidence must close the source-release gate, not just the state machine.

The state machine blocks a campaign with ``errors.fatal > 0``, but the release
gate used to read only ``errors.unresolved``. An imported evidence document can
therefore carry a fatal error and still be declared releasable. These tests use
only in-memory objects built by ``custody_helpers``: no bundle, mount, card,
network, clock or live state.
"""
from __future__ import annotations

from dataclasses import replace

from custody_helpers import fully_copied_campaign, strict_policy

from auto_ingest.custody import CampaignState, derive_state, evaluate_release


def _with_errors(ev, **changes):
    return replace(ev, errors=replace(ev.errors, **changes))


def _codes(decision):
    return {b.code for b in decision.blockers}


def test_fatal_errors_alone_close_the_release_gate():
    camp, ev = fully_copied_campaign()
    assert evaluate_release(camp, ev, strict_policy()).allowed is True  # control

    fatal = _with_errors(ev, fatal=2)
    decision = evaluate_release(camp, fatal, strict_policy())
    assert decision.allowed is False
    assert "fatal_errors_present" in _codes(decision)
    # The gate must agree with the state machine about what "fatal" means.
    assert derive_state(camp, fatal, strict_policy()).state is CampaignState.BLOCKED


def test_fatal_and_unresolved_errors_are_both_reported():
    camp, ev = fully_copied_campaign(unresolved=1)
    decision = evaluate_release(camp, _with_errors(ev, fatal=1), strict_policy())
    assert {"fatal_errors_present", "unresolved_errors"} <= _codes(decision)


def test_zero_fatal_errors_leaves_a_clean_campaign_releasable():
    camp, ev = fully_copied_campaign()
    decision = evaluate_release(camp, _with_errors(ev, fatal=0), strict_policy())
    assert decision.allowed is True
    assert "fatal_errors_present" not in _codes(decision)
