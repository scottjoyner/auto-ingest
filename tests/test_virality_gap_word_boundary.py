"""Gap-word detection must match words, not substrings of unrelated words.

Regression for the vacuous-condition defect in
``auto_ingest.shorts.virality._has_gap``: a hook such as "A mission worth
finishing" used to earn the full curiosity-gap bonus because the three letters
"miss" appear inside "mission". These tests are pure (no Neo4j, no moviepy, no
network) and use only in-memory model objects.
"""
from __future__ import annotations

import pytest

from auto_ingest.shorts import virality
from auto_ingest.shorts.models import Cue, PlannedShort

# Ordinary words that merely CONTAIN a gap word as a substring. None of these
# is a curiosity gap, so _has_gap must return False for every one.
_SUBSTRING_FALSE_POSITIVES = [
    "A mission worth finishing",          # "miss" inside "mission"
    "I believe the numbers",              # "lie" inside "believe"
    "The client called today",            # "lie" inside "client"
    "Reality is stranger than fiction",   # "real" inside "Reality"
    "A truthful account of events",       # "truth" inside "truthful"
    "An unchanged result was reported",   # "changed" inside "unchanged"
    "The commission released its data",   # "miss" inside "commission"
]


# Genuine gap words, phrases and deliberate stems/inflections. These must keep
# matching after the fix (regression controls for the real contract).
_REAL_GAPS = [
    "You were wrong about this",          # exact gap word
    "Why does it work",                   # exact gap word
    "Nobody expects this",                # exact gap word
    "What if we tried again",             # multi-word gap phrase
    "The mistakes were found",            # inflection of "mistake"
    "Debunked at last",                   # inflection of "debunk"
    "Missing pieces of the puzzle",       # inflection of "miss"
    "Mystery solved in one step",         # deliberate "myster" stem
    "Mysterious signals in the data",     # deliberate "myster" stem
]


@pytest.mark.parametrize("text", _SUBSTRING_FALSE_POSITIVES)
def test_gap_word_inside_longer_word_is_not_a_gap(text):
    assert virality._has_gap(text) is False


@pytest.mark.parametrize("text", _REAL_GAPS)
def test_real_gap_words_still_match(text):
    assert virality._has_gap(text) is True


def _short(hook_text: str) -> PlannedShort:
    return PlannedShort(
        id="x",
        brief_topic="t",
        title="T",
        cues=[Cue(0.4, 3.4, hook_text, kind="hook")],
    )


def test_score_short_does_not_award_gap_bonus_for_substring():
    """A non-gap hook must not inherit the +30 hook / +40 CTR gap bonuses."""
    fake = virality.score_short(_short("A mission worth finishing"))
    real = virality.score_short(_short("You were wrong about this"))

    # Hook factor: 30 base + 30 gap + 10 short + 5 capitalised = 75 with a real
    # gap; 45 without one. CTR: 40 base + 40 gap = 80, or 55 for a weak hook.
    assert real.factors["hook"] == 75.0
    assert fake.factors["hook"] == 45.0
    assert real.factors["ctr"] == 80.0
    assert fake.factors["ctr"] == 55.0
    assert real.total - fake.total >= 10.0


def test_non_gap_hook_gets_the_actionable_suggestions():
    scored = virality.score_short(_short("A mission worth finishing"))
    assert any("lacks a curiosity gap" in s for s in scored.suggestions)
    assert any("CTR-weak" in s for s in scored.suggestions)
