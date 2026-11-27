"""Hermetic regression: series-outro label for untitled briefs (r42).

Pure dataclass inputs only -- no filesystem, DB, network, clock or media.
"""

from auto_ingest.shorts import hook_bank
from auto_ingest.shorts.models import Brief


def _brief(title="", topic="", hook="h"):
    return Brief(topic=topic, title=title, hook=hook, points=["p"])


def test_final_part_outro_uses_series_fallback_when_untitled():
    cue = hook_bank.build_series_outro(_brief(), total_parts=3, short_index=2, duration=12.0)
    assert cue.text == "Series complete — Series"


def test_final_part_outro_never_renders_none():
    brief = _brief()
    brief.title = None
    brief.topic = None
    cue = hook_bank.build_series_outro(brief, total_parts=2, short_index=1, duration=12.0)
    assert cue.text == "Series complete — Series"
    assert "None" not in cue.text


def test_final_part_outro_with_title_is_unchanged():
    cue = hook_bank.build_series_outro(
        _brief(title="Cool Topic"), total_parts=3, short_index=2, duration=12.0
    )
    assert cue.text == "Series complete — Cool Topic"


def test_mid_series_outro_is_unchanged():
    cue = hook_bank.build_series_outro(
        _brief(title="Cool Topic"), total_parts=3, short_index=0, duration=12.0
    )
    assert cue.text == "Next: Part 2 of 3"
    assert cue.kind == "series"


def test_intro_and_final_outro_labels_agree():
    brief = _brief()
    intro = hook_bank.build_series_intro(brief, total_parts=2, short_index=0)
    outro = hook_bank.build_series_outro(brief, total_parts=2, short_index=1, duration=12.0)
    assert "Series" in intro.text and "Part 1 of 2" in intro.text
    assert "Series" in outro.text and "None" not in outro.text
