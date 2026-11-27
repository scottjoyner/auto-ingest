"""Round 44: hermetic contract tests for caption speaker colour and cue-kind style.

These helpers are pure string/data transforms in ``auto_ingest.shorts.captions``.
They decide which stable RGBA colour a speaker gets and which accent/tag a cue
kind renders with, and they were previously unpinned by any test. Nothing here
renders an image, opens a file, touches a database or reaches the network.
"""

from __future__ import annotations

from auto_ingest.shorts.captions import (
    _CUE_KIND_STYLE,
    SpeakerMapEntry,
    _cue_kind_style,
    _hash_color,
    _speaker_color,
    _speaker_display_name,
)

DEFAULT_AMBER = (255, 235, 59, 230)


# --- _hash_color -----------------------------------------------------------
def test_hash_color_is_deterministic():
    """The same identity must always get the same colour, across calls."""
    assert _hash_color("G-1") == _hash_color("G-1")
    assert _hash_color("speaker-1") == _hash_color("speaker-1")


def test_hash_color_is_rgba_with_fixed_alpha():
    """Colours are PIL-ready RGBA tuples with the caption's fixed alpha."""
    for key in ("", "a", "scott", "G-1", "üñî"):
        colour = _hash_color(key)
        assert isinstance(colour, tuple)
        assert len(colour) == 4
        assert colour[3] == 230
        assert all(isinstance(v, int) and 0 <= v <= 255 for v in colour[:3])


def test_hash_color_separates_distinct_identities():
    """Different speakers should not all collapse onto one colour."""
    colours = {_hash_color(f"speaker-{i}") for i in range(12)}
    assert len(colours) >= 2


def test_hash_color_handles_non_ascii_identities():
    """Non-ASCII labels hash by UTF-8 bytes, not by locale-dependent encoding."""
    assert _hash_color("Scott") != _hash_color("Sčot")
    assert _hash_color("Sčot") == _hash_color("Sčot")


# --- _speaker_color --------------------------------------------------------
def test_speaker_color_defaults_to_amber_without_a_label():
    assert _speaker_color(None, {}) == DEFAULT_AMBER
    assert _speaker_color("", {}) == DEFAULT_AMBER


def test_speaker_color_prefers_global_id_when_present():
    smap = {"S1": SpeakerMapEntry(local_label="S1", global_id="G1", name="Alice")}
    assert _speaker_color("S1", smap) == _hash_color("G1")


def test_speaker_color_falls_back_to_local_label_without_global_id():
    """A mapped speaker with no global id is coloured by its local label."""
    smap = {"S1": SpeakerMapEntry(local_label="S1", global_id=None, name="Alice")}
    assert _speaker_color("S1", smap) == _hash_color("S1")


def test_speaker_color_hashes_an_unmapped_label():
    assert _speaker_color("unknown-1", {}) == _hash_color("unknown-1")


# --- _speaker_display_name -------------------------------------------------
def test_speaker_display_name_uses_mapped_name():
    smap = {"S1": SpeakerMapEntry(local_label="S1", global_id="G1", name="Alice")}
    assert _speaker_display_name("S1", smap) == "Alice"


def test_speaker_display_name_falls_back_to_the_label():
    smap = {"S1": SpeakerMapEntry(local_label="S1", global_id="G1", name=None)}
    assert _speaker_display_name("S1", smap) == "S1"
    assert _speaker_display_name("unmapped", smap) == "unmapped"


def test_speaker_display_name_is_none_for_a_blank_label():
    assert _speaker_display_name(None, {}) is None
    assert _speaker_display_name("", {}) is None


# --- _cue_kind_style -------------------------------------------------------
def test_cue_kind_style_known_kind():
    style = _cue_kind_style("hook")
    assert style["accent"] == (255, 196, 0, 235)
    assert style["font_scale"] == 1.15
    assert style["tag"] == "Topic"


def test_cue_kind_style_unknown_kind_falls_back_to_line():
    assert _cue_kind_style("not-a-kind") == _cue_kind_style("line")
    assert _cue_kind_style("not-a-kind")["tag"] is None


def test_cue_kind_style_every_declared_kind_has_a_full_shape():
    """No declared cue kind may omit a field the renderers read."""
    for kind in _CUE_KIND_STYLE:
        style = _cue_kind_style(kind)
        assert set(style) == {"accent", "font_scale", "tag"}
        assert len(style["accent"]) == 4
        assert style["font_scale"] > 0
