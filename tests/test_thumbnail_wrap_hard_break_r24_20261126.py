"""Regression: cover-title wrapping must hard-break overlong tokens (r24).

Hermetic by construction: the only dependency of ``_wrap`` is the ``draw``
object's ``textlength`` method, so these tests use a deterministic stub and
never render an image, load a font, touch the network or a graph store.
"""
from __future__ import annotations

import pytest

# PYTHONPATH selects the source under test (pristine copy for fail-before,
# fixed worktree for pass-after); no path is forced here.
from auto_ingest.shorts.thumbnail import _wrap


class UniformDraw:
    """Deterministic 10px per character; no PIL image/font needed."""

    def __init__(self, per_char: int = 10) -> None:
        self.per_char = per_char

    def textlength(self, text, font=None):
        return len(text) * self.per_char


class VariableDraw:
    """Per-character widths so a narrow 'i' and a wide 'm' differ."""

    WIDTHS = {"i": 4, "l": 4, "m": 16, "w": 16}

    def textlength(self, text, font=None):
        return sum(self.WIDTHS.get(ch, 10) for ch in text)


def test_single_overlong_token_is_hard_broken():
    """A token wider than the band must be split into fitting lines."""
    draw = UniformDraw()
    max_w = 100  # 10 chars
    lines = _wrap(draw, "supercalifragilisticexpialidocious", None, max_w)
    assert lines, "an overlong word must still produce some text"
    assert all(draw.textlength(line) <= max_w for line in lines), (
        f"every line must fit the band, got {lines!r}"
    )


def test_overlong_token_followed_by_words():
    """Mixing an overlong token with normal words must not leave an overflow."""
    draw = UniformDraw()
    max_w = 100
    # 20-char token hard-breaks to 2 lines, leaving the 3rd for the next word.
    lines = _wrap(draw, "supercalifragilistic enough", None, max_w)
    assert all(draw.textlength(line) <= max_w for line in lines), lines
    assert "enough" in " ".join(lines)


def test_normal_wrapping_is_unchanged():
    """Control: ordinary multi-word text wraps exactly as before."""
    draw = UniformDraw()
    assert _wrap(draw, "short normal words fit fine", None, 100) == [
        "short", "normal", "words fit",
    ]


def test_variable_width_font_measurement_respected():
    """Hard-break honours the draw's real measurement, not len()."""
    draw = VariableDraw()
    max_w = 30
    lines = _wrap(draw, "iiiiiiiiiiiiiiii", None, max_w)  # 16 narrow chars = 64px
    assert all(draw.textlength(line) <= max_w for line in lines), lines


def test_three_line_cap_preserved():
    """The existing 3-line cap is intentional and must survive the fix."""
    draw = UniformDraw()
    lines = _wrap(draw, "aaa bbb ccc ddd eee fff", None, 100)
    assert len(lines) <= 3


@pytest.mark.parametrize("text", ["", "   ", "\t\n"])
def test_empty_or_whitespace_text(text):
    draw = UniformDraw()
    assert _wrap(draw, text, None, 100) == []
