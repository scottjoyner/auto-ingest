"""Hermetic contract for the talking-head corner-placement helper (r45).

``auto_ingest.shorts.persona._corner_pos`` is the pure geometry step that maps a
configured corner name to the ``(x, y)`` offset passed to moviepy's
``set_position``. It has no model, network, DB or media dependency, but nothing
in the suite pinned its output before this file: the surrounding persona tests
cover source resolution and fallback composition, not the placement arithmetic.

These tests lock the published contract so a future change to the padding or
the vertical offset cannot silently move an avatar off-screen. No file fixture,
service or live run is involved.
"""

from __future__ import annotations

from auto_ingest.shorts.persona import _corner_pos

# A representative 1080x1920 vertical short with a 216px-wide face overlay.
W, H, FW = 1080, 1920, 216


def test_bottom_right_anchor() -> None:
    assert _corner_pos("bottom-right", W, H, FW) == (824, 1484)


def test_bottom_left_anchor() -> None:
    assert _corner_pos("bottom-left", W, H, FW) == (40, 1484)


def test_top_right_anchor() -> None:
    assert _corner_pos("top-right", W, H, FW) == (824, 80)


def test_top_left_anchor() -> None:
    assert _corner_pos("top-left", W, H, FW) == (40, 80)


def test_unknown_corner_falls_back_to_bottom_right() -> None:
    fallback = _corner_pos("bottom-right", W, H, FW)
    for corner in ("middle", "", "center", None, 7):
        assert _corner_pos(corner, W, H, FW) == fallback


def test_corner_match_is_exact_and_case_sensitive() -> None:
    fallback = _corner_pos("bottom-right", W, H, FW)
    assert _corner_pos("Bottom-Right", W, H, FW) == fallback
    assert _corner_pos("bottom-right ", W, H, FW) == fallback
    assert _corner_pos("bottom-right", W, H, FW) != _corner_pos("top-right", W, H, FW)


def test_both_bottom_corners_share_the_same_vertical_offset() -> None:
    assert _corner_pos("bottom-right", W, H, FW)[1] == H - FW - 220
    assert _corner_pos("bottom-left", W, H, FW)[1] == H - FW - 220


def test_both_top_corners_use_the_fixed_80px_vertical_offset() -> None:
    assert _corner_pos("top-right", W, H, FW)[1] == 80
    assert _corner_pos("top-left", W, H, FW)[1] == 80


def test_horizontal_edges_are_40px_from_the_selected_side() -> None:
    assert _corner_pos("bottom-left", W, H, FW)[0] == 40
    assert _corner_pos("top-left", W, H, FW)[0] == 40
    assert _corner_pos("bottom-right", W, H, FW)[0] == W - FW - 40
    assert _corner_pos("top-right", W, H, FW)[0] == W - FW - 40


def test_face_width_shifts_right_edge_and_bottom_vertical_together() -> None:
    a = _corner_pos("bottom-right", W, H, 100)
    b = _corner_pos("bottom-right", W, H, 300)
    assert a == (940, 1600)
    assert b == (740, 1400)
    assert b[0] - a[0] == -200
    assert b[1] - a[1] == -200


def test_top_vertical_is_independent_of_face_width() -> None:
    assert _corner_pos("top-left", W, H, 100) == _corner_pos("top-left", W, H, 300)


def test_all_four_named_corners_are_distinct() -> None:
    positions = {
        _corner_pos(name, W, H, FW)
        for name in ("bottom-right", "bottom-left", "top-right", "top-left")
    }
    assert len(positions) == 4


def test_result_is_a_plain_pair_of_ints_and_is_deterministic() -> None:
    pos = _corner_pos("bottom-right", W, H, FW)
    assert isinstance(pos, tuple)
    assert len(pos) == 2
    assert all(isinstance(v, int) for v in pos)
    assert pos == _corner_pos("bottom-right", W, H, FW)
