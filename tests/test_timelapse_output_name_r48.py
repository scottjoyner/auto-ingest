"""Regression contract for timelapse destination naming (r48).

``output_path_for`` is the single place that decides where a timelapse lands.
The module docstring promises ``<key>_FRTL.MP4`` for a ``<key>_FR.MP4`` input in
both modes: with ``--out-dir`` and with ``--in-place``. The in-place branch
strips the ``_FR`` marker; the out-dir branch did not, so it produced
``<key>_FR_FRTL.MP4`` and silently broke the documented output contract.

These tests are hermetic: every path is a pytest ``tmp_path`` and no ffmpeg or
filesystem write beyond the temp tree is involved.
"""

from pathlib import Path

import pytest

import timelapse


@pytest.mark.parametrize(
    ("name", "expected_name"),
    [
        ("K123_FR.MP4", "K123_FRTL.MP4"),
        ("dash_2025_0911_FR.MP4", "dash_2025_0911_FRTL.MP4"),
        ("already_timelapsed_FRTL.MP4", "already_timelapsed_FRTL_FRTL.MP4"),
        ("plain.MP4", "plain_FRTL.MP4"),
    ],
)
def test_out_dir_name_matches_documented_contract(tmp_path, name, expected_name):
    src = tmp_path / name
    out = tmp_path / "timelapse-out"

    got = timelapse.output_path_for(src, out, False)

    assert got == out / expected_name


@pytest.mark.parametrize("name", ["K123_FR.MP4", "dash_2025_0911_FR.MP4", "plain.MP4"])
def test_out_dir_and_in_place_agree_on_basename(tmp_path, name):
    src = tmp_path / name
    out = tmp_path / "timelapse-out"

    out_dir_name = timelapse.output_path_for(src, out, False).name
    in_place_name = timelapse.output_path_for(src, None, True).name

    assert out_dir_name == in_place_name


def test_default_out_dir_is_sibling_named_by_key(tmp_path):
    card = tmp_path / "2025" / "09" / "11"
    card.mkdir(parents=True)
    src = card / "K123_FR.MP4"

    got = timelapse.output_path_for(src, None, False)

    assert got == card / timelapse.DEFAULT_OUT_DIRNAME / "K123_FRTL.MP4"


def test_in_place_output_stays_next_to_source(tmp_path):
    card = tmp_path / "card"
    card.mkdir()
    src = card / "K123_FR.MP4"

    got = timelapse.output_path_for(src, card / "ignored", True)

    assert got == card / "K123_FRTL.MP4"


def test_explicit_out_dir_is_used_verbatim(tmp_path):
    src = tmp_path / "K123_FR.MP4"
    out = Path(tmp_path) / "elsewhere"

    got = timelapse.output_path_for(src, out, False)

    assert got.parent == out
