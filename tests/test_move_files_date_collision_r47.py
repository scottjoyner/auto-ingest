"""Hermetic contract tests for move_files recorder-date parsing and collision planning.

Round 47, test-only. The subject helpers are pure stdlib: filename timestamp
parsing, extension normalization, YYYY/MM/DD destination construction, content
hashing and the collision-strategy decision. No fixture file, service,
credential, clock or network is involved; every filesystem path is a pytest
tmp_path under the SSD scratch basetemp.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import move_files as mf


def _named(tmp_path: Path, name: str, data: bytes = b"abc") -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


def test_parse_name_underscored_full_timestamp():
    parsed = mf.parse_name("2025_0911_223045_F.MP4")
    assert parsed is not None
    assert (parsed.year, parsed.month, parsed.day) == (2025, 9, 11)
    assert (parsed.hour, parsed.minute, parsed.second) == (22, 30, 45)
    assert parsed.ts_compact == "20250911223045"
    assert parsed.ts_underscored == "2025_0911_223045"
    assert parsed.source == "filename"


def test_parse_name_compact_timestamp():
    parsed = mf.parse_name("20250911223045.WAV")
    assert parsed is not None
    assert (parsed.year, parsed.month, parsed.day, parsed.hour, parsed.minute, parsed.second) == (
        2025,
        9,
        11,
        22,
        30,
        45,
    )
    assert parsed.ts_compact == "20250911223045"
    assert parsed.ts_underscored == "2025_0911_223045"


@pytest.mark.parametrize(
    "basename",
    ["R2025_0911_223045.mp3", "REC_2025-09-11_22-30-45.m4a"],
)
def test_parse_name_accepts_prefixed_recorder_layouts(basename):
    parsed = mf.parse_name(basename)
    assert parsed is not None
    assert (parsed.year, parsed.month, parsed.day, parsed.hour, parsed.minute, parsed.second) == (
        2025,
        9,
        11,
        22,
        30,
        45,
    )


def test_parse_name_date_only_defaults_to_midnight():
    parsed = mf.parse_name("2025-09-11.CSV")
    assert parsed is not None
    assert (parsed.hour, parsed.minute, parsed.second) == (0, 0, 0)
    assert parsed.ts_compact == "20250911000000"
    assert parsed.ts_underscored == "2025_0911_000000"


@pytest.mark.parametrize(
    "basename",
    ["2025_1311_223045_F.MP4", "2025_0230_223045_F.MP4", "notes.txt"],
)
def test_parse_name_rejects_impossible_or_absent_dates(basename):
    assert mf.parse_name(basename) is None


def test_parsed_from_parts_defaults_missing_time_fields():
    parsed = mf._parsed_from_parts({"year": "2024", "month": "01", "day": "02"}, "filename")
    assert parsed is not None
    assert (parsed.hour, parsed.minute, parsed.second) == (0, 0, 0)
    assert parsed.ts_compact == "20240102000000"
    assert mf._parsed_from_parts({"year": "2024", "month": "13", "day": "02"}, "filename") is None


def test_normalize_extensions_all_and_mixed():
    assert mf.normalize_extensions("all") is None
    assert mf.normalize_extensions("ALL") is None
    assert mf.normalize_extensions("MP3, wav,.flac, ,") == {".mp3", ".wav", ".flac"}
    assert mf.normalize_extensions("") == set()


def test_build_destination_uses_zero_padded_year_month_day():
    parsed = mf.parse_name("2025_0911_223045_F.MP4")
    assert parsed is not None
    assert mf.build_destination(Path("/out"), "clip.mp4", parsed) == Path(
        "/out/2025/09/11/clip.mp4"
    )


def test_compute_hash_is_deterministic_and_algorithm_specific(tmp_path):
    src = _named(tmp_path, "src.mp4", b"abc")
    assert mf.compute_hash(src, "md5") == "900150983cd24fb0d6963f7d28e17f72"
    assert mf.compute_hash(src, "sha256") == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
    other = _named(tmp_path, "other.mp4", b"xyz")
    assert mf.compute_hash(other, "md5") != mf.compute_hash(src, "md5")


def test_choose_collision_action_missing_destination_and_same_path(tmp_path):
    src = _named(tmp_path, "src.mp4")
    dst = tmp_path / "dst.mp4"
    assert mf.choose_collision_action(src, dst, "skip", None) == dst
    assert mf.choose_collision_action(src, src, "skip", None) is None


def test_choose_collision_action_skip_and_overwrite(tmp_path):
    src = _named(tmp_path, "src.mp4", b"abc")
    dst = _named(tmp_path, "dst.mp4", b"xyz")
    assert mf.choose_collision_action(src, dst, "skip", None) is None
    assert mf.choose_collision_action(src, dst, "skip", "md5") is None
    assert mf.choose_collision_action(src, dst, "overwrite", None) == dst


def test_choose_collision_action_fail_and_unknown_strategy(tmp_path):
    src = _named(tmp_path, "src.mp4")
    dst = _named(tmp_path, "dst.mp4", b"xyz")
    with pytest.raises(FileExistsError, match="Destination exists"):
        mf.choose_collision_action(src, dst, "fail", None)
    with pytest.raises(ValueError, match="Unknown strategy"):
        mf.choose_collision_action(src, dst, "bogus", None)


def test_choose_collision_action_rename_skips_existing_candidates(tmp_path):
    src = _named(tmp_path, "src.mp4")
    dst = _named(tmp_path, "dst.mp4", b"xyz")
    first = mf.choose_collision_action(src, dst, "rename", None)
    assert first == tmp_path / "dst_1.mp4"
    _named(tmp_path, "dst_1.mp4", b"keep")
    second = mf.choose_collision_action(src, dst, "rename", None)
    assert second == tmp_path / "dst_2.mp4"


def test_choose_collision_action_directory_destination_appends_source_name(tmp_path):
    src = _named(tmp_path, "src.mp4")
    directory = tmp_path / "outdir"
    directory.mkdir()
    assert mf.choose_collision_action(src, directory, "rename", None) == (directory / "src.mp4")
