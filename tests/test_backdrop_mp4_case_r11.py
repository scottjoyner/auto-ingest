"""r11 regression tests: case-insensitive clip resolution in shorts.backdrop.

``_fr_path_for_key`` hard-coded ``.MP4`` while the dashcam cards write both
``.MP4`` and ``.mp4``. A lowercase clip silently vanished from the highway
B-roll pool even though ffmpeg accepts it. Pure filesystem fixtures only:
no Neo4j, no NAS mount, no network, no config backend, no live state.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from auto_ingest.shorts import backdrop

KEY = "2025_1002_175045"
DATE_PARTS = ("2025", "10", "02")


def _make_clip(root: Path, name: str, *, directory: bool = False) -> Path:
    date_dir = root.joinpath(*DATE_PARTS)
    date_dir.mkdir(parents=True, exist_ok=True)
    target = date_dir / name
    if directory:
        target.mkdir(exist_ok=True)
    else:
        target.write_bytes(b"fake mp4")
    return target


@pytest.mark.parametrize("ext", [".MP4", ".mp4", ".Mp4", ".mP4"])
def test_all_case_variants_of_extension_resolve(tmp_path, ext):
    expected = _make_clip(tmp_path, f"{KEY}_F{ext}")
    assert backdrop._fr_path_for_key(f"{KEY}_F", tmp_path) == expected


def test_lowercase_extension_was_the_regression(tmp_path):
    """Explicit fail-before case: the old code checked only '.MP4'."""
    expected = _make_clip(tmp_path, f"{KEY}_F.mp4")
    assert backdrop._fr_path_for_key(f"{KEY}_F", tmp_path) == expected


def test_uppercase_is_preferred_when_both_spellings_exist(tmp_path):
    upper = _make_clip(tmp_path, f"{KEY}_F.MP4")
    _make_clip(tmp_path, f"{KEY}_F.mp4")
    assert backdrop._fr_path_for_key(f"{KEY}_F", tmp_path) == upper


def test_missing_clip_returns_none(tmp_path):
    assert backdrop._fr_path_for_key(f"{KEY}_F", tmp_path) is None


def test_directory_named_clip_is_not_returned(tmp_path):
    """Fail closed: a directory is not a usable footage file."""
    _make_clip(tmp_path, f"{KEY}_F.MP4", directory=True)
    assert backdrop._fr_path_for_key(f"{KEY}_F", tmp_path) is None


def test_key_with_existing_r_suffix_resolves(tmp_path):
    expected = _make_clip(tmp_path, f"{KEY}_R.mp4")
    assert backdrop._fr_path_for_key(f"{KEY}_R", tmp_path) == expected


def test_key_without_camera_suffix_gets_f_appended(tmp_path):
    expected = _make_clip(tmp_path, f"{KEY}_F.mp4")
    assert backdrop._fr_path_for_key(KEY, tmp_path) == expected


def test_no_recursive_fallback_outside_the_date_directory(tmp_path):
    """A same-named clip elsewhere in the tree must not be picked up."""
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "elsewhere" / f"{KEY}_F.mp4").write_bytes(b"wrong clip")
    assert backdrop._fr_path_for_key(f"{KEY}_F", tmp_path) is None


def test_short_key_returns_none(tmp_path):
    assert backdrop._fr_path_for_key("2025_1002", tmp_path) is None


def test_malformed_mmdd_returns_none(tmp_path):
    assert backdrop._fr_path_for_key("2025_1_F", tmp_path) is None


def test_resolved_path_stays_under_requested_root(tmp_path):
    expected = _make_clip(tmp_path, f"{KEY}_F.mp4")
    resolved = backdrop._fr_path_for_key(f"{KEY}_F", tmp_path)
    assert resolved == expected
    assert resolved.is_relative_to(tmp_path)


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def data(self):
        return list(self._rows)


class _FakeSession:
    def __init__(self, rows):
        self._rows = rows

    def run(self, *args, **kwargs):
        return _FakeResult(self._rows)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeDriver:
    def __init__(self, rows):
        self._rows = rows

    def session(self):
        return _FakeSession(self._rows)


def test_highway_pool_resolves_lowercase_clip_end_to_end(tmp_path, monkeypatch):
    _make_clip(tmp_path, f"{KEY}_F.mp4")
    monkeypatch.setattr(backdrop, "_dashcam_root", lambda: tmp_path)
    driver = _FakeDriver(
        [
            {
                "clip_key": f"{KEY}_F",
                "frame_no": 90,
                "mph": 65.0,
                "fps": 30.0,
                "path": None,
            }
        ]
    )
    anchors = backdrop.select_highway_pool(driver, mph_min=45.0, limit=10)
    assert len(anchors) == 1
    assert anchors[0]["fr_path"].endswith(f"{KEY}_F.mp4")
