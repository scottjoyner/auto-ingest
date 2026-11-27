"""An empty or non-file 'clip' must never be reported as resolvable footage.

Hermetic: temp files and in-memory stubs only. No Neo4j, mounts, network,
credentials, services or model loads.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from auto_ingest.shorts import backdrop

KEY_EMPTY = "2025_1002_175045_F"
KEY_GOOD = "2025_1002_180000_F"
KEY_DIR = "2025_1002_190000_F"


@pytest.fixture
def dashcam_root(tmp_path: Path) -> Path:
    """Three look-alike clips: truncated, usable, and a directory."""
    day = tmp_path / "2025" / "10" / "02"
    day.mkdir(parents=True)
    (day / f"{KEY_EMPTY}.MP4").write_bytes(b"")          # zero-byte recording
    (day / f"{KEY_GOOD}.MP4").write_bytes(b"\x00" * 64)  # usable recording
    (day / f"{KEY_DIR}.MP4").mkdir()                     # not a file at all
    return tmp_path


def test_nonempty_clip_is_resolvable(dashcam_root: Path):
    """Control: a real clip is still found (the fix must not over-block)."""
    resolved = backdrop._fr_path_for_key(KEY_GOOD, dashcam_root)
    assert resolved is not None
    assert resolved.name == f"{KEY_GOOD}.MP4"


def test_zero_byte_clip_is_not_resolvable(dashcam_root: Path):
    """A truncated recording is missing footage, not renderable footage.

    Matches the repo's own dashcam contract in
    ``auto_ingest.dashcam.media_pairing.find_sibling_media``.
    """
    assert backdrop._fr_path_for_key(KEY_EMPTY, dashcam_root) is None


def test_directory_named_like_clip_is_not_resolvable(dashcam_root: Path):
    """A directory at the clip path is not a playable clip."""
    assert backdrop._fr_path_for_key(KEY_DIR, dashcam_root) is None


def test_pool_and_pick_shots_never_select_empty_footage(dashcam_root: Path,
                                                        monkeypatch):
    """End-to-end: a truncated clip reaches neither the pool nor the shot list."""
    rows = [
        {"clip_key": KEY_EMPTY, "frame_no": 0, "mph": 65.0,
         "fps": 30.0, "path": "irrelevant"},
        {"clip_key": KEY_GOOD, "frame_no": 0, "mph": 66.0,
         "fps": 30.0, "path": "irrelevant"},
    ]

    class _Result:
        def data(self):
            return rows

    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def run(self, query, **params):
            return _Result()

    class _Driver:
        def session(self):
            return _Session()

    monkeypatch.setattr(backdrop, "_dashcam_root", lambda: dashcam_root)
    pool = backdrop.select_highway_pool(_Driver(), limit=10)
    by_key = {a["clip_key"]: a for a in pool}
    assert by_key[KEY_EMPTY]["fr_path"] is None
    assert by_key[KEY_GOOD]["fr_path"] is not None

    picked = backdrop.pick_shots(pool, count=3, min_gap_sec=0.0)
    assert [shot["clip_key"] for shot in picked] == [KEY_GOOD]
