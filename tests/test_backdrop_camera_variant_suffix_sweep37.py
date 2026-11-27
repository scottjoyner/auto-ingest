"""Hermetic contract test: backdrop must resolve every camera-variant clip key.

DashcamClip keys in the graph carry the *full* view suffix. The repo's own
contracts enumerate ``F / R / FR / RF / BC / BC-N``:

  * ``dashcam_yolo_embeddings_ents.clip_base_key`` strips ``_F/_R/_FR``;
  * ``auto_ingest.ingest.transcripts._PAT_CAM_SUFFIX`` matches all six forms.

``select_highway_pool`` seeds from ``view == 'F'``, which includes ``_FR`` clips,
so a resolver that only understands ``_F``/``_R`` silently drops real footage.

No network, no live graph, no mount traversal: every path is a tiny file under
pytest's ``tmp_path`` (redirected to the SSD scratch dir via TMPDIR).
"""
from __future__ import annotations

from pathlib import Path

from auto_ingest.shorts import backdrop


def _make_clip(root: Path, key: str) -> Path:
    """Create a tiny, non-empty clip at the layout _fr_path_for_key expects."""
    parts = key.split("_")
    year, mmdd = parts[0], parts[1]
    d = root / year / mmdd[:2] / mmdd[2:]
    d.mkdir(parents=True, exist_ok=True)
    clip = d / f"{key}.MP4"
    clip.write_bytes(b"\x00\x00\x00\x00ftypisom")
    return clip


def test_plain_key_still_gets_front_suffix(tmp_path):
    """Control: a key with no view suffix must still resolve to <key>_F.MP4."""
    key = "2025_1002_175045"
    real = _make_clip(tmp_path, f"{key}_F")
    got = backdrop._fr_path_for_key(key, tmp_path)
    assert Path(got) == real


def test_front_key_is_not_double_suffixed(tmp_path):
    key = "2025_1002_175045_F"
    real = _make_clip(tmp_path, key)
    assert Path(backdrop._fr_path_for_key(key, tmp_path)) == real


def test_fr_key_resolves_to_its_own_clip(tmp_path):
    """_FR is a front-view key in the graph and must not get a second _F."""
    key = "2025_1002_175045_FR"
    real = _make_clip(tmp_path, key)
    assert Path(backdrop._fr_path_for_key(key, tmp_path)) == real


def test_rf_key_resolves_to_its_own_clip(tmp_path):
    key = "2025_1002_175045_RF"
    real = _make_clip(tmp_path, key)
    assert Path(backdrop._fr_path_for_key(key, tmp_path)) == real


def test_bodycam_indexed_key_resolves_to_its_own_clip(tmp_path):
    key = "2025_1002_175045_BC-2"
    real = _make_clip(tmp_path, key)
    assert Path(backdrop._fr_path_for_key(key, tmp_path)) == real


class _FakeSession:
    def __init__(self, rows):
        self._rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, query, **params):
        class _R:
            def data(self_inner):
                return self._rows
        return _R()


class _FakeDriver:
    def __init__(self, rows):
        self._rows = rows

    def session(self):
        return _FakeSession(self._rows)


def test_highway_pool_keeps_fr_clip_when_file_is_present(tmp_path, monkeypatch):
    """End-to-end-ish: an _FR anchor must carry a resolvable fr_path."""
    key = "2025_1002_175045_FR"
    _make_clip(tmp_path, key)
    monkeypatch.setattr(backdrop, "_dashcam_root", lambda: tmp_path)
    drv = _FakeDriver([{
        "clip_key": key, "frame_no": 900, "mph": 65.0,
        "fps": 30.0, "path": "unused",
    }])
    anchors = backdrop.select_highway_pool(drv)
    assert len(anchors) == 1
    assert anchors[0]["fr_path"] == str(tmp_path / "2025" / "10" / "02" / f"{key}.MP4")
