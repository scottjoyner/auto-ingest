"""Hermetic config-validation tests for the cold-root mount fence.

No live mounts, no network, no credentials: every path is under ``tmp_path`` and
the config finder is monkeypatched to a synthetic file.
"""
from __future__ import annotations

import pytest

import auto_ingest_config


def _point_cold_root_at(tmp_path, monkeypatch, candidate):
    """Force get_storage_layout()'s cold root to a synthetic candidate."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "machine_paths:\n"
        "  synthetic:\n"
        "    hostname_pattern: x1-370\n"
        f"    nas_mirror_root: {candidate}\n"
    )
    monkeypatch.setattr(auto_ingest_config, "_find_config_path", lambda: cfg)
    for var in ("COLD_STORAGE_ROOT", "HOT_STORAGE_ROOT", "FILESERVER_ROOT"):
        monkeypatch.delenv(var, raising=False)


def test_cold_root_rejects_regular_file(tmp_path, monkeypatch):
    """A configured cold root that is a regular file must not pass the fence."""
    fake_root = tmp_path / "not-a-directory"
    fake_root.write_text("data")
    _point_cold_root_at(tmp_path, monkeypatch, fake_root)
    with pytest.raises(RuntimeError, match="not a directory"):
        auto_ingest_config.get_cold_root()


def test_require_mounted_accepts_non_empty_directory(tmp_path):
    live = tmp_path / "cold"
    live.mkdir()
    (live / "probe").write_text("x")
    assert auto_ingest_config._require_mounted(live) == str(live)


def test_require_mounted_still_rejects_empty_directory(tmp_path):
    empty = tmp_path / "empty-cold"
    empty.mkdir()
    with pytest.raises(RuntimeError, match="not mounted"):
        auto_ingest_config._require_mounted(empty)


def test_require_mounted_still_rejects_missing_path(tmp_path):
    missing = tmp_path / "absent-cold"
    with pytest.raises(RuntimeError, match="does not exist"):
        auto_ingest_config._require_mounted(missing)
