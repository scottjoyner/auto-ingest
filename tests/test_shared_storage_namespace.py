"""Tests for the fleet shared-storage namespace contract.

These tests distinguish the canonical shared data plane (/nas) from host-local
processing/cache roots. They perform no network or storage writes outside
pytest's tmp_path.
"""

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_config(tmp_path: Path, *, shared: Path, fileserver: Path) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(
        f"""
machine_paths:
  x1-370:
    hostname_pattern: x1-370
    shared_root: {shared}
    fileserver_root: {fileserver}
    hot_ssd_root: {tmp_path / "hot"}
    nas_mirror_root: {shared}
knowledge_map:
  mirror_vault_path: {shared / "shared-knowledge"}
"""
    )
    return p


def _prepare_roots(tmp_path: Path) -> tuple[Path, Path]:
    shared = tmp_path / "nas"
    fileserver = tmp_path / "local-fileserver"
    for root in (shared, fileserver):
        root.mkdir()
        (root / ".contract-marker").write_text("test")
    return shared, fileserver


def test_shared_root_is_independent_from_processing_root(tmp_path, monkeypatch):
    import auto_ingest_config as cfg

    shared, fileserver = _prepare_roots(tmp_path)
    config = _write_config(tmp_path, shared=shared, fileserver=fileserver)
    monkeypatch.setattr(cfg, "_find_config_path", lambda: config)
    monkeypatch.setattr(cfg.socket, "gethostname", lambda: "x1-370")
    monkeypatch.delenv("NAS_ROOT", raising=False)
    monkeypatch.delenv("FILESERVER_ROOT", raising=False)

    assert cfg.get_shared_root() == str(shared)
    assert cfg.get_fileserver_root() == str(fileserver)
    assert cfg.get_shared_root() != cfg.get_fileserver_root()


def test_nas_root_environment_override_wins(tmp_path, monkeypatch):
    import auto_ingest_config as cfg

    configured, fileserver = _prepare_roots(tmp_path)
    override = tmp_path / "override-nas"
    override.mkdir()
    (override / ".contract-marker").write_text("test")
    config = _write_config(tmp_path, shared=configured, fileserver=fileserver)
    monkeypatch.setattr(cfg, "_find_config_path", lambda: config)
    monkeypatch.setattr(cfg.socket, "gethostname", lambda: "x1-370")
    monkeypatch.setenv("NAS_ROOT", str(override))

    assert cfg.get_shared_root() == str(override)
    assert cfg.get_storage_layout()["shared_root"] == str(override)


def test_shared_root_fails_closed_when_unconfigured(monkeypatch):
    import auto_ingest_config as cfg

    monkeypatch.setattr(cfg, "_find_config_path", lambda: None)
    monkeypatch.delenv("NAS_ROOT", raising=False)

    with pytest.raises(RuntimeError, match="Shared NAS root is not configured"):
        cfg.get_shared_root()


def test_shared_root_rejects_empty_unmounted_directory(tmp_path, monkeypatch):
    import auto_ingest_config as cfg

    empty = tmp_path / "empty-nas"
    empty.mkdir()
    monkeypatch.setattr(cfg, "_find_config_path", lambda: None)
    monkeypatch.setenv("NAS_ROOT", str(empty))

    with pytest.raises(RuntimeError, match="not mounted"):
        cfg.get_shared_root()


def test_shared_path_is_bounded_to_namespace(tmp_path, monkeypatch):
    import auto_ingest_config as cfg

    shared, fileserver = _prepare_roots(tmp_path)
    config = _write_config(tmp_path, shared=shared, fileserver=fileserver)
    monkeypatch.setattr(cfg, "_find_config_path", lambda: config)
    monkeypatch.setattr(cfg.socket, "gethostname", lambda: "x1-370")
    monkeypatch.delenv("NAS_ROOT", raising=False)

    assert cfg.get_shared_path("archive", "dashcam") == str(
        (shared / "archive" / "dashcam").resolve()
    )
    with pytest.raises(ValueError, match="escapes NAS root"):
        cfg.get_shared_path("..", "outside")


def test_repository_profiles_pin_fleet_shared_root_to_nas():
    cfg = yaml.safe_load((REPO_ROOT / "config.yaml").read_text())

    for name in ("x1-370", "any", "deathstar"):
        profile = cfg["machine_paths"][name]
        assert profile["shared_root"] == "/nas"
        assert profile["nas_mirror_root"] == "/nas"

    # Local processing roots are deliberately not collapsed into the shared root.
    assert cfg["machine_paths"]["x1-370"]["fileserver_root"] != "/nas"
    assert cfg["machine_paths"]["deathstar"]["fileserver_root"] != "/nas"


def test_repository_custody_and_knowledge_paths_use_namespace_contract():
    cfg = yaml.safe_load((REPO_ROOT / "config.yaml").read_text())

    assert cfg["custody"]["destination_relative_path"] == "archive/dashcam"
    assert cfg["knowledge_map"]["mirror_vault_path"] == "/nas/shared-knowledge"
    assert cfg["knowledge_map"]["canonical_vault_path"] == "/nas/shared-knowledge"


@pytest.mark.parametrize("filename", [".env.example", "deploy/path_profiles.env.example"])
def test_container_profiles_use_nas_namespace(filename):
    lines = {}
    for raw in (REPO_ROOT / filename).read_text().splitlines():
        if not raw or raw.lstrip().startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        lines[key.strip()] = value.strip()

    assert lines["NAS_ROOT"] == "/nas"
    for key in (
        "DROP_ROOT",
        "AUDIO_ROOT",
        "DASHCAM_ROOT",
        "BODYCAM_ROOT",
        "TRANSCRIPT_ROOT",
        "LEGACY_ROOT",
        "LEGACY_DROP_ROOT",
        "YOLO_ROOT",
        "META_ROOT",
        "CANONICAL_ROOT",
    ):
        assert lines[key].startswith("/nas"), (filename, key, lines[key])

    for path in lines["SCAN_ROOTS"].split(","):
        assert path.startswith("/nas"), (filename, path)


def test_package_shim_exports_shared_resolvers():
    from auto_ingest import _config

    assert callable(_config.get_shared_root)
    assert callable(_config.get_shared_path)
