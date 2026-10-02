"""Storage-path contract tests for the housekeeping/static-hardening campaign.

These pin the canonical resolver semantics of ``auto_ingest_config.py``:
env override wins, config.yaml is the configured authority, and silent
fallback to legacy machine mounts is forbidden. All I/O is tmp_path-based;
no real storage is touched.
"""

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

CONFIG_TMPL = """
machine_paths:
  testhost:
    fileserver_root: {fs}
    neo4j_uri: bolt://127.0.0.1:7687
    neo4j_user: neo4j
    neo4j_password: ${{NEO4J_PASSWORD}}
    hostname_pattern: testhost
    hot_ssd_root: {hot}
    nas_mirror_root: {cold}
"""

LEGACY_LITERALS = [
    "/media/scott/NAS3",
    "/media/scott/NAS4",
    "/media/scott/NAS1",
    "/media/scott/NAS2",
    "/mnt/8TBHDD",
    "/home_scott_",
    "home_scott_",
    "/media/scott/data/git",
    "/media/scott/SSD_4TB/hermes-home",
]

ACTIVE_FILES = [
    "auto_ingest_config.py",
    "auto_ingest/_config.py",
    "process_usb_device.py",
    "transcribe_usb_batch.py",
    "worker_content.py",
    "dashcam_monitor.py",
    "dashcam_salvage.py",
    "dashcam_frame_vision.py",
    "deploy/manage.sh",
    "deploy/job_trigger_api.py",
    "scripts/get_embedding_endpoint.py",
    "scripts/knowledge_sync_handler.py",
    "scripts/sync_knowledge_to_nas.sh",
    "scripts/run_narrated_mix.py",
    "scripts/anchor_speaker_me.py",
    "scripts/backfill_audio_day.sh",
    "scripts/qa_audio_day.sh",
    "finish_location_backfill.sh",
    "install-signal-bridge-cron.sh",
    "run_pipeline.sh",
    "mount_nextcloud.sh",
    "bulk_ingest_dashcam.sh",
    "run_dashcam_vision.sh",
    "auto_ingest/events.py",
    "auto_ingest/shorts/cli.py",
    "auto_ingest/shorts/publish.py",
    "auto_ingest/shorts/uploader.py",
    "add_audio.sh",
    "parse_vehicel_detections.py",
    "save_detected_files.py",
    "yolo_vehicle_detction.py",
    "scripts/ingest_supervisor.py",
    "runall.sh",
    "bulk_ingest_dashcam.sh",
]


def _write(tmp_path, fs, hot, cold):
    p = tmp_path / "config.yaml"
    p.write_text(CONFIG_TMPL.format(fs=fs, hot=hot, cold=cold))
    return p


@pytest.fixture
def roots(tmp_path):
    fs = tmp_path / "srv" / "fileserver"
    hot = tmp_path / "srv" / "hot"
    cold = tmp_path / "srv" / "cold"
    for d in (fs, hot, cold):
        d.mkdir(parents=True)
        (d / ".keep").write_text("")
    return fs, hot, cold


def _clear(monkeypatch):
    for v in ("HOT_STORAGE_ROOT", "COLD_STORAGE_ROOT", "FILESERVER_ROOT",
              "AUDIO_ROOT", "DASHCAM_ROOT", "BODYCAM_ROOT", "NEXTCLOUD_ROOT"):
        monkeypatch.delenv(v, raising=False)


def test_configured_path_resolves(tmp_path, monkeypatch, roots):
    import auto_ingest_config as cfg
    fs, hot, cold = roots
    _clear(monkeypatch)
    monkeypatch.setattr("auto_ingest_config._find_config_path",
                        lambda: _write(tmp_path, fs, hot, cold))
    assert cfg.get_fileserver_root() == str(fs)
    assert cfg.get_hot_root() == str(hot)
    assert cfg.get_dashcam_root() == os.path.join(str(fs), "dashcam")


def test_env_override_precedence(tmp_path, monkeypatch, roots):
    import auto_ingest_config as cfg
    fs, hot, cold = roots
    _clear(monkeypatch)
    monkeypatch.setattr("auto_ingest_config._find_config_path",
                        lambda: _write(tmp_path, fs, hot, cold))
    monkeypatch.setenv("FILESERVER_ROOT", "/env/fs")
    assert cfg.get_fileserver_root() == "/env/fs"
    layout = cfg.get_storage_layout()
    assert layout["fileserver_root"] == "/env/fs"


def test_missing_storage_fails_closed(tmp_path, monkeypatch):
    import auto_ingest_config as cfg
    _clear(monkeypatch)
    monkeypatch.setattr("auto_ingest_config._find_config_path", lambda: None)
    with pytest.raises(RuntimeError):
        cfg.get_fileserver_root()
    with pytest.raises(RuntimeError):
        cfg.get_hot_root()
    with pytest.raises(RuntimeError):
        cfg.get_audio_root()
    with pytest.raises(RuntimeError):
        cfg.get_cold_root()
    with pytest.raises(RuntimeError):
        cfg.get_nextcloud_root()


def test_legacy_nas3_dir_does_not_win(tmp_path, monkeypatch, roots):
    import auto_ingest_config as cfg
    fs, hot, cold = roots
    _clear(monkeypatch)
    # Simulate a legacy NAS3 mount existing on the host.
    legacy = tmp_path / "media" / "scott" / "NAS3"
    legacy.mkdir(parents=True)
    (legacy / ".keep").write_text("")
    monkeypatch.setattr("auto_ingest_config._find_config_path",
                        lambda: _write(tmp_path, fs, hot, cold))
    assert cfg.get_fileserver_root() == str(fs)
    assert "NAS3" not in cfg.get_fileserver_root()


def test_legacy_ssd4tb_dir_does_not_win(tmp_path, monkeypatch, roots):
    import auto_ingest_config as cfg
    fs, hot, cold = roots
    _clear(monkeypatch)
    legacy = tmp_path / "media" / "scott" / "SSD_4TB" / "fileserver"
    legacy.mkdir(parents=True)
    (legacy / ".keep").write_text("")
    monkeypatch.setattr("auto_ingest_config._find_config_path",
                        lambda: _write(tmp_path, fs, hot, cold))
    assert cfg.get_fileserver_root() == str(fs)
    assert "SSD_4TB" not in cfg.get_fileserver_root()


def test_generic_host_config_has_no_scott_paths(tmp_path, monkeypatch, roots):
    import auto_ingest_config as cfg
    fs, hot, cold = roots
    _clear(monkeypatch)
    monkeypatch.setattr("auto_ingest_config._find_config_path",
                        lambda: _write(tmp_path, fs, hot, cold))
    # The entire pipeline resolves with tmp/generic paths only.
    assert not cfg.get_fileserver_root().startswith("/media/scott")
    assert not cfg.get_hot_root().startswith("/media/scott")
    assert not cfg.get_cold_root().startswith("/media/scott")


def test_resolver_is_deterministic(tmp_path, monkeypatch, roots):
    import auto_ingest_config as cfg
    fs, hot, cold = roots
    _clear(monkeypatch)
    monkeypatch.setattr("auto_ingest_config._find_config_path",
                        lambda: _write(tmp_path, fs, hot, cold))
    first = cfg.get_fileserver_root()
    second = cfg.get_fileserver_root()
    assert first == second
    assert cfg.get_storage_layout() == cfg.get_storage_layout()


def test_no_legacy_literals_in_active_files():
    offenders = []
    for rel in ACTIVE_FILES:
        text = (REPO_ROOT / rel).read_text()
        for lit in LEGACY_LITERALS:
            if lit in text:
                offenders.append(f"{rel}: {lit}")
    assert not offenders, "legacy literals in active files: " + "; ".join(offenders)
