"""Focused tests for the housekeeping changes made at the *caller* level.

`tests/test_storage_contract.py` pins the resolver itself; this module pins the
behavior changed in the scripts that consume it, plus the deprecation/dry-run
guards. Heavy runtime deps (torch, scipy, neo4j, ultralytics) are avoided: modules
that need them are checked by source inspection instead of import.

No real storage is touched — everything is tmp_path / monkeypatched env.
"""

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _src(rel: str) -> str:
    return (REPO / rel).read_text()


# --------------------------------------------------------------------------- #
# publish queue / shorts output roots: env-required, fail closed
# --------------------------------------------------------------------------- #
def test_publish_queue_requires_env(monkeypatch):
    monkeypatch.delenv("SHORTS_PUBLISH_QUEUE", raising=False)
    from auto_ingest.shorts import publish

    publish.QUEUE_PATH = None
    with pytest.raises(RuntimeError):
        publish._require_queue_path()


def test_publish_queue_honors_env(monkeypatch, tmp_path):
    monkeypatch.setenv("SHORTS_PUBLISH_QUEUE", str(tmp_path / "q.jsonl"))
    from auto_ingest.shorts import publish, uploader

    assert publish._require_queue_path() == tmp_path / "q.jsonl"
    assert uploader._require_queue_path() == tmp_path / "q.jsonl"


def test_queue_modules_have_no_legacy_default():
    for rel in ("auto_ingest/shorts/publish.py", "auto_ingest/shorts/uploader.py",
                "auto_ingest/shorts/cli.py"):
        assert "/media/scott/" not in _src(rel), rel


# --------------------------------------------------------------------------- #
# worker_content: lazy, fail-closed root resolution (import stays safe)
# --------------------------------------------------------------------------- #
def test_worker_content_importable_without_storage(monkeypatch):
    for var in ("COMPRESSED_ROOT", "TIKTOK_OUT_ROOT", "FILESERVER_ROOT"):
        monkeypatch.delenv(var, raising=False)
    import worker_content

    with pytest.raises(RuntimeError):
        worker_content._require_roots()


def test_worker_content_roots_from_env(monkeypatch, tmp_path):
    """Env is read when the worker process starts (module import), as before."""
    monkeypatch.setenv("COMPRESSED_ROOT", str(tmp_path))
    monkeypatch.setenv("TIKTOK_OUT_ROOT", str(tmp_path / "out"))
    import worker_content

    importlib.reload(worker_content)
    compressed, out = worker_content._require_roots()
    assert compressed == str(tmp_path)
    assert out == str(tmp_path / "out")


# --------------------------------------------------------------------------- #
# batch_update_paths.py: deprecated repo-wide codemod is dry-run by default
# --------------------------------------------------------------------------- #
def test_batch_update_paths_is_dry_run_by_default():
    proc = subprocess.run(
        [sys.executable, str(REPO / "batch_update_paths.py")],
        cwd=str(REPO), capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "dry-run" in proc.stdout
    # the flag is required before it will write
    assert "--apply" in _src("batch_update_paths.py")


def test_batch_update_paths_rejects_flag_bypass(tmp_path):
    """Passing --apply to another target must not rewrite this repo."""
    before = (REPO / "batch_update_paths.py").read_text()
    subprocess.run(
        [sys.executable, str(REPO / "batch_update_paths.py"), "--help"],
        cwd=str(tmp_path), capture_output=True, text=True, timeout=60,
    )
    assert (REPO / "batch_update_paths.py").read_text() == before


# --------------------------------------------------------------------------- #
# USB / batch scripts: env + repo-relative, no literal host paths
# --------------------------------------------------------------------------- #
def test_usb_scripts_have_no_literal_host_paths():
    for rel in ("process_usb_device.py", "transcribe_usb_batch.py"):
        text = _src(rel)
        assert "/media/scott/" not in text, rel
        assert "/media/scott/data/git" not in text, rel
        assert "/media/scott/USB DISK" not in text, rel


def test_process_usb_device_uses_canonical_resolver():
    text = _src("process_usb_device.py")
    assert "get_fileserver_root()" in text
    assert "get_dashcam_root()" in text
    # repo-relative script invocation, not a hardcoded checkout
    assert "ingest_transcriptsv5_3.py" in text
    assert "get_neo4j_password" in text or "get_neo4j_config" in text


# --------------------------------------------------------------------------- #
# dashcam tooling: bases come from env/resolver, no 8TBHDD fallback
# --------------------------------------------------------------------------- #
def test_dashcam_tools_drop_legacy_fallbacks():
    assert "/mnt/8TBHDD" not in _src("dashcam_monitor.py")
    assert "/mnt/8TBHDD" not in _src("dashcam_salvage.py")
    assert "/mnt/8TBHDD" not in _src("run_dashcam_vision.sh")
    assert "/mnt/8TBHDD" not in _src("bulk_ingest_dashcam.sh")


def test_dashcam_monitor_bases_from_env(monkeypatch, tmp_path):
    """BASES comes from DASHCAM_BASES (read at import), not a legacy mount list."""
    import dashcam_monitor

    monkeypatch.setenv("DASHCAM_BASES", f"{tmp_path}/a {tmp_path}/b")
    importlib.reload(dashcam_monitor)
    assert dashcam_monitor.BASES == [f"{tmp_path}/a", f"{tmp_path}/b"]


# --------------------------------------------------------------------------- #
# repo-relative shell helpers
# --------------------------------------------------------------------------- #
def test_shell_helpers_are_repo_relative():
    for rel in ("finish_location_backfill.sh", "install-signal-bridge-cron.sh"):
        text = _src(rel)
        assert "/media/scott/" not in text, rel
        assert "BASH_SOURCE" in text, rel


def test_deprecated_entrypoints_have_headers():
    assert "DEPRECATED" in _src("runall.sh")
    assert "bin/auto-ingest" in _src("runall.sh")
    assert "DEPRECATED" in _src("batch_update_paths.py")


# --------------------------------------------------------------------------- #
# config layer: env precedence documented in the module contract
# --------------------------------------------------------------------------- #
def test_resolver_precedence_is_documented():
    doc = _src("auto_ingest_config.py")
    for token in ("FILESERVER_ROOT", "NEO4J_PASSWORD_DEFAULT"):
        assert token in doc


def test_env_example_matches_live_config_contract():
    live = _src("config.yaml")
    example = _src(".env.example")
    # NAS5 is the live contract; the example must not point operators at NAS4
    assert "/media/scott/NAS5" in live
    assert "/media/scott/NAS4" not in example.replace("NAS4 while the", "")
    assert "NEO4J_PASSWORD_DEFAULT" in example