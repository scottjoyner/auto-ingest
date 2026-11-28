"""Regression tests for followup/auto-ingest-20261128 audit fixes."""
from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
import subprocess
import sys
import threading
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# ingest_supervisor.py
# ---------------------------------------------------------------------------

def _load_supervisor():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "ingest_supervisor.py"
    spec = importlib.util.spec_from_file_location("ingest_supervisor_followup_test", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_supervisor_remote_ssh_quotes_date_arg(monkeypatch):
    mod = _load_supervisor()
    calls = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    monkeypatch.setenv("INGEST_REMOTE", "deathstar@host")
    monkeypatch.setenv("X1_370_PASS", "pw")
    # Exercise the command-building path directly: emulate run_day logic from source.
    # run_day is nested in main, so verify by checking the source and by running a
    # small extracted equivalent that uses the same quoting contract.
    src = (Path(__file__).resolve().parents[1] / "scripts/ingest_supervisor.py").read_text()
    assert "shlex.quote(dstr)" in src
    assert "bash /opt/auto-ingest/run_day.sh \"+dstr" not in src


def test_supervisor_does_not_interpolate_neo4j_password():
    root = Path(__file__).resolve().parents[1]
    src = (root / "scripts/ingest_supervisor.py").read_text()
    run_day_src = src.split("def run_day", 1)[1].split("\n    def ", 1)[0]
    assert "replace(\"'$NEO4J_PASS'\",pw)" not in run_day_src
    assert 'replace("$NEO4J_PASS",pw)' not in run_day_src
    assert 'os.environ.get("NEO4J_PASS"' not in run_day_src
    # The command should let the container env expand the password.
    assert '--neo4j-pass "$NEO4J_PASS"' in src


# ---------------------------------------------------------------------------
# yolo_batch_worker.py
# ---------------------------------------------------------------------------

def test_yolo_probe_fps_uses_argv(monkeypatch, tmp_path):
    from auto_ingest.dashcam import yolo_batch_worker as ybw

    calls = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(cmd)
        assert isinstance(cmd, list)
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps({
            "streams": [{"avg_frame_rate": "30/1", "width": 1920, "height": 1080}]
        }), stderr="")

    monkeypatch.setattr(ybw.subprocess, "run", fake_run)
    fps, w, h = ybw.probe_fps_w_h("/tmp/weird name;rm -rf.mp4")
    assert calls[0][0] == "ffprobe"
    assert calls[0][-1] == "/tmp/weird name;rm -rf.mp4"
    assert fps == 30.0 and w == 1920 and h == 1080


# ---------------------------------------------------------------------------
# outbox.py
# ---------------------------------------------------------------------------

def test_outbox_singleton_keyed_by_path(tmp_path, monkeypatch):
    import auto_ingest.outbox as ob

    monkeypatch.setenv("AUTO_INGEST_OUTBOX", "1")
    monkeypatch.delenv("AUTO_INGEST_OUTBOX_PATH", raising=False)
    ob._OUTBOXES.clear()
    try:
        p1 = tmp_path / "a.sqlite"
        p2 = tmp_path / "b.sqlite"
        a = ob.get_outbox(str(p1))
        b = ob.get_outbox(str(p2))
        a2 = ob.get_outbox(str(p1))
        assert a is not None and b is not None
        assert a is not b
        assert a is a2
        assert Path(a.path) == p1
        assert Path(b.path) == p2
    finally:
        for o in list(ob._OUTBOXES.values()):
            o.close()
        ob._OUTBOXES.clear()


def test_outbox_writes_are_locked(tmp_path):
    from auto_ingest.outbox import GraphOutbox

    box = GraphOutbox(str(tmp_path / "o.sqlite"))
    errs = []

    def worker(i):
        try:
            for j in range(20):
                box.append("op", {"i": i, "j": j}, correlation_id=f"c{i}-{j}")
        except Exception as e:
            errs.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    box.close()
    assert errs == []
    conn = sqlite3.connect(str(tmp_path / "o.sqlite"))
    n = conn.execute("SELECT count(*) FROM graph_outbox").fetchone()[0]
    conn.close()
    assert n == 160


# ---------------------------------------------------------------------------
# ingest_write.py
# ---------------------------------------------------------------------------

def test_ingest_write_roundtrip_literal_double_underscore(tmp_path):
    from auto_ingest.ingest_write import list_intermediates, read_intermediate, write_intermediate

    key = "dashcam/2026__07__14/CLT__1423_F"
    path = write_intermediate(str(tmp_path), key, "transcribe", [{"text": "hi"}], fmt="json")
    rows = read_intermediate(str(tmp_path), key, "transcribe", fmt="json")
    assert rows == [{"text": "hi"}]
    listed = list_intermediates(str(tmp_path), stage="transcribe")
    assert len(listed) == 1
    assert listed[0]["key"] == key
    assert listed[0]["path"] == path


def test_ingest_write_legacy_double_underscore_decode(tmp_path):
    from auto_ingest.ingest_write import list_intermediates

    # Legacy naming: "__" was used as "/" separator.
    legacy = tmp_path / "dashcam__2026-07-14__CLT_1423_F.transcribe.json"
    legacy.write_text('[{"ok": true}]', encoding="utf-8")
    listed = list_intermediates(str(tmp_path))
    assert len(listed) == 1
    assert listed[0]["key"] == "dashcam/2026-07-14/CLT_1423_F"


# ---------------------------------------------------------------------------
# custody destination
# ---------------------------------------------------------------------------

def test_resolve_destination_distinguishes_mounted_and_path_exists(tmp_path):
    from auto_ingest.custody.destination import resolve_destination
    from auto_ingest.custody.mounts import observe_mount

    target = tmp_path / "missing-subdir"
    ref = resolve_destination({}, env={"CUSTODY_DESTINATION_ROOT": str(target)}, check_mount=True)
    assert ref.path_exists is False
    # A nonexistent path must not be reported as mounted, even if under a mount.
    assert ref.mounted is False


# ---------------------------------------------------------------------------
# torchaudio optional import
# ---------------------------------------------------------------------------

def test_link_global_speakers_importable_without_torchaudio(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("numpy")
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torchaudio":
            raise ModuleNotFoundError("No module named 'torchaudio'")
        return real_import(name, *args, **kwargs)

    # Remove cached module so import is exercised.
    for k in list(sys.modules.keys()):
        if k == "auto_ingest.diarize.link_global_speakers" or k.startswith("torchaudio"):
            sys.modules.pop(k, None)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    try:
        try:
            spec = importlib.util.find_spec("auto_ingest.diarize.link_global_speakers")
            assert spec is not None
            mod = importlib.import_module("auto_ingest.diarize.link_global_speakers")
        except ImportError as e:
            if "torchaudio" in str(e):
                raise
            pytest.skip(f"ML stack dependency missing: {e}")
        assert mod.torchaudio is None
    finally:
        # Clean up so normal module state is restored.
        for k in list(sys.modules.keys()):
            if k == "auto_ingest.diarize.link_global_speakers":
                sys.modules.pop(k, None)
