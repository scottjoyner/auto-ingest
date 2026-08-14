from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from auto_ingest import vaapi_probe


def test_device_status_reports_permissions(monkeypatch, tmp_path: Path):
    device = tmp_path / "renderD128"
    device.write_bytes(b"")
    monkeypatch.setattr(vaapi_probe.os, "access", lambda path, mode: mode == vaapi_probe.os.R_OK)
    status = vaapi_probe.device_status(device)
    assert status == {
        "path": str(device),
        "exists": True,
        "readable": True,
        "writable": False,
    }


def test_ffmpeg_capabilities_parses_hwaccel_and_encoders(monkeypatch):
    monkeypatch.setattr(vaapi_probe.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    outputs = iter(
        [
            {"ok": True, "stdout": "Hardware acceleration methods:\nvaapi\n", "stderr": ""},
            {"ok": True, "stdout": " V..... hevc_vaapi\n V..... h264_vaapi\n", "stderr": ""},
        ]
    )
    monkeypatch.setattr(vaapi_probe, "_run", lambda command: next(outputs))
    report = vaapi_probe.ffmpeg_capabilities()
    assert report["vaapi_hwaccel"] is True
    assert report["encoders"] == {"hevc_vaapi": True, "h264_vaapi": True}


def test_smoke_encode_uses_named_filter_device(monkeypatch):
    monkeypatch.setattr(vaapi_probe.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    seen = {}

    def fake_run(command, timeout=10):
        seen["command"] = list(command)
        seen["timeout"] = timeout
        return {"ok": True, "returncode": 0, "stdout": "", "stderr": ""}

    monkeypatch.setattr(vaapi_probe, "_run", fake_run)
    report = vaapi_probe.smoke_encode(device="/dev/dri/renderD129", encoder="hevc_vaapi")
    command = report["command"]
    assert report["ok"] is True
    assert "vaapi=va:/dev/dri/renderD129" in command
    assert command[command.index("-filter_hw_device") + 1] == "va"
    assert command[command.index("-vf") + 1] == "format=nv12,hwupload"
    assert "-hwaccel" not in command
    assert command[command.index("-c:v") + 1] == "hevc_vaapi"


def test_probe_smokes_only_advertised_encoders(monkeypatch):
    monkeypatch.setattr(
        vaapi_probe,
        "device_status",
        lambda device: {"path": device, "exists": True, "readable": True, "writable": True},
    )
    monkeypatch.setattr(
        vaapi_probe,
        "ffmpeg_capabilities",
        lambda ffmpeg="ffmpeg": {
            "available": True,
            "vaapi_hwaccel": True,
            "encoders": {"hevc_vaapi": True, "h264_vaapi": False},
        },
    )
    monkeypatch.setattr(vaapi_probe, "vainfo_status", lambda device: {"available": False})
    calls = []
    monkeypatch.setattr(
        vaapi_probe,
        "smoke_encode",
        lambda **kwargs: calls.append(kwargs) or {"ok": True, "encoder": kwargs["encoder"]},
    )
    report = vaapi_probe.probe(device="/dev/dri/renderD128")
    assert report["usable"] is True
    assert calls == [
        {"device": "/dev/dri/renderD128", "encoder": "hevc_vaapi", "ffmpeg": "ffmpeg"}
    ]


def test_probe_is_not_usable_without_device(monkeypatch):
    monkeypatch.setattr(
        vaapi_probe,
        "device_status",
        lambda device: {"path": device, "exists": False, "readable": False, "writable": False},
    )
    monkeypatch.setattr(
        vaapi_probe,
        "ffmpeg_capabilities",
        lambda ffmpeg="ffmpeg": {
            "available": True,
            "vaapi_hwaccel": True,
            "encoders": {"hevc_vaapi": True, "h264_vaapi": True},
        },
    )
    monkeypatch.setattr(vaapi_probe, "vainfo_status", lambda device: {"available": False})
    assert vaapi_probe.probe()["usable"] is False


def test_run_handles_timeout_or_execution_error(monkeypatch):
    monkeypatch.setattr(
        vaapi_probe.subprocess,
        "run",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    result = vaapi_probe._run(("ffmpeg", "-version"))
    assert result["ok"] is False
    assert "RuntimeError" in result["error"]


def test_vainfo_summary_is_bounded(monkeypatch):
    monkeypatch.setattr(vaapi_probe.shutil, "which", lambda name: "/usr/bin/vainfo")
    monkeypatch.setattr(
        vaapi_probe,
        "_run",
        lambda command, timeout=10: {
            "ok": True,
            "returncode": 0,
            "stdout": "\n".join(f"line-{i}" for i in range(100)),
            "stderr": "",
        },
    )
    report = vaapi_probe.vainfo_status()
    assert report["ok"] is True
    assert len(report["summary"]) == 80
