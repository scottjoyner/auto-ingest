"""Runtime VAAPI capability/preflight probe.

Encoder names in ``ffmpeg -encoders`` only prove build-time support. This module
also opens the selected DRM render node and performs a one-frame encode so a host
can prove its userspace driver, permissions, FFmpeg device context, upload filter,
and hardware encoder all work together.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Sequence

DEFAULT_DEVICE = "/dev/dri/renderD128"
ENCODERS = ("hevc_vaapi", "h264_vaapi")


def _run(command: Sequence[str], timeout: int = 10) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except Exception as exc:
        return {
            "ok": False,
            "returncode": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
    }


def device_status(device: str | Path) -> dict[str, Any]:
    path = Path(device)
    return {
        "path": str(path),
        "exists": path.exists(),
        "readable": os.access(path, os.R_OK) if path.exists() else False,
        "writable": os.access(path, os.W_OK) if path.exists() else False,
    }


def ffmpeg_capabilities(ffmpeg: str = "ffmpeg") -> dict[str, Any]:
    executable = shutil.which(ffmpeg)
    if not executable:
        return {
            "available": False,
            "vaapi_hwaccel": False,
            "encoders": {name: False for name in ENCODERS},
        }
    hwaccels = _run((executable, "-hide_banner", "-hwaccels"))
    encoders = _run((executable, "-hide_banner", "-encoders"))
    hw_text = f"{hwaccels.get('stdout', '')}\n{hwaccels.get('stderr', '')}"
    enc_text = f"{encoders.get('stdout', '')}\n{encoders.get('stderr', '')}"
    return {
        "available": True,
        "path": executable,
        "vaapi_hwaccel": "vaapi" in hw_text.lower(),
        "encoders": {name: name in enc_text for name in ENCODERS},
    }


def smoke_encode(
    *,
    device: str = DEFAULT_DEVICE,
    encoder: str = "hevc_vaapi",
    ffmpeg: str = "ffmpeg",
) -> dict[str, Any]:
    executable = shutil.which(ffmpeg)
    if not executable:
        return {"ok": False, "error": "ffmpeg not found"}
    command = (
        executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-init_hw_device",
        f"vaapi=va:{device}",
        "-filter_hw_device",
        "va",
        "-f",
        "lavfi",
        "-i",
        "color=size=64x64:rate=1:duration=1",
        "-vf",
        "format=nv12,hwupload",
        "-frames:v",
        "1",
        "-c:v",
        encoder,
        "-f",
        "null",
        "-",
    )
    result = _run(command, timeout=15)
    result["encoder"] = encoder
    result["device"] = device
    result["command"] = list(command)
    return result


def vainfo_status(device: str = DEFAULT_DEVICE) -> dict[str, Any]:
    executable = shutil.which("vainfo")
    if not executable:
        return {"available": False}
    result = _run((executable, "--display", "drm", "--device", device), timeout=10)
    text = f"{result.get('stdout', '')}\n{result.get('stderr', '')}"
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return {
        "available": True,
        "ok": bool(result.get("ok")),
        "summary": lines[:80],
    }


def probe(
    *,
    device: str = DEFAULT_DEVICE,
    ffmpeg: str = "ffmpeg",
    run_smoke: bool = True,
) -> dict[str, Any]:
    dev = device_status(device)
    caps = ffmpeg_capabilities(ffmpeg)
    smoke: dict[str, Any] = {}
    if run_smoke and dev["exists"] and dev["readable"] and dev["writable"] and caps["available"]:
        for encoder in ENCODERS:
            if caps["encoders"].get(encoder):
                smoke[encoder] = smoke_encode(device=device, encoder=encoder, ffmpeg=ffmpeg)
    usable = any(item.get("ok") for item in smoke.values()) if run_smoke else (
        dev["exists"]
        and dev["readable"]
        and dev["writable"]
        and any(caps["encoders"].values())
    )
    return {
        "usable": bool(usable),
        "device": dev,
        "ffmpeg": caps,
        "vainfo": vainfo_status(device),
        "smoke": smoke,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m auto_ingest.vaapi_probe")
    parser.add_argument("--device", default=os.environ.get("AUTO_INGEST_VAAPI_DEVICE", DEFAULT_DEVICE))
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--no-smoke", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    report = probe(device=args.device, ffmpeg=args.ffmpeg, run_smoke=not args.no_smoke)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"VAAPI usable: {report['usable']}")
        print(f"Device: {report['device']}")
        print(f"FFmpeg: {report['ffmpeg']}")
        for encoder, result in report["smoke"].items():
            print(f"{encoder}: {'PASS' if result.get('ok') else 'FAIL'}")
            if not result.get("ok") and result.get("stderr"):
                print(result["stderr"].strip())
    return 0 if report["usable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
