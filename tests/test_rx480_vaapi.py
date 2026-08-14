from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

from auto_ingest import cv2_vaapi


def test_open_capture_uses_open_only_vaapi_params(monkeypatch):
    calls = []

    class Cap:
        def __init__(self, opened=True):
            self.opened = opened
            self.released = False

        def isOpened(self):
            return self.opened

        def release(self):
            self.released = True

    fake = ModuleType("cv2")
    fake.CAP_FFMPEG = 1900
    fake.CAP_PROP_HW_ACCELERATION = 50
    fake.VIDEO_ACCELERATION_VAAPI = 3

    def video_capture(*args):
        calls.append(args)
        return Cap(True)

    fake.VideoCapture = video_capture
    monkeypatch.setitem(__import__("sys").modules, "cv2", fake)
    monkeypatch.setenv("AUTO_INGEST_VAAPI", "1")

    cap = cv2_vaapi.open_capture("clip.mp4")
    assert cap.isOpened()
    assert calls == [
        (
            "clip.mp4",
            fake.CAP_FFMPEG,
            [fake.CAP_PROP_HW_ACCELERATION, fake.VIDEO_ACCELERATION_VAAPI],
        )
    ]


def test_open_capture_falls_back_when_vaapi_open_fails(monkeypatch):
    calls = []

    class Cap:
        def __init__(self, opened):
            self.opened = opened
            self.released = False

        def isOpened(self):
            return self.opened

        def release(self):
            self.released = True

    failed = Cap(False)
    cpu = Cap(True)
    fake = ModuleType("cv2")
    fake.CAP_FFMPEG = 1900
    fake.CAP_PROP_HW_ACCELERATION = 50
    fake.VIDEO_ACCELERATION_VAAPI = 3

    def video_capture(*args):
        calls.append(args)
        return failed if len(args) == 3 else cpu

    fake.VideoCapture = video_capture
    monkeypatch.setitem(__import__("sys").modules, "cv2", fake)
    monkeypatch.setenv("AUTO_INGEST_VAAPI", "1")

    result = cv2_vaapi.open_capture("clip.mp4")
    assert result is cpu
    assert failed.released is True
    assert calls[-1] == ("clip.mp4",)


def test_vaapi_requested_respects_explicit_disable(monkeypatch):
    monkeypatch.setenv("AUTO_INGEST_VAAPI", "0")
    assert cv2_vaapi.vaapi_requested() is False
    monkeypatch.setenv("AUTO_INGEST_VAAPI", "1")
    assert cv2_vaapi.vaapi_requested() is True


def _write_fake_python(bin_dir: Path, calls_path: Path) -> None:
    script = bin_dir / "python3"
    script.write_text(
        "#!/bin/bash\n"
        "if [[ \"$1\" == \"-c\" ]]; then\n"
        "  case \"$2\" in\n"
        "    *get_fileserver_root*) echo /tmp/fileserver ;;\n"
        "    *neo4j_config*uri*) echo bolt://localhost:7687 ;;\n"
        "    *neo4j_config*user*) echo neo4j ;;\n"
        "    *neo4j_config*password*) echo test ;;\n"
        "    *) echo x ;;\n"
        "  esac\n"
        "  exit 0\n"
        "fi\n"
        f"printf '%s\\n' \"$*\" >> {calls_path}\n"
        "exit 0\n",
        encoding="utf-8",
    )
    script.chmod(0o755)


def _run_wrapper(tmp_path: Path, *, vaapi: str) -> str:
    calls = tmp_path / "calls.txt"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_fake_python(bin_dir, calls)
    env = dict(os.environ)
    env.update(
        PATH=f"{bin_dir}:{env['PATH']}",
        VAAPI=vaapi,
        FILESERVER_ROOT="/tmp/fileserver",
        INPUT_ROOT="/tmp/in",
        OUTPUT_ROOT="/tmp/out",
        LIMIT="0",
    )
    result = subprocess.run(
        ["bash", "run_compress_dashcam.sh"],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return calls.read_text(encoding="utf-8")


def test_wrapper_vaapi_zero_does_not_pass_flag(tmp_path: Path):
    args = _run_wrapper(tmp_path, vaapi="0")
    assert "--vaapi" not in args
    assert "--limit" not in args


def test_wrapper_vaapi_one_passes_device_flag(tmp_path: Path):
    args = _run_wrapper(tmp_path, vaapi="1")
    assert "--vaapi" in args
    assert "--vaapi-device /dev/dri/renderD128" in args


def test_wrapper_rejects_invalid_boolean(tmp_path: Path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls.txt"
    _write_fake_python(bin_dir, calls)
    env = dict(os.environ)
    env.update(PATH=f"{bin_dir}:{env['PATH']}", VAAPI="sometimes", FILESERVER_ROOT="/tmp/fileserver")
    result = subprocess.run(
        ["bash", "run_compress_dashcam.sh"],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "VAAPI must be" in result.stderr
