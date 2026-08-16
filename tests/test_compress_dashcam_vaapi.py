from __future__ import annotations

from pathlib import Path

import compress_dashcam2 as compress


def test_detect_codecs_prefers_hevc_vaapi_when_requested(monkeypatch):
    monkeypatch.setattr(compress, "ffmpeg_supports", lambda codec: codec == "hevc_vaapi")
    assert compress.detect_codecs(True, "/dev/dri/renderD128") == (
        "hevc_vaapi",
        "hevc",
        "/dev/dri/renderD128",
    )


def test_cpu_codec_detection_does_not_select_vaapi(monkeypatch):
    monkeypatch.setattr(compress, "ffmpeg_supports", lambda codec: codec == "libx265")
    assert compress.detect_codecs(False) == ("libx265", "hevc", None)


def test_vaapi_command_has_device_upload_and_bitrate():
    cmd = compress.build_ffmpeg_cmd(
        "ffmpeg",
        Path("input.mp4"),
        Path("out.partial"),
        "hevc_vaapi",
        "hevc",
        26,
        "medium",
        1280,
        30,
        96,
        None,
        "",
        True,
        "/dev/dri/renderD128",
        "6M",
    )
    joined = " ".join(str(part) for part in cmd)
    assert "-init_hw_device vaapi=va:/dev/dri/renderD128" in joined
    assert "-hwaccel vaapi" in joined
    assert "format=nv12,hwupload" in joined
    assert "-c:v hevc_vaapi" in joined
    assert "-b:v 6M" in joined
    assert "-tag:v hvc1" in joined


def test_vaapi_default_bitrate_is_clamped_from_crf():
    low = compress.build_ffmpeg_cmd(
        "ffmpeg", Path("in.mp4"), Path("out"), "hevc_vaapi", "hevc",
        1, "medium", 1280, 30, 96, None, "", False,
        "/dev/dri/renderD128", None,
    )
    high = compress.build_ffmpeg_cmd(
        "ffmpeg", Path("in.mp4"), Path("out"), "hevc_vaapi", "hevc",
        99, "medium", 1280, 30, 96, None, "", False,
        "/dev/dri/renderD128", None,
    )
    low_i = low.index("-b:v")
    high_i = high.index("-b:v")
    assert low[low_i + 1] == "12000k"
    assert high[high_i + 1] == "4000k"


def test_cpu_command_has_no_vaapi_flags():
    cmd = compress.build_ffmpeg_cmd(
        "ffmpeg", Path("in.mp4"), Path("out"), "libx265", "hevc",
        26, "medium", 1280, 30, 96, None, "aq-mode=3", True,
    )
    joined = " ".join(str(part) for part in cmd)
    assert "vaapi" not in joined
    assert "-crf 26" in joined
    assert "-x265-params aq-mode=3" in joined
