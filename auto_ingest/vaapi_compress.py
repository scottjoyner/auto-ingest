"""Hardened VAAPI adapter for the legacy dashcam compression batch runner.

The batch/discovery/verification logic remains in ``compress_dashcam2``. This
module replaces only FFmpeg command construction when the wrapper explicitly
requests VAAPI. Keeping the adapter small makes the hardware boundary testable
without rewriting the mature batch loop.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import compress_dashcam2 as legacy


def _vaapi_bitrate(crf: int, explicit: str | None) -> str:
    if explicit:
        return explicit
    quality = min(max(crf, 18), 30)
    bitrate_k = 12000 - (quality - 18) * 650
    bitrate_k = int(bitrate_k / 500) * 500
    return f"{bitrate_k}k"


def build_ffmpeg_cmd(
    ffmpeg,
    src,
    dst_tmp,
    vcodec,
    family,
    crf,
    preset,
    max_width,
    fps,
    audio_k,
    tune,
    extra_x265,
    tag_apple,
    vaapi_device=None,
    vaapi_bitrate=None,
):
    """Build an FFmpeg command with explicit CPU->VAAPI frame upload.

    CPU decode and software scale/fps filters run before ``hwupload``. The
    named VAAPI device is supplied to the filter graph with
    ``-filter_hw_device va``. We intentionally do not request ``-hwaccel
    vaapi`` here: that is a distinct hardware-decode path and can force extra
    device/system-memory transfers when software filters are used.
    """
    if not str(vcodec).endswith("_vaapi"):
        return legacy._original_build_ffmpeg_cmd(
            ffmpeg,
            src,
            dst_tmp,
            vcodec,
            family,
            crf,
            preset,
            max_width,
            fps,
            audio_k,
            tune,
            extra_x265,
            tag_apple,
            vaapi_device,
            vaapi_bitrate,
        )
    if not vaapi_device:
        raise ValueError("VAAPI encoder requires a render device")

    filters: list[str] = []
    if max_width:
        filters.append(
            f"scale='min({max_width},iw)':'-2':force_original_aspect_ratio=decrease"
        )
    if fps:
        filters.append(f"fps={fps}")
    filters.append("format=nv12")
    filters.append("hwupload")

    command = [
        str(ffmpeg),
        "-hide_banner",
        "-y",
        "-init_hw_device",
        f"vaapi=va:{vaapi_device}",
        "-filter_hw_device",
        "va",
        "-i",
        str(src),
        "-vf",
        ",".join(filters),
        "-map_metadata",
        "0",
        "-map",
        "0",
        "-f",
        "mp4",
        "-movflags",
        "+faststart",
        "-c:v",
        str(vcodec),
        "-b:v",
        _vaapi_bitrate(int(crf), vaapi_bitrate),
    ]
    if family == "hevc" and tag_apple:
        command += ["-tag:v", "hvc1"]
    command += ["-c:a", "aac", "-b:a", f"{audio_k}k"]
    if fps:
        command += ["-vsync", "vfr"]
    command.append(str(dst_tmp))
    return command


def install() -> None:
    """Install the hardened builder into the legacy batch runner once."""
    if not hasattr(legacy, "_original_build_ffmpeg_cmd"):
        legacy._original_build_ffmpeg_cmd = legacy.build_ffmpeg_cmd
    legacy.build_ffmpeg_cmd = build_ffmpeg_cmd


def main(argv: Sequence[str] | None = None) -> None:
    install()
    if argv is not None:
        import sys

        old = sys.argv
        try:
            sys.argv = [old[0], *argv]
            legacy.main()
        finally:
            sys.argv = old
        return
    legacy.main()


if __name__ == "__main__":
    main()
