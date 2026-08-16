"""Best-effort OpenCV VAAPI hardware decoding.

OpenCV exposes hardware acceleration as an *open-only* VideoCapture property.
That means setting CAP_PROP_HW_ACCELERATION after ``VideoCapture(path)`` has
already opened the stream is not sufficient. Call ``open_capture`` for the
accelerated path; it attempts an FFmpeg/VAAPI open and falls back to the normal
CPU-backed capture if the backend, driver, codec, or build does not support it.

The helper is intentionally safe on non-Linux and non-VAAPI machines. Set
``AUTO_INGEST_VAAPI=0`` to disable attempts, or ``AUTO_INGEST_VAAPI=1`` to
request an attempt whenever OpenCV advertises the required API. The render-node
probe is retained as a host capability guard and can be overridden with
``AUTO_INGEST_VAAPI_DEVICE``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

_VAAPI_NODES = (
    "/dev/dri/renderD128",
    "/dev/dri/renderD129",
)


def _cv2_vaapi_supported(cv2) -> bool:
    return all(
        getattr(cv2, name, None) is not None
        for name in (
            "CAP_FFMPEG",
            "CAP_PROP_HW_ACCELERATION",
            "VIDEO_ACCELERATION_VAAPI",
        )
    )


def vaapi_device() -> Optional[str]:
    """Return the configured/first existing VAAPI render node, or ``None``."""
    env = os.environ.get("AUTO_INGEST_VAAPI_DEVICE")
    if env:
        return env if Path(env).exists() else None
    for node in _VAAPI_NODES:
        if Path(node).exists():
            return node
    return None


def vaapi_requested() -> bool:
    """Whether this host should attempt a VAAPI capture open."""
    value = os.environ.get("AUTO_INGEST_VAAPI")
    if value == "0":
        return False
    if value == "1":
        return True
    return os.name == "posix" and vaapi_device() is not None


def enable_vaapi(cap) -> bool:
    """Compatibility probe for already-open captures.

    ``CAP_PROP_HW_ACCELERATION`` is open-only in OpenCV, so this function no
    longer claims to enable acceleration after construction. It returns whether
    the current capture reports VAAPI, when that property is readable. Existing
    callers remain safe, but new code should use :func:`open_capture`.
    """
    if not vaapi_requested():
        return False
    try:
        import cv2  # noqa: PLC0415
    except Exception:
        return False
    if not _cv2_vaapi_supported(cv2):
        return False
    try:
        return int(cap.get(cv2.CAP_PROP_HW_ACCELERATION)) == int(
            cv2.VIDEO_ACCELERATION_VAAPI
        )
    except Exception:
        return False


def _open_vaapi(cv2, path: str):
    params = [
        int(cv2.CAP_PROP_HW_ACCELERATION),
        int(cv2.VIDEO_ACCELERATION_VAAPI),
    ]
    return cv2.VideoCapture(path, cv2.CAP_FFMPEG, params)


def open_capture(path: str):
    """Open ``path`` with VAAPI when possible, otherwise fall back to CPU."""
    import cv2  # noqa: PLC0415

    if vaapi_requested() and _cv2_vaapi_supported(cv2):
        cap = None
        try:
            cap = _open_vaapi(cv2, path)
            if cap is not None and cap.isOpened():
                return cap
        except Exception:
            pass
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
    return cv2.VideoCapture(path)
