"""CV2 VAAPI hardware-decode helper (RX 480 / Polaris).

OpenCV 4.10+ exposes CAP_PROP_HW_ACCELERATION. On AMD/Intel with a working
VAAPI render node this moves H.264/H.265 *decode* onto the GPU, cutting CPU
time ~8x for pure decode-heavy frame work (YOLO CSV generation, patch crops,
HUD OCR). Encode is NOT done here (see compress_dashcam2 --vaapi).

Enable automatically when the platform is Linux, cv2 advertises VAAPI, and the
render node exists. Callers that want the flag regardless can set
AUTO_INGEST_VAAPI=0 to disable or AUTO_INGEST_VAAPI=1 to force.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

# Common VAAPI render nodes in priority order (first existing wins).
_VAAPI_NODES = (
    "/dev/dri/renderD128",
    "/dev/dri/renderD129",
)


def _cv2_vaapi_supported(cv2) -> bool:
    return (
        getattr(cv2, "CAP_PROP_HW_ACCELERATION", None) is not None
        and getattr(cv2, "VIDEO_ACCELERATION_VAAPI", None) is not None
    )


def vaapi_device() -> Optional[str]:
    """Return the first usable VAAPI render node, or None.

    An explicit override must be a character device: a regular file or
    directory that merely exists is not a render node, and handing one to
    OpenCV would make ``enable_vaapi`` claim hardware decode that cannot work.
    ``is_char_device()`` is false for a missing path too, so the previous
    fail-closed behaviour for a bad override is preserved.
    """
    env = os.environ.get("AUTO_INGEST_VAAPI_DEVICE")
    if env:
        return env if Path(env).is_char_device() else None
    for node in _VAAPI_NODES:
        if Path(node).exists():
            return node
    return None


def enable_vaapi(cap) -> None:
    """Best-effort: enable VAAPI hw decode on an already-created VideoCapture.

    Safe to call unconditionally; silently no-ops when unsupported/disabled.
    """
    if os.environ.get("AUTO_INGEST_VAAPI") == "0":
        return
    try:
        import cv2  # noqa: PLC0415
    except Exception:
        return
    if not _cv2_vaapi_supported(cv2):
        return
    if vaapi_device() is None:
        return
    try:
        # Property must be set *before* frames are read.
        cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_VAAPI)
    except Exception:
        pass


def open_capture(path: str):
    """Open a VideoCapture with VAAPI hw decode enabled (best-effort)."""
    import cv2  # noqa: PLC0415
    cap = cv2.VideoCapture(path)
    enable_vaapi(cap)
    return cap
