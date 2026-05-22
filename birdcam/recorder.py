from __future__ import annotations
from pathlib import Path
import cv2


def write_clip(path: str, frames: list, fps: int = 10):
    if not frames:
        return False
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    return Path(path).exists()
