from __future__ import annotations
import time
from collections import deque
import cv2

class VideoStream:
    def __init__(self, url: str, buffer_seconds: int = 5, fps: float = 10.0):
        self.url = url
        self.buffer = deque(maxlen=max(1, int(buffer_seconds * fps)))

    def frames(self):
        backoff = 1
        while True:
            cap = cv2.VideoCapture(self.url)
            if not cap.isOpened():
                time.sleep(backoff)
                backoff = min(30, backoff * 2)
                continue
            backoff = 1
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                self.buffer.append(frame)
                yield frame, list(self.buffer)
            cap.release()
