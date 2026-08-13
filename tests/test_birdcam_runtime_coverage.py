from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from birdcam import recorder, storage, stream, worker


def test_storage_paths_metadata_and_thumbnail(monkeypatch, tmp_path):
    store = storage.Storage(str(tmp_path))
    ts = datetime(2026, 8, 13, 12, 0, 0)
    clip, thumb, meta = store.event_paths("evt", ts)
    assert clip == tmp_path / "clips/2026/08/13/evt.mp4"
    assert thumb == tmp_path / "thumbs/2026/08/13/evt.jpg"
    assert meta == tmp_path / "metadata/2026/08/13/evt.json"
    assert clip.parent.exists() and thumb.parent.exists() and meta.parent.exists()

    store.write_metadata(meta, {"event": "evt"})
    assert '"event": "evt"' in meta.read_text(encoding="utf-8")
    seen = {}
    monkeypatch.setattr(storage.cv2, "imwrite", lambda path, frame: seen.update(path=path, frame=frame) or True)
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    store.write_thumbnail(thumb, frame)
    assert seen["path"] == str(thumb)
    assert seen["frame"] is frame


def test_recorder_empty_and_writer_lifecycle(monkeypatch, tmp_path):
    assert recorder.write_clip(str(tmp_path / "none.mp4"), []) is False
    writes = []
    target = tmp_path / "clip.mp4"

    class Writer:
        def write(self, frame):
            writes.append(frame)

        def release(self):
            target.touch()

    monkeypatch.setattr(recorder.cv2, "VideoWriter_fourcc", lambda *args: 123)
    monkeypatch.setattr(recorder.cv2, "VideoWriter", lambda path, fourcc, fps, size: Writer())
    frames = [np.zeros((4, 6, 3), dtype=np.uint8), np.ones((4, 6, 3), dtype=np.uint8)]
    assert recorder.write_clip(str(target), frames, fps=12) is True
    assert writes == frames


def test_video_stream_buffer_and_reconnect_backoff(monkeypatch):
    frame1 = np.zeros((1, 1, 3), dtype=np.uint8)
    frame2 = np.ones((1, 1, 3), dtype=np.uint8)

    class OpenCap:
        def __init__(self):
            self.reads = iter([(True, frame1), (True, frame2)])

        def isOpened(self):
            return True

        def read(self):
            return next(self.reads)

        def release(self):
            pass

    monkeypatch.setattr(stream.cv2, "VideoCapture", lambda url: OpenCap())
    video = stream.VideoStream("x", buffer_seconds=1, fps=2)
    gen = video.frames()
    got1, buf1 = next(gen)
    got2, buf2 = next(gen)
    assert got1 is frame1 and got2 is frame2
    assert buf1 == [frame1]
    assert buf2 == [frame1, frame2]
    gen.close()

    class ClosedCap:
        def isOpened(self):
            return False

    monkeypatch.setattr(stream.cv2, "VideoCapture", lambda url: ClosedCap())
    sleeps = []

    def stop_after_sleep(seconds):
        sleeps.append(seconds)
        raise RuntimeError("stop")

    monkeypatch.setattr(stream.time, "sleep", stop_after_sleep)
    with pytest.raises(RuntimeError, match="stop"):
        next(stream.VideoStream("bad").frames())
    assert sleeps == [1]


def test_worker_run_file_and_finalize(monkeypatch, tmp_path):
    settings = SimpleNamespace(
        event_merge_seconds=1,
        cooldown_seconds=1,
        detection_persistence_frames=1,
        storage_root=str(tmp_path),
        camera_id="cam",
        pre_roll_seconds=1,
        post_roll_seconds=1,
        model_name_or_path="model",
    )
    repo_calls = []

    class Repo:
        def upsert_camera(self, payload):
            repo_calls.append(("camera", payload))

        def create_detection_event(self, payload):
            repo_calls.append(("event", payload))

        def add_detection(self, payload):
            repo_calls.append(("detection", payload))

        def attach_clip(self, event_id, payload):
            repo_calls.append(("clip", payload))

        def attach_thumbnail(self, event_id, payload):
            repo_calls.append(("thumb", payload))

    detection = SimpleNamespace(label="bird", confidence=0.9, bbox=(1, 2, 5, 8))

    class Detector:
        def detect(self, frame):
            return [detection]

    event = SimpleNamespace(
        id="evt",
        start=datetime(2026, 8, 13, 12, 0, 0),
        last_seen=datetime(2026, 8, 13, 12, 0, 1),
        detections=[detection],
    )

    class EventManager:
        def __init__(self, *args):
            self.calls = 0

        def update(self, now, detections):
            self.calls += 1
            return ("active", event) if self.calls == 1 else ("finalized", event)

    frames = [np.zeros((10, 20, 3), dtype=np.uint8), np.ones((10, 20, 3), dtype=np.uint8)]

    class VideoStream:
        def __init__(self, *args, **kwargs):
            pass

        def frames(self):
            yield frames[0], [frames[0]]
            yield frames[1], frames

    monkeypatch.setattr(worker, "EventManager", EventManager)
    monkeypatch.setattr(worker, "VideoStream", VideoStream)
    monkeypatch.setattr(worker, "write_clip", lambda path, buf: Path(path).touch() or True)
    monkeypatch.setattr(worker.Storage, "write_thumbnail", lambda self, path, frame: Path(path).touch())

    w = worker.Worker(settings, Detector(), Repo())
    w.run_file("input", max_frames=2)
    kinds = [kind for kind, _payload in repo_calls]
    assert kinds.count("camera") == 1
    assert kinds.count("event") == 1
    assert kinds.count("detection") == 1
    assert kinds.count("clip") == 1
    assert kinds.count("thumb") == 1
    det_payload = next(payload for kind, payload in repo_calls if kind == "detection")
    assert det_payload["class_name"] == "bird"
    assert det_payload["bbox_xywh"] == [1, 2, 4, 6]
    clip_payload = next(payload for kind, payload in repo_calls if kind == "clip")
    assert clip_payload["width"] == 20 and clip_payload["height"] == 10
    assert clip_payload["size_bytes"] == 0
