from __future__ import annotations

import importlib
import json
import sys
from argparse import Namespace
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest


def test_event_envelope_fallback_and_emit(monkeypatch):
    monkeypatch.setitem(sys.modules, "assistx", None)
    monkeypatch.setitem(sys.modules, "assistx.contracts", None)
    sys.modules.pop("auto_ingest.events", None)
    events = importlib.import_module("auto_ingest.events")
    env = events.emit("ingest.test", {"x": 1})
    data = env.to_dict()
    assert events._USING_CANONICAL is False
    assert data["source_repo"] == "auto-ingest"
    assert data["payload"] == {"x": 1}
    with pytest.raises(ValueError):
        events.EventEnvelope("1", "r", "e", correlation_id="not-a-uuid")


def _fleet_args(**overrides):
    base = dict(
        items=[], input=None, glob="*", auth_user="admin", auth_pass="pw",
        capabilities="script, media", command="do {item}", yolo_command=None,
        title_prefix="job", batch_id=None, priority="background",
        assistx_url="http://assistx",
    )
    base.update(overrides)
    return Namespace(**base)


def test_fleet_build_create_and_run(monkeypatch, tmp_path):
    from auto_ingest import fleet_batch as f

    (tmp_path / "b.mp4").write_text("x")
    (tmp_path / "a.mp4").write_text("x")
    args = _fleet_args(items=["first", "first"], input=str(tmp_path), glob="*.mp4")
    items = f.build_items(args)
    assert items[0] == "first" and len(items) == 3

    class Resp:
        def raise_for_status(self): return None
        def json(self): return {"task_id": "t1"}

    calls = []
    monkeypatch.setattr(f, "requests", SimpleNamespace(post=lambda *a, **kw: calls.append((a, kw)) or Resp()))
    out = f.create_task(
        "http://x", ("u", "p"), "title", ["script"], command="echo x",
        yolo_command="yolo x", payload={"a": 1}, correlation_id="cid",
    )
    assert out["task_id"] == "t1"
    assert calls[0][1]["json"]["payload"]["command"] == "echo x"
    monkeypatch.setattr(f, "requests", None)
    with pytest.raises(RuntimeError):
        f.create_task("x", None, "t", [])

    created = []
    monkeypatch.setattr(f, "create_task", lambda *a, **kw: created.append((a, kw)) or {"task_id": "ok"})
    assert f.run_batch(_fleet_args(items=["one", "two"], auth_pass="")) == 0
    assert len(created) == 2
    assert f.run_batch(_fleet_args(items=[], input=None)) == 1
    monkeypatch.setattr(f, "create_task", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("no")))
    assert f.run_batch(_fleet_args(items=["one"])) == 0


def test_birdcam_storage_recorder_and_stream(monkeypatch, tmp_path):
    from birdcam import recorder, storage, stream

    st = storage.Storage(str(tmp_path))
    ts = datetime(2026, 8, 13)
    clip, thumb, meta = st.event_paths("e1", ts)
    st.write_metadata(meta, {"x": 1})
    assert json.loads(meta.read_text())["x"] == 1
    monkeypatch.setattr(storage.cv2, "imwrite", lambda p, frame: True)
    st.write_thumbnail(thumb, np.zeros((2, 3, 3), dtype=np.uint8))

    assert recorder.write_clip(str(clip), []) is False
    writes = []
    class Writer:
        def write(self, frame): writes.append(frame)
        def release(self): clip.write_bytes(b"x")
    monkeypatch.setattr(recorder.cv2, "VideoWriter_fourcc", lambda *a: 1)
    monkeypatch.setattr(recorder.cv2, "VideoWriter", lambda *a, **k: Writer())
    frames = [np.zeros((2, 3, 3), dtype=np.uint8)]
    assert recorder.write_clip(str(clip), frames) is True

    class Cap:
        def __init__(self, opened, rows): self.opened=opened; self.rows=list(rows)
        def isOpened(self): return self.opened
        def read(self): return self.rows.pop(0) if self.rows else (False, None)
        def release(self): pass
    seq = iter([Cap(False, []), Cap(True, [(True, frames[0]), (False, None)])])
    monkeypatch.setattr(stream.cv2, "VideoCapture", lambda _url: next(seq))
    monkeypatch.setattr(stream.time, "sleep", lambda _n: None)
    vs = stream.VideoStream("x", buffer_seconds=1, fps=1)
    frame, buf = next(vs.frames())
    assert frame.shape == (2, 3, 3) and len(buf) == 1


def test_birdcam_worker_finalize(monkeypatch, tmp_path):
    from birdcam import worker

    settings = SimpleNamespace(
        event_merge_seconds=1, cooldown_seconds=1, detection_persistence_frames=1,
        storage_root=str(tmp_path), camera_id="cam", pre_roll_seconds=0,
        post_roll_seconds=1, model_name_or_path="m",
    )
    attached = []
    repo = SimpleNamespace(
        upsert_camera=lambda *_a, **_k: None,
        attach_clip=lambda *a, **k: attached.append(("clip", a, k)),
        attach_thumbnail=lambda *a, **k: attached.append(("thumb", a, k)),
    )
    w = worker.Worker(settings, SimpleNamespace(detect=lambda _f: []), repo)
    ts = datetime(2026, 8, 13)
    evt = SimpleNamespace(id="e1", start=ts)
    frames = [np.zeros((2, 3, 3), dtype=np.uint8)]
    clip, thumb, _ = w.storage.event_paths(evt.id, evt.start)
    monkeypatch.setattr(worker, "write_clip", lambda p, b: clip.write_bytes(b"x") or True)
    monkeypatch.setattr(w.storage, "write_thumbnail", lambda p, f: thumb.write_bytes(b"x"))
    w._finalize(evt, frames)
    assert {x[0] for x in attached} == {"clip", "thumb"}


def test_birdcam_cli_dispatch(monkeypatch):
    import birdcam.cli as cli

    s = SimpleNamespace(
        neo4j=SimpleNamespace(uri="u", username="n", password="p", database="d", use_outbox_on_failure=True),
        sqlite_path="x", model_name_or_path="m", detection_class="bird",
        confidence_threshold=0.5, stream_url="stream",
    )
    monkeypatch.setattr(cli, "load_settings", lambda _p: s)
    driver = SimpleNamespace(execute=lambda q: [{"ok": 1}])
    repo = SimpleNamespace(replay_outbox=lambda: 1, list_events=lambda **kw: [], get_event=lambda x: {"id": x})
    monkeypatch.setattr(cli, "build_repo", lambda _s: (driver, repo))
    monkeypatch.setattr(cli, "init_schema", lambda _d: None)
    monkeypatch.setattr(cli, "YoloDetector", lambda *a: object())
    runs = []
    monkeypatch.setattr(cli, "Worker", lambda *a: SimpleNamespace(run_file=lambda p: runs.append(p)))
    monkeypatch.setattr(cli.uvicorn, "run", lambda *a, **kw: runs.append("api"))
    for argv in [
        ["birdcam", "graph", "init-schema", "--config", "c"],
        ["birdcam", "graph", "check", "--config", "c"],
        ["birdcam", "graph", "replay-outbox", "--config", "c"],
        ["birdcam", "graph", "list-events", "--config", "c"],
        ["birdcam", "graph", "event", "--config", "c", "--event-id", "e"],
        ["birdcam", "detect-file", "--config", "c", "--input", "in"],
        ["birdcam", "api", "--config", "c"],
        ["birdcam", "run", "--config", "c"],
    ]:
        monkeypatch.setattr(sys, "argv", argv)
        cli.main()
    assert "in" in runs and "api" in runs and "stream" in runs
