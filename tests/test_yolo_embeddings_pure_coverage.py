from __future__ import annotations

import datetime as dt
import json
from collections import namedtuple
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from auto_ingest.dashcam import yolo_embeddings as y


def test_identity_seconds_and_numeric_helpers(monkeypatch):
    assert y.clip_base_key("clip_FR") == "clip"
    assert y.clip_base_key("clip") == "clip"
    frame = pd.DataFrame({"Frame": [0, 30, np.nan, 9999]})
    out = y.assign_seconds(frame, 30.0, 3.0, "k")
    assert out["__sec__"].tolist() == [0, 1]
    for col in ("t", "time", "timestamp"):
        out = y.assign_seconds(pd.DataFrame({col: [0.2, 1.9, -1, 99]}), 30, 3, "k")
        assert out["__sec__"].tolist() == [0, 1]
    monkeypatch.setattr(np.random, "randint", lambda a, b, size: np.array([0, 1, 2]))
    out = y.assign_seconds(pd.DataFrame({"x": [1, 2, 3]}), 30, 3, "k")
    assert out["__sec__"].tolist() == [0, 1, 2]
    assert y.second_from_frame(None, 30) is None
    assert y.second_from_frame(float("nan"), 30) is None
    assert y.second_from_frame(61, 30) == 2
    assert y.haversine_m(35, -80, 35, -80) == pytest.approx(0)
    assert y.plausible_fix(None, 35, -80, 20)
    assert not y.plausible_fix(None, None, -80, 20)
    assert not y.plausible_fix(None, 35, -80, 200)
    assert not y.plausible_fix((35, -80, 20), 40, -80, 20)
    assert y.safe_literal_eval("[1,2]") == [1, 2]
    assert y.safe_literal_eval("nope") is None
    z = np.zeros(3)
    assert y.ensure_unit_l2(z) is z
    v = y.ensure_unit_l2(np.array([3.0, 4.0]))
    assert np.linalg.norm(v) == pytest.approx(1)
    assert y.flatten_grid(np.array([[1, 2]])).dtype == np.float32
    assert y.bbox_overlap_area(0, 0, 2, 2, 1, 1, 3, 3) == 1
    assert y.bbox_overlap_area(0, 0, 1, 1, 2, 2, 3, 3) == 0


def test_grid_bbox_parsing_and_detection_filters(tmp_path):
    g = np.zeros((2, 2), dtype=np.float32)
    spec = y.GridSpec(2, 2)
    y.add_bbox_to_grid(g, (0, 0, 100, 100), 100, 100, 1, spec, density=True)
    assert g.sum() == pytest.approx(1)
    before = g.copy()
    y.add_bbox_to_grid(g, (5, 5, 5, 10), 100, 100, 1, spec)
    assert np.array_equal(g, before)
    assert y._split_ignoring_brackets("a,[1,2,3,4],c") == ["a", "[1,2,3,4]", "c"]
    assert y._maybe_list4([1, 2, 3, 4]) == [1.0, 2.0, 3.0, 4.0]
    assert y._maybe_list4("[1 2 3 4]") == [1.0, 2.0, 3.0, 4.0]
    assert y._maybe_list4(float("nan")) is None
    assert y._maybe_list4("x") is None
    assert y._percent_to_float01("75%") == pytest.approx(0.75)
    assert y._percent_to_float01("0.4") == pytest.approx(0.4)
    assert np.isnan(y._percent_to_float01("bad"))

    p = tmp_path / "y.csv"
    p.write_text(
        "frame,confidence,xyxy,cls\n0,80%,[0,0,10,10],2\n30,car,[1 2 3 4],2\n",
        encoding="utf-8",
    )
    df = y.parse_yolo_csv(str(p))
    assert len(df) == 2
    assert df.iloc[0]["confidence"] == pytest.approx(0.8)
    assert df.iloc[1]["name"] == "car"
    empty = tmp_path / "empty.csv"
    empty.write_text("frame,name\n", encoding="utf-8")
    assert y.parse_yolo_csv(str(empty)).empty

    Row = namedtuple("Row", "a b")
    assert y.row_to_dict(Row(1, 2)) == {"a": 1, "b": 2}
    assert y.row_to_dict(pd.Series({"a": 1})) == {"a": 1}
    assert y.row_to_dict((1, 2), ["a", "b"]) == {"a": 1, "b": 2}
    assert y.row_to_dict(object()) == {}

    assert y.to_xyxy_abs({"xyxy": [1, 2, 5, 6]}, 100, 100) == (1, 2, 5, 6)
    assert y.to_xyxy_abs({"xyxyn": [0.1, 0.2, 0.5, 0.6]}, 100, 200) == (10, 40, 50, 120)
    assert y.to_xyxy_abs({"xywh": [50, 50, 20, 10]}, 100, 100) == (40, 45, 60, 55)
    assert y.to_xyxy_abs({"xywhn": [0.5, 0.5, 0.2, 0.2]}, 100, 100) == (40, 40, 60, 60)
    assert y.to_xyxy_abs({"x1": 1, "y1": 2, "x2": 3, "y2": 4}, 100, 100) == (1, 2, 3, 4)
    assert y.to_xyxy_abs({"xc": 0.5, "yc": 0.5, "w": 0.2, "h": 0.2}, 100, 100) == (40, 40, 60, 60)
    assert y.to_xyxy_abs({}, 100, 100) is None
    keep = {"car", "motorbike"}
    assert y.keep_detection({"name": "car"}, keep, {})
    assert y.keep_detection({"name": "motorcycle"}, keep, {})
    assert y.keep_detection({"name": "", "class": 2}, keep, {2: "car"})
    assert not y.keep_detection({"name": "person", "class": 0}, keep, {})


def test_time_location_selection_and_resolution(monkeypatch):
    when = y.parse_key_datetime("2026_0102_030405_F")
    assert when == dt.datetime(2026, 1, 2, 3, 4, 5, tzinfo=dt.timezone.utc)
    assert y.parse_key_datetime("bad") is None
    assert y.parse_key_datetime("2026_9999_999999") is None
    s, c = y.cyc(0, 1)
    assert s == pytest.approx(0)
    assert c == pytest.approx(1)
    vec, sc = y.location_feature(35, -80, 40, when, True)
    assert len(vec) == 7 and sc["mph"] == 40
    vec0, sc0 = y.location_feature(None, None, None, None, False)
    assert np.allclose(vec0, 0) and sc0["lat"] is None

    primary = {"lat": 35.0, "lon": -80.0, "mph": 20}
    fallback = {"lat": 35.1, "lon": -80.0, "mph": 20}
    assert y.choose_location(primary, fallback, None, None)[1] == "LocationEvent"
    assert y.choose_location({"lat": None}, fallback, None, None)[1] == "PhoneLog"
    meta = {"lat": 35.1, "lon": -80.0, "mph": 20}
    assert y.choose_location({"lat": None}, fallback, meta, None)[1] == "PhoneLog"
    assert y.choose_location(None, None, meta, None)[1] == "metadata_csv"
    assert y.choose_location(None, None, None, None)[1] == "none"

    class Result:
        def __init__(self, row):
            self.row = row

        def single(self):
            return self.row

    class Sess:
        def run(self, q, **kw):
            if q == y.NQ_FIND_NEAREST_LOCEVENT:
                sec = abs(dt.datetime.fromisoformat(kw["t_utc"]).hour - 3)
                return Result({"elem_id": "L", "seconds": sec, "lat": 35.0, "lon": -80.0, "mph": 20})
            return Result({"elem_id": "P", "seconds": 4, "lat": 35.0, "lon": -80.0, "mph": 20})

    p, f = y.find_best_locevent_then_phonelog(Sess(), when, 10, None)
    assert p["elem_id"] == "L" and f["elem_id"] == "P"

    monkeypatch.setattr(
        y,
        "neo4j_find_nearest_frame",
        lambda *a, **k: {"elem_id": "F", "frame": 31, "lat": 35.0, "lon": -80.0, "mph": 20},
    )
    sc, src, eid, delta = y.resolve_location_for_second(Sess(), "k", 30, 1, when, 10, None, None)
    assert (src, eid) == ("Frame", "F") and delta == pytest.approx(1 / 30)
    monkeypatch.setattr(y, "neo4j_find_nearest_frame", lambda *a, **k: None)
    sc, src, eid, delta = y.resolve_location_for_second(Sess(), "k", 30, 1, when, 10, None, None)
    assert src == "LocationEvent"
    monkeypatch.setattr(y, "find_best_locevent_then_phonelog", lambda *a, **k: (None, None))
    sc, src, eid, delta = y.resolve_location_for_second(Sess(), "k", 30, 1, when, 10, meta, None)
    assert src == "metadata_csv"


def test_video_metadata_and_embedding_helpers(monkeypatch, tmp_path):
    assert y._parse_rate("30000/1001") == pytest.approx(29.97002997)
    assert y._parse_rate("30") == 30
    assert y._parse_rate("bad") is None and y._parse_rate(None) is None
    good = SimpleNamespace(
        returncode=0,
        stdout=json.dumps(
            {"streams": [{"width": 1920, "height": 1080, "avg_frame_rate": "30/1", "nb_frames": "300"}], "format": {}}
        ),
        stderr="",
    )
    monkeypatch.setattr(y.subprocess, "run", lambda *a, **k: good)
    assert y.ffprobe_video_meta("x") == (1920, 1080, 30.0, 10.0)
    bad = SimpleNamespace(returncode=1, stdout="", stderr="nope")
    monkeypatch.setattr(y.subprocess, "run", lambda *a, **k: bad)
    with pytest.raises(RuntimeError):
        y.ffprobe_video_meta("x")
    monkeypatch.setattr(y.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout="{}", stderr=""))
    with pytest.raises(RuntimeError):
        y.ffprobe_video_meta("x")

    csv = tmp_path / "d.csv"
    csv.write_text("frame,name\n0,car\n89,car\n", encoding="utf-8")
    assert y.infer_meta_from_csv(str(csv), 30) == (0, 0, 30.0, 3.0)
    assert y.infer_meta_from_csv(str(tmp_path / "none.csv")) is None
    monkeypatch.setattr(y.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0))
    assert y.try_fix_missing_moov("a", "b")
    monkeypatch.setattr(y, "ffprobe_video_meta", lambda p: (1, 2, 3.0, 4.0))
    assert y.get_video_meta("x") == (1, 2, 3.0, 4.0)
    monkeypatch.setattr(y, "ffprobe_video_meta", lambda p: (_ for _ in ()).throw(RuntimeError()))
    monkeypatch.setattr(y, "opencv_video_meta", lambda p: (5, 6, 7.0, 8.0))
    assert y.get_video_meta("x") == (5, 6, 7.0, 8.0)

    rows = [{"xyxy": [0, 0, 50, 50], "confidence": 0.5}, {"xyxy": [50, 50, 100, 100], "confidence": 1.0}]
    vec = y.build_grid_embedding_for_interval(rows, 100, 100, [y.GridSpec(2, 2)], True, True)
    assert vec.shape == (4,) and vec.sum() == pytest.approx(1.5)
    assert y.aggregate_seconds_to_minute([], mode="sum") is None
    arrs = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    assert np.allclose(y.aggregate_seconds_to_minute(arrs, mode="sum"), [4, 6])
    assert np.allclose(y.aggregate_seconds_to_minute(arrs, mode="mean"), [2, 3])


def test_metadata_and_file_discovery(tmp_path):
    meta = tmp_path / "clip_metadata.csv"
    meta.write_text("Frame,Latitude,Longitude,Speed\n0,35,-80,20\n1,36,-81,bad\n", encoding="utf-8")
    df = y.read_clip_metadata_csv(str(tmp_path), "clip")
    assert list(df.columns) == ["frame", "lat", "lon", "mph"]
    assert pd.isna(df.iloc[1]["mph"])
    assert y.read_clip_metadata_csv(str(tmp_path), "missing") is None
    bad = tmp_path / "bad_metadata.csv"
    bad.write_text("x,y\n1,2\n", encoding="utf-8")
    assert y.read_clip_metadata_csv(str(tmp_path), "bad") is None

    day = tmp_path / "2026" / "08" / "13"
    day.mkdir(parents=True)
    (day / "a_YOLOv8n.csv").write_text("frame\n", encoding="utf-8")
    (day / "a.MP4").write_bytes(b"x")
    (day / "b_YOLOv8n.csv").write_text("frame\n", encoding="utf-8")
    (day / "b.MP4").write_bytes(b"")
    assert y.find_file_keys(str(day)) == ["a"]
    assert y.is_yyyymmdd_dir(str(day))
    assert not y.is_yyyymmdd_dir(str(tmp_path / "2026" / "99" / "99"))
    assert str(day) in y.walk_date_dirs(str(tmp_path))
