from __future__ import annotations

import datetime
from collections import namedtuple
from types import SimpleNamespace

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("moviepy")

from auto_ingest.dashcam import yolo_embeddings as ye


def test_key_time_and_numeric_utilities(monkeypatch):
    assert ye.clip_base_key("2026_0102_030405_FR") == "2026_0102_030405"
    assert ye.clip_base_key("plain") == "plain"
    assert ye.haversine_m(35, -80, 35, -80) == 0
    assert ye.plausible_fix(None, 35, -80, 10)
    assert not ye.plausible_fix(None, None, -80, 10)
    assert not ye.plausible_fix(None, 35, -80, -1)
    assert not ye.plausible_fix(None, 35, -80, 200)
    assert not ye.plausible_fix((35, -80, 0), 40, -70, 10)
    assert ye.safe_literal_eval("[1,2]") == [1, 2]
    assert ye.safe_literal_eval("not python") is None
    assert np.array_equal(ye.ensure_unit_l2(np.zeros(2)), np.zeros(2))
    normed = ye.ensure_unit_l2(np.array([3.0, 4.0]))
    assert np.linalg.norm(normed) == pytest.approx(1.0)
    assert ye.flatten_grid(np.array([[1, 2]])).dtype == np.float32
    assert ye.bbox_overlap_area(0, 0, 2, 2, 1, 1, 3, 3) == 1.0
    assert ye.bbox_overlap_area(0, 0, 1, 1, 2, 2, 3, 3) == 0.0
    assert ye.second_from_frame(61, 30) == 2
    assert ye.second_from_frame(None, 30) is None
    assert ye.second_from_frame(float("nan"), 30) is None
    assert ye.parse_key_datetime("2026_0102_030405_F") == datetime.datetime(
        2026, 1, 2, 3, 4, 5, tzinfo=datetime.timezone.utc
    )
    assert ye.parse_key_datetime("bad") is None
    assert ye.parse_key_datetime("2026_9999_999999") is None

    monkeypatch.setattr(ye.np.random, "randint", lambda low, high, size: np.arange(size) % high)
    fallback = ye.assign_seconds(pd.DataFrame({"name": ["a", "b"]}), 30, 2, "k")
    assert list(fallback["__sec__"].astype(int)) == [0, 1]


def test_assign_seconds_all_time_sources():
    frame = ye.assign_seconds(pd.DataFrame({"frame": [0, 30, 90]}), 30, 3, "frame")
    assert list(frame["__sec__"].astype(int)) == [0, 1]
    t = ye.assign_seconds(pd.DataFrame({"t": [0.2, 1.9]}), 30, 3, "t")
    assert list(t["__sec__"].astype(int)) == [0, 1]
    time_df = ye.assign_seconds(pd.DataFrame({"time": [1.2]}), 30, 3, "time")
    assert int(time_df.iloc[0]["__sec__"]) == 1
    stamp = ye.assign_seconds(pd.DataFrame({"timestamp": [2.7]}), 30, 4, "timestamp")
    assert int(stamp.iloc[0]["__sec__"]) == 2


def test_grid_accumulation_area_and_density():
    spec = ye.GridSpec(2, 2)
    grid = np.zeros((2, 2), dtype=np.float32)
    ye.add_bbox_to_grid(grid, (0, 0, 100, 100), 100, 100, 2.0, spec, density=False)
    assert grid.sum() == pytest.approx(20_000.0)
    density = np.zeros((2, 2), dtype=np.float32)
    ye.add_bbox_to_grid(density, (0, 0, 100, 100), 100, 100, 2.0, spec, density=True)
    assert density.sum() == pytest.approx(2.0)
    before = density.copy()
    ye.add_bbox_to_grid(density, (10, 10, 10, 20), 100, 100, 1, spec)
    assert np.array_equal(density, before)
    ye.add_bbox_to_grid(density, (-50, -50, 20, 20), 100, 100, 1, spec)
    assert density[0, 0] > before[0, 0]


def test_csv_split_list_percent_and_parser(tmp_path):
    assert ye._split_ignoring_brackets("a,[1,2,3,4],c") == ["a", "[1,2,3,4]", "c"]
    assert ye._maybe_list4([1, 2, 3, 4]) == [1.0, 2.0, 3.0, 4.0]
    assert ye._maybe_list4("[1 2 3 4]") == [1.0, 2.0, 3.0, 4.0]
    assert ye._maybe_list4("1,2,3,4") == [1.0, 2.0, 3.0, 4.0]
    assert ye._maybe_list4("no") is None
    assert ye._percent_to_float01("79.5%") == pytest.approx(0.795)
    assert ye._percent_to_float01("0.5") == 0.5
    assert np.isnan(ye._percent_to_float01("bad"))

    path = tmp_path / "yolo.csv"
    path.write_text(
        "frame,name,confidence,xyxy,cls\n"
        "0,car,80%,[0,0,10,10],2\n"
        "30,truck,bad,[1,2,11,12],7\n",
        encoding="utf-8",
    )
    df = ye.parse_yolo_csv(str(path))
    assert list(df["name"]) == ["car", "truck"]
    assert df.iloc[0]["confidence"] == pytest.approx(0.8)
    assert df.iloc[1]["confidence"] == 1.0
    assert df.iloc[0]["xyxy"] == [0.0, 0.0, 10.0, 10.0]
    assert "class" in df.columns

    empty = tmp_path / "empty.csv"
    empty.write_text("frame,name\n", encoding="utf-8")
    assert ye.parse_yolo_csv(str(empty)).empty

    classification = tmp_path / "classification.csv"
    classification.write_text(
        "frame,classification,xywhn\n0,75%,[0.5,0.5,0.2,0.4]\n",
        encoding="utf-8",
    )
    df2 = ye.parse_yolo_csv(str(classification))
    assert df2.iloc[0]["confidence"] == pytest.approx(0.75)
    assert df2.iloc[0]["name"] == ""


def test_row_conversion_and_bbox_forms():
    NT = namedtuple("NT", "a b")
    assert ye.row_to_dict({"a": 1}) == {"a": 1}
    assert ye.row_to_dict(NT(1, 2)) == {"a": 1, "b": 2}
    assert ye.row_to_dict(pd.Series({"a": 1})) == {"a": 1}
    assert ye.row_to_dict([1, 2], ["a", "b"]) == {"a": 1, "b": 2}
    assert ye.row_to_dict(object()) == {}

    assert ye.to_xyxy_abs({"xyxy": [1, 2, 11, 12]}, 100, 50) == (1, 2, 11, 12)
    assert ye.to_xyxy_abs({"xyxyn": [0.1, 0.2, 0.5, 0.6]}, 100, 50) == (10, 10, 50, 30)
    assert ye.to_xyxy_abs({"xywh": [50, 25, 20, 10]}, 100, 50) == (40, 20, 60, 30)
    assert ye.to_xyxy_abs({"xywhn": [0.5, 0.5, 0.2, 0.4]}, 100, 50) == (40, 15, 60, 35)
    assert ye.to_xyxy_abs({"x1": 1, "y1": 2, "x2": 3, "y2": 4}, 100, 50) == (1, 2, 3, 4)
    assert ye.to_xyxy_abs({"xc": 0.5, "yc": 0.5, "w": 0.2, "h": 0.2}, 100, 50) == (40, 20, 60, 30)
    assert ye.to_xyxy_abs({}, 100, 50) is None


def test_detection_filtering():
    keep = {"car", "truck", "motorbike"}
    assert ye.keep_detection({"name": "car"}, keep, ye.DEFAULT_CLASS_ID_MAP)
    assert ye.keep_detection({"name": "motorcycle"}, keep, ye.DEFAULT_CLASS_ID_MAP)
    assert ye.keep_detection({"name": "", "class": 7}, keep, ye.DEFAULT_CLASS_ID_MAP)
    assert not ye.keep_detection({"name": "person", "class": 0}, keep, ye.DEFAULT_CLASS_ID_MAP)
    assert not ye.keep_detection({"name": "", "class": "bad"}, keep, ye.DEFAULT_CLASS_ID_MAP)


def test_location_feature_and_choose_location():
    dt = datetime.datetime(2026, 1, 1, 12, 0, tzinfo=datetime.timezone.utc)
    vec, scalars = ye.location_feature(35, -80, 40, dt, include_time=True)
    assert len(vec) == 7
    assert scalars == {"lat": 35.0, "lon": -80.0, "mph": 40.0}
    vec2, scalars2 = ye.location_feature(None, None, None, None, include_time=False)
    assert np.array_equal(vec2, np.zeros(7, dtype=np.float32))
    assert scalars2["lat"] is None

    primary = {"lat": 35, "lon": -80, "mph": 10}
    fallback = {"lat": 35.1, "lon": -80, "mph": 10}
    assert ye.choose_location(primary, fallback, None, None)[1] == "LocationEvent"
    bad_primary = {"lat": None, "lon": None, "mph": 10}
    assert ye.choose_location(bad_primary, fallback, None, None)[1] == "PhoneLog"
    meta = {"lat": 35.10001, "lon": -80, "mph": 5}
    assert ye.choose_location(bad_primary, fallback, meta, None)[1] == "PhoneLog"
    bad_fallback = {"lat": None, "lon": None, "mph": None}
    assert ye.choose_location(bad_primary, bad_fallback, meta, None)[1] == "metadata_csv"
    none, source = ye.choose_location(bad_primary, bad_fallback, {"lat": None, "lon": None}, None)
    assert source == "none"
    assert none["lat"] is None


def test_find_best_location_queries():
    class Result:
        def __init__(self, rec):
            self.rec = rec

        def single(self):
            return self.rec

    class Session:
        def __init__(self):
            self.i = 0

        def run(self, query, **kwargs):
            self.i += 1
            if query == ye.NQ_FIND_NEAREST_LOCEVENT:
                values = [
                    {"seconds": 9, "lat": 35, "lon": -80},
                    {"seconds": 2, "lat": 35, "lon": -80},
                    None,
                ]
                return Result(values[self.i - 1])
            return Result({"seconds": 1, "lat": 35, "lon": -80})

    dt = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
    primary, fallback = ye.find_best_locevent_then_phonelog(Session(), dt, 10, None)
    assert primary["seconds"] == 2
    assert primary["_offset"] == 3600
    assert fallback["seconds"] == 1


def test_resolve_location_priority(monkeypatch):
    dt = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
    frame = {"frame": 31, "lat": 35, "lon": -80, "mph": 5, "elem_id": "f"}
    monkeypatch.setattr(ye, "neo4j_find_nearest_frame", lambda *a, **k: frame)
    chosen, source, elem, delta = ye.resolve_location_for_second(
        object(), "2026_0101_000000_F", 30, 1, dt, 10, None, None
    )
    assert source == "Frame" and elem == "f" and delta == pytest.approx(1 / 30)
    assert chosen["lat"] == 35

    monkeypatch.setattr(ye, "neo4j_find_nearest_frame", lambda *a, **k: None)
    monkeypatch.setattr(
        ye,
        "find_best_locevent_then_phonelog",
        lambda *a, **k: (
            {"lat": 35, "lon": -80, "mph": 1, "elem_id": "l", "seconds": 3},
            {"lat": 36, "lon": -80, "mph": 1, "elem_id": "p", "seconds": 4},
        ),
    )
    _, source, elem, delta = ye.resolve_location_for_second(
        object(), "k", 30, 1, dt, 10, None, None
    )
    assert (source, elem, delta) == ("LocationEvent", "l", 3.0)

    monkeypatch.setattr(
        ye,
        "find_best_locevent_then_phonelog",
        lambda *a, **k: (None, None),
    )
    _, source, elem, delta = ye.resolve_location_for_second(
        object(), "k", 30, 1, dt, 10, {"lat": 35, "lon": -80, "mph": 1}, None
    )
    assert (source, elem, delta) == ("metadata_csv", None, None)
    chosen, source, _, _ = ye.resolve_location_for_second(
        object(), "k", 30, 1, dt, 10, None, None
    )
    assert source == "none" and chosen["lat"] is None
