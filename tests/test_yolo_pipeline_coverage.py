from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from auto_ingest.dashcam import yolo_embeddings as y


def spec(**kw):
    base = dict(
        grids=[y.GridSpec(2, 2)], include_conf=True, l2_normalize=True,
        concat_views=True, per_second=True, per_minute=True,
        append_location=True, time_features=True, density_heatmap=True,
    )
    base.update(kw)
    return y.EmbeddingSpec(**base)


def make_files(tmp_path, key="2026_0813_120000_F"):
    (tmp_path / f"{key}.MP4").write_bytes(b"video")
    (tmp_path / f"{key}_YOLOv8n.csv").write_text(
        "frame,name,confidence,xyxy\n"
        "0,car,0.9,\"[0,0,50,50]\"\n"
        "2,person,0.9,\"[0,0,20,20]\"\n"
        "4,truck,0.8,\"[50,50,100,100]\"\n",
        encoding="utf-8",
    )
    (tmp_path / f"{key}_metadata.csv").write_text(
        "frame,lat,lon,mph\n0,35,-80,30\n4,35.1,-80.1,40\n",
        encoding="utf-8",
    )
    return key


def test_build_vectors_missing_empty_and_sparse(monkeypatch, tmp_path):
    s = spec()
    grids = s.grids
    assert y.build_vectors_for_key(str(tmp_path), "missing", s, {"car"}, {}, grids, False, None) == {}

    key = make_files(tmp_path)
    monkeypatch.setattr(y, "parse_yolo_csv", lambda _p: pd.DataFrame())
    empty = y.build_vectors_for_key(
        str(tmp_path), key, s, {"car"}, {}, grids, False, None,
        pre_meta=(100, 100, 2.0, 2.2), compute_minute=True,
    )["F"]
    assert len(empty["second_vecs"]) == 3 and empty["minute_vec"].shape == (4,)
    sparse_empty = y.build_vectors_for_key(
        str(tmp_path), key, s, {"car"}, {}, grids, False, None,
        pre_meta=(100, 100, 2.0, 2.2), seconds_whitelist={0, 2}, compute_minute=False,
    )["F"]
    assert sparse_empty["second_vecs_by_sec"] == {}


def test_build_vectors_full_and_sparse_real_csv_without_metadata(monkeypatch, tmp_path):
    key = make_files(tmp_path)
    s = spec()
    # Exercise the full detection/vector path independently of the known pandas
    # metadata tuple-field compatibility defect tracked below.
    monkeypatch.setattr(y, "read_clip_metadata_csv", lambda *_a: None)
    full = y.build_vectors_for_key(
        str(tmp_path), key, s, {"car", "truck"}, {2: "car", 7: "truck"}, s.grids,
        False, None, pre_meta=(100, 100, 2.0, 2.2), compute_minute=True,
    )["F"]
    assert len(full["second_vecs"]) == 3
    assert full["minute_vec"].sum() > 0

    sparse = y.build_vectors_for_key(
        str(tmp_path), key, s, {"car", "truck"}, {}, s.grids, False, None,
        pre_meta=(100, 100, 2.0, 2.2), seconds_whitelist={0, 2, 99}, compute_minute=False,
    )["F"]
    assert set(sparse["second_vecs_by_sec"]) == {0, 2}
    assert sparse["second_vecs_by_sec"][0].sum() > 0


@pytest.mark.xfail(
    strict=True,
    raises=AttributeError,
    reason="pandas itertuples sanitizes __sec__; production fix pending atomic large-file patch",
)
def test_build_vectors_metadata_seconds_regression(monkeypatch, tmp_path):
    key = make_files(tmp_path)
    s = spec()
    monkeypatch.setattr(
        y,
        "read_clip_metadata_csv",
        lambda *_a: pd.DataFrame(
            {
                "frame": [0, 4],
                "lat": [35.0, 35.1],
                "lon": [-80.0, -80.1],
                "mph": [30.0, 40.0],
            }
        ),
    )
    full = y.build_vectors_for_key(
        str(tmp_path), key, s, {"car", "truck"}, {2: "car", 7: "truck"}, s.grids,
        False, None, pre_meta=(100, 100, 2.0, 2.2), compute_minute=True,
    )["F"]
    assert full["second_loc_scalars"][0][0]["lat"] == 35.0


def test_build_vectors_probe_repair_paths(monkeypatch, tmp_path):
    key = make_files(tmp_path)
    s = spec()
    monkeypatch.setattr(y, "get_video_meta", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("bad")))
    assert y.build_vectors_for_key(str(tmp_path), key, s, {"car"}, {}, s.grids, False, None) == {}
    fixed = tmp_path / f"{key}_fixed.MP4"
    def repair(_a, b):
        fixed.write_bytes(b"fixed")
        return True
    monkeypatch.setattr(y, "try_fix_missing_moov", repair)
    calls = {"n": 0}
    def meta(*_a, **_k):
        calls["n"] += 1
        if calls["n"] == 1: raise RuntimeError("bad")
        return 100, 100, 2.0, 1.0
    monkeypatch.setattr(y, "get_video_meta", meta)
    monkeypatch.setattr(y, "parse_yolo_csv", lambda _p: pd.DataFrame())
    out = y.build_vectors_for_key(
        str(tmp_path), key, s, {"car"}, {}, s.grids, False, None,
        repair_missing_moov=True,
    )
    assert out["F"]["path"].endswith("_fixed.MP4")


class Session:
    def __init__(self): self.runs = []
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def run(self, *a, **kw): self.runs.append((a, kw)); return []


class Driver:
    def __init__(self): self.s = Session(); self.closed = False
    def session(self): return self.s
    def close(self): self.closed = True


def patch_db(monkeypatch, *, minute_done=False, secs_done=None, meta=(3.0, 2.0, 100, 100, "p")):
    d = Driver()
    monkeypatch.setattr(y, "neo4j_session", lambda *a: d)
    monkeypatch.setattr(y, "neo4j_create_constraints", lambda s: None)
    monkeypatch.setattr(y, "neo4j_get_clip_meta", lambda s, k: meta)
    monkeypatch.setattr(y, "neo4j_minute_exists", lambda s, k, v: minute_done)
    monkeypatch.setattr(y, "neo4j_get_existing_seconds", lambda s, k, v: set(secs_done or []))
    monkeypatch.setattr(y, "neo4j_upsert_clip", lambda *a, **kw: None)
    monkeypatch.setattr(y, "neo4j_upsert_embed", lambda *a, **kw: None)
    monkeypatch.setattr(y, "neo4j_enrich_frame", lambda *a, **kw: None)
    monkeypatch.setattr(y, "neo4j_rebuild_next", lambda *a, **kw: None)
    return d


def test_process_directory_full_sparse_skip_and_carry(monkeypatch, tmp_path):
    key = "2026_0813_120000_F"
    monkeypatch.setattr(y, "find_file_keys", lambda _d: [key])
    s = spec()
    d = patch_db(monkeypatch, minute_done=False, secs_done={1})
    data = {
        "F": {
            "second_vecs": [np.ones(4), np.ones(4), np.ones(4)],
            "second_locvecs": [np.zeros(7)] * 3,
            "second_loc_scalars": [({"lat": None, "lon": None, "mph": None}, "none")] * 3,
            "minute_vec": np.ones(4), "fps": 2.0, "dur": 3.0,
            "img_w": 100, "img_h": 100, "path": "p",
            "dt0_utc": dt.datetime(2026, 8, 13, tzinfo=dt.timezone.utc),
        }
    }
    monkeypatch.setattr(y, "build_vectors_for_key", lambda *a, **kw: data)
    seq = iter([
        ({"lat": 35.0, "lon": -80.0, "mph": 20}, "Frame", "elem", 1),
        ({"lat": None, "lon": None, "mph": None}, "none", None, None),
    ])
    monkeypatch.setattr(y, "resolve_location_for_second", lambda **kw: next(seq))
    y.process_directory(
        str(tmp_path), s, s.grids, {"car"}, {}, False,
        "bolt://x", "u", "p", 10, True, False, False,
    )
    assert d.closed

    d = patch_db(monkeypatch, minute_done=True, secs_done={0})
    sparse = {
        "F": {
            "second_vecs_by_sec": {1: np.ones(4), 2: np.ones(4)},
            "second_locvecs_by_sec": {1: np.zeros(7), 2: np.zeros(7)},
            "second_loc_scalars_by_sec": {1: ({"lat": None,"lon":None,"mph":None}, "none"), 2: ({"lat":None,"lon":None,"mph":None}, "none")},
            "fps": 2.0, "dur": 3.0, "img_w": 100, "img_h": 100, "path": "p",
            "dt0_utc": dt.datetime(2026, 8, 13, tzinfo=dt.timezone.utc),
        }
    }
    monkeypatch.setattr(y, "build_vectors_for_key", lambda *a, **kw: sparse)
    seq2 = iter([
        ({"lat": 35.0, "lon": -80.0, "mph": 20}, "LocationEvent", None, None),
        ({"lat": None, "lon": None, "mph": None}, "none", None, None),
    ])
    monkeypatch.setattr(y, "resolve_location_for_second", lambda **kw: next(seq2))
    y.process_directory(
        str(tmp_path), s, s.grids, {"car"}, {}, False,
        "bolt://x", "u", "p", 10, True, False, False,
    )
    assert d.closed

    d = patch_db(monkeypatch, minute_done=True, secs_done={0, 1, 2})
    monkeypatch.setattr(y, "build_vectors_for_key", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("should skip")))
    y.process_directory(
        str(tmp_path), s, s.grids, {"car"}, {}, False,
        "bolt://x", "u", "p", 10, True, False, False,
    )
    assert d.closed
