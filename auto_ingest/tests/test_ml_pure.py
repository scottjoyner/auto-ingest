"""
Pure-logic tests for ML subpackages, exercising ONLY functions that need no
real inference at import or call time.

Technique: each ML-heavy module is imported under a `unittest.mock.patch` on
`sys.modules` that stubs torch/torchaudio/transformers/ultralytics/moviepy/pandas
etc. so module-load does NOT pull the heavy stack. We then call only the
deterministic, ML-free helpers in those modules. No real inference ever runs.
"""
import importlib
import sys
import unittest.mock as mock
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")


# ---------------------------------------------------------------------------
# Helpers to perform the mock-import
# ---------------------------------------------------------------------------
def _mock_import_module(module_name, extra_stubs=None):
    """Import `module_name` while stubbing heavy ML deps in sys.modules.

    A MagicMock covers every attribute access, so submodule imports like
    `from torch.nn.functional import normalize` resolve to a dummy callable.
    """
    stubs = {
        "torch": mock.MagicMock(),
        "torchaudio": mock.MagicMock(),
        "torch.nn": mock.MagicMock(),
        "torch.nn.functional": mock.MagicMock(),
        "transformers": mock.MagicMock(),
        "soundfile": mock.MagicMock(),
        "faiss": mock.MagicMock(),
        "speechbrain": mock.MagicMock(),
        "speechbrain.pretrained": mock.MagicMock(),
        "pyannote": mock.MagicMock(),
        "pyannote.audio": mock.MagicMock(),
        "ultralytics": mock.MagicMock(),
        "ultralytics.utils": mock.MagicMock(),
        "ultralytics.utils.plotting": mock.MagicMock(),
        "cv2": mock.MagicMock(),
        "moviepy": mock.MagicMock(),
        "moviepy.editor": mock.MagicMock(),
        "PIL": mock.MagicMock(),
        "matplotlib": mock.MagicMock(),
    }
    if extra_stubs:
        stubs.update(extra_stubs)
    with mock.patch.dict(sys.modules, stubs):
        return importlib.import_module(module_name)


# ---------------------------------------------------------------------------
# 1. auto_ingest.diarize.link_global_speakers
#    Imports torch/pyannote/speechbrain/faiss at TOP -> mock-import.
# ---------------------------------------------------------------------------
def test_link_global_speakers_pure():
    L = _mock_import_module("auto_ingest.diarize.link_global_speakers")

    # --- _norm_stem: sidecar stripping + R-suffix normalize ---
    assert L._norm_stem("2025_0202_171732_medium_transcription_speakers") == "2025_0202_171732"
    assert L._norm_stem("2025_0202_171732_F") == "2025_0202_171732"
    assert L._norm_stem("clip.wav") == "clip"
    assert L._norm_stem("a_large_transcription") == "a"

    # --- stable_id: deterministic, order-sensitive md5 ---
    a = L.stable_id("x", "y")
    b = L.stable_id("y", "x")
    assert a != b
    assert a == L.stable_id("x", "y")

    # --- cosine / unit: pure numpy math ---
    v = np.array([3.0, 4.0], dtype=np.float32)
    assert np.isclose(L.cosine(v, v), 1.0)
    assert np.isclose(L.cosine(v, -v), -1.0)
    assert np.isclose(L.cosine(v, np.array([0.0, 0.0], dtype=np.float32)), 0.0)
    u = L.unit(v)
    assert np.isclose(float(np.linalg.norm(u)), 1.0)
    # unit of zero vector returns itself (no division by zero)
    z = np.zeros(3, dtype=np.float32)
    assert np.array_equal(L.unit(z), z)

    # --- cap_snips_per_file: per-file cap + global total cap ---
    items = [(f"f{i%2}", 0.0, 1.0, "t", 1.0) for i in range(10)]
    rng = __import__("random").Random(0)
    sel = L.cap_snips_per_file(items, max_per_file=3, max_total=4, rng=rng)
    assert len(sel) <= 4
    # at most 3 per file
    from collections import Counter
    c = Counter(s[0] for s in sel)
    assert all(v <= 3 for v in c.values())

    # --- _regex_ok: pure regex helper ---
    rx = __import__("re").compile("scott", __import__("re").I)
    assert L._regex_ok("Scott here", rx) is True
    assert L._regex_ok(None, rx) is False
    assert L._regex_ok("other", rx) is False

    # --- load_audio_cache / save_audio_cache (JSON only, no ML) ---
    import tempfile
    tmp = Path(tempfile.mkdtemp()) / "cache.json"
    assert L.load_audio_cache(None, False) == {}
    assert L.load_audio_cache(tmp, False) == {}
    L.save_audio_cache(tmp, {"k": "v"})
    assert L.load_audio_cache(tmp, False) == {"k": "v"}
    assert L.load_audio_cache(tmp, refresh=True) == {}

    # --- parse_args: --exclude-non-speech / --no-exclude-non-speech ---
    args = L.parse_args(["prog", "--exclude-non-speech"])
    assert args.exclude_non_speech is True
    args = L.parse_args(["prog", "--no-exclude-non-speech"])
    assert args.exclude_non_speech is False
    # defaults round-trip
    args = L.parse_args(["prog"])
    assert args.min_seg == L.DEFAULT_MIN_SEG
    assert args.thresh == L.DEFAULT_THRESH


# ---------------------------------------------------------------------------
# 2. auto_ingest.ingest.transcripts
#    Imports torch/transformers/neo4j at TOP -> mock-import. Note: module-level
#    `AutoTokenizer.from_pretrained(...)` runs at import -> stubbed by MagicMock.
# ---------------------------------------------------------------------------
def test_transcripts_pure():
    T = _mock_import_module("auto_ingest.ingest.transcripts")

    # --- _chunks: classic batching ---
    assert list(T._chunks([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]
    assert list(T._chunks([], 3)) == []

    # --- stable_id deterministic ---
    assert T.stable_id("a") == T.stable_id("a")
    assert T.stable_id("a") != T.stable_id("b")

    # --- _parse_any_iso_or_epoch: ISO, epoch seconds, epoch ms ---
    from datetime import datetime, timezone
    iso_dt = T._parse_any_iso_or_epoch("2024-01-02T03:04:05+00:00")
    assert iso_dt is not None
    epoch = T._parse_any_iso_or_epoch(1700000000)
    assert epoch == datetime.fromtimestamp(1700000000, tz=timezone.utc)
    ms = T._parse_any_iso_or_epoch(1700000000000)
    assert ms == datetime.fromtimestamp(1700000000, tz=timezone.utc)
    assert T._parse_any_iso_or_epoch("garbage") is None
    assert T._parse_any_iso_or_epoch(None) is None

    # --- canonicalize_key / file_key_from_name: sidecar stripping ---
    assert T.file_key_from_name("2025_0202_171732_medium_transcription.txt") == "2025_0202_171732"
    assert T.file_key_from_name("x_speakers.rttm") == "x"
    assert T.file_key_from_name("x_metadata.csv") == "x"
    key = T.canonicalize_key("2025_0202_171732", "whatever")
    assert key == "2025_0202_171732"

    # --- overlap: pure interval math ---
    assert T.overlap(0, 10, 5, 20) == 5.0
    assert T.overlap(0, 5, 10, 20) == 0.0
    assert T.overlap(0, 10, 2, 8) == 6.0

    # --- model_rank: preference ordering ---
    assert T.model_rank("large-v3") < T.model_rank("tiny")
    assert T.model_rank("") == 10_000

    # --- validate_and_clean_segments: neg dur + nonfinite + empty text ---
    segs = [
        {"start": 1.0, "end": 0.5, "text": "a"},
        {"start": float("nan"), "end": 1.0, "text": "b"},
        {"start": 0.0, "end": 1.0, "text": ""},
        {"start": 0.0, "end": 1.0, "text": "ok"},
    ]
    fixed, stats = T.validate_and_clean_segments("k", segs)
    assert stats["neg_dur"] == 1
    assert stats["nonfinite"] == 1
    assert stats["empty_txt"] == 1
    assert stats["kept"] == 2  # only the two valid ones

    # --- load_rttm: pure RTTM parsing ---
    import tempfile
    import textwrap
    rttm_path = Path(tempfile.mkdtemp()) / "x_speakers.rttm"
    rttm_path.write_text(textwrap.dedent(
        """\
        # comment
        SPEAKER x 1 0.0 2.5 <NA> <NA> spk1 <NA>
        SPEAKER x 1 3.0 1.0 <NA> <NA> UNKNOWN <NA>
        """
    ), encoding="utf-8")
    intervals = T.load_rttm(str(rttm_path))
    assert intervals[0] == (0.0, 2.5, "spk1")
    assert intervals[1] == (3.0, 4.0, "UNKNOWN")

    # --- aggregate_entities: bucketing ---
    ents = [
        {"text": "Bob", "label": "Person", "score": 0.9, "start": 0, "end": 1},
        {"text": "Bob", "label": "Person", "score": 0.7, "start": 2, "end": 3},
    ]
    agg = T.aggregate_entities(ents)
    assert len(agg) == 1
    assert agg[0]["count"] == 2
    assert np.isclose(agg[0]["avg_score"], 0.8)

    # --- _parse_bbox / _in_bbox / _norm_lat_lon: pure geo helpers ---
    bb = T._parse_bbox("30,40,31,41")
    assert bb == (30, 40, 31, 41)
    assert T._in_bbox(30.5, 40.5, bb) is True
    assert T._in_bbox(50, 50, bb) is False
    assert T._parse_bbox("bad") is None
    lat, lon = T._norm_lat_lon("35", " -120 ")
    assert lat == 35.0 and lon == -120.0
    lat, lon = T._norm_lat_lon("999", "0")
    assert lat is None  # out of range -> None

    # --- _repair_latlon: west-fix + swap ---
    lat, lon, flags = T._repair_latlon(35.0, 120.0, bbox=None,
                                       lon_auto_west=True, allow_swap=False)
    assert "FIXED_WEST" in flags and lon == -120.0
    lat, lon, flags = T._repair_latlon(120.0, 35.0, bbox=None,
                                       lon_auto_west=False, allow_swap=True)
    # when no bbox and allow_swap but not in bbox: swapping 120/35 -> still out,
    # ensure it doesn't crash and returns a tuple
    assert isinstance(flags, list)


# ---------------------------------------------------------------------------
# 3. auto_ingest.dashcam.yolo_embeddings
#    Imports pandas/moviepy/neo4j at TOP -> mock-import (pandas kept real for
#    parse_yolo_csv, so we only stub moviepy/neo4j).
# ---------------------------------------------------------------------------
def test_yolo_embeddings_pure():
    # Keep pandas REAL (parser needs it); stub only moviepy/neo4j.
    stubs = {
        "moviepy": mock.MagicMock(),
        "moviepy.editor": mock.MagicMock(),
        "neo4j": mock.MagicMock(),
        "neo4j.exceptions": mock.MagicMock(),
    }
    with mock.patch.dict(sys.modules, stubs):
        Y = importlib.import_module("auto_ingest.dashcam.yolo_embeddings")

    # --- _split_ignoring_brackets: split on commas outside [...] ---
    parts = Y._split_ignoring_brackets("a, b, [1, 2, 3], c")
    assert parts == ["a", "b", "[1, 2, 3]", "c"]

    # --- _maybe_list4: ast list, bracketed, comma, fallback ---
    assert Y._maybe_list4("[1, 2, 3, 4]") == [1.0, 2.0, 3.0, 4.0]
    assert Y._maybe_list4("1, 2, 3, 4") == [1.0, 2.0, 3.0, 4.0]
    assert Y._maybe_list4("[1040 500 1150 620]") == [1040.0, 500.0, 1150.0, 620.0]
    assert Y._maybe_list4("nope") is None
    assert Y._maybe_list4([1, 2, 3, 4]) == [1.0, 2.0, 3.0, 4.0]

    # --- _percent_to_float01 ---
    assert np.isclose(Y._percent_to_float01("79.48%"), 0.7948)
    assert np.isclose(Y._percent_to_float01("0.85"), 0.85)
    assert np.isnan(Y._percent_to_float01("??"))

    # --- clip_base_key: strip _F/_R/_FR ---
    assert Y.clip_base_key("2025_0202_171732_F") == "2025_0202_171732"
    assert Y.clip_base_key("2025_0202_171732_FR") == "2025_0202_171732"
    assert Y.clip_base_key("2025_0202_171732") == "2025_0202_171732"

    # --- parse_key_datetime: key -> datetime ---
    dt = Y.parse_key_datetime("2025_0202_171732")
    assert dt is not None and dt.year == 2025 and dt.second == 32
    assert Y.parse_key_datetime("notakey") is None

    # --- keep_detection: name + id_map matching ---
    keep = {"car", "truck"}
    id_map = {2: "car", 7: "truck"}
    assert Y.keep_detection({"name": "car"}, keep, id_map) is True
    assert Y.keep_detection({"name": "motorcycle"}, keep, id_map) is True  # -> motorbike
    assert Y.keep_detection({"name": "cat"}, keep, id_map) is False
    assert Y.keep_detection({"class": 7}, keep, id_map) is True

    # --- safe_literal_eval ---
    assert Y.safe_literal_eval("[1,2]") == [1, 2]
    assert Y.safe_literal_eval("not a list") is None

    # --- to_xyxy_abs: vector + scalar conversions ---
    row = {"xyxy": [10, 20, 110, 120]}
    assert Y.to_xyxy_abs(row, 200, 200) == (10, 20, 110, 120)
    row = {"xywh": [50, 50, 100, 100]}
    assert Y.to_xyxy_abs(row, 200, 200) == (0, 0, 100, 100)
    row = {"x1": 1, "y1": 2, "x2": 3, "y2": 4}
    assert Y.to_xyxy_abs(row, 200, 200) == (1, 2, 3, 4)

    # --- bbox_overlap_area ---
    assert np.isclose(Y.bbox_overlap_area(0, 0, 10, 10, 5, 5, 15, 15), 25.0)
    assert Y.bbox_overlap_area(0, 0, 5, 5, 10, 10, 15, 15) == 0.0

    # --- plausible_fix ---
    assert Y.plausible_fix(None, 35.0, -120.0, 30.0) is True
    assert Y.plausible_fix(None, 35.0, -120.0, 999.0) is False  # bad mph
    assert Y.plausible_fix((35.0, -120.0), 35.0, -120.0, 30.0) is True

    # --- second_from_frame ---
    assert Y.second_from_frame(120, 30) == 4
    assert Y.second_from_frame(None, 30) is None

    # --- location_feature ---
    vec, scalars = Y.location_feature(35.0, -120.0, 60.0,
                                      datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc))
    assert vec.shape == (7,)
    assert scalars["lat"] == 35.0 and scalars["mph"] == 60.0

    # --- ensure_unit_l2 / flatten_grid ---
    v = np.array([3.0, 4.0, 0.0], dtype=np.float32)
    assert np.isclose(float(np.linalg.norm(Y.ensure_unit_l2(v))), 1.0)
    g = np.array([[1, 2], [3, 4]], dtype=np.float32)
    assert np.array_equal(Y.flatten_grid(g), [1, 2, 3, 4])

    # --- add_bbox_to_grid: accumulate into grid ---
    grid = np.zeros((2, 2), dtype=np.float32)
    Y.add_bbox_to_grid(grid, (0, 0, 100, 100), 100, 100, 1.0,
                       Y.GridSpec(2, 2), density=False)
    assert grid.sum() > 0  # overlapped pixel area added

    # --- parse_yolo_csv: end-to-end pure CSV parse (no moviepy/neo4j needed) ---
    import tempfile
    import textwrap
    csv_path = Path(tempfile.mkdtemp()) / "clip_YOLOv8n.csv"
    csv_path.write_text(textwrap.dedent(
        """\
        Key,vehicle_id,confidence,classification,xywh,xyxy,Frame
        2025_0202_171732_F_1, id1, 95.0%, car, [10 20 30 40], [10, 20, 40, 60], 1
        2025_0202_171732_F_1, id2, 80%, truck, [50 50 20 20], [50, 50, 70, 70], 2
        """
    ), encoding="utf-8")
    df = Y.parse_yolo_csv(str(csv_path))
    assert len(df) == 2
    assert df["name"].tolist() == ["car", "truck"]
    assert np.isclose(df["confidence"].tolist()[0], 0.95)
    assert df["xyxy"].tolist()[0] == [10.0, 20.0, 40.0, 60.0]
    assert df["frame"].tolist() == [1, 2]


# ---------------------------------------------------------------------------
# 4. yolo_vehicle_detction.py (repo root, flat)
#    Imports ultralytics at TOP and runs ultralytics.checks() / list_directories
#    at module load -> mock-import; we only test the pure file-listing helpers.
# ---------------------------------------------------------------------------
def test_yolo_vehicle_detction_pure():
    V = _mock_import_module("yolo_vehicle_detction")

    # --- list_files: discovers _R/_F keys and skips ones with CSV sidecar ---
    import os
    import tempfile
    d = tempfile.mkdtemp()
    for name in ["CLIP_A_R.MP4", "CLIP_A_F.MP4", "CLIP_B_R.MP4",
                 "CLIP_B_F.MP4", "CLIP_B_YOLOv8n.csv", "IGNORE.MP4"]:
        open(os.path.join(d, name), "w").close()
    keys = V.list_files(d)
    assert "CLIP_A" in keys
    assert "CLIP_B" not in keys  # has CSV already -> excluded
    assert all("_" not in k or not k.endswith((".MP4",)) for k in keys)

    # --- is_valid_date_structure: YYYY/MM/DD ---
    assert V.is_valid_date_structure("2024/06/23") is True
    assert V.is_valid_date_structure("2024-06-23") is False
    assert V.is_valid_date_structure("not/a/date") is False
