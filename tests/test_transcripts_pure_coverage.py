from __future__ import annotations

import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


class _FakeTokenizer:
    @classmethod
    def from_pretrained(cls, _name):
        return cls()

    def __call__(self, text, *args, **kwargs):
        if isinstance(text, list):
            return {"attention_mask": np.ones((len(text), 1), dtype=np.int64)}
        return SimpleNamespace(input_ids=str(text).split())


class _FakeModel:
    @classmethod
    def from_pretrained(cls, _name):
        return cls()

    def to(self, _device):
        return self

    def eval(self):
        return self


@pytest.fixture()
def tr(monkeypatch, tmp_path):
    import auto_ingest.backend as backend
    import auto_ingest_config as cfg

    monkeypatch.setattr(backend, "_BACKEND", "onnx")
    monkeypatch.setattr(cfg, "get_fileserver_path", lambda suffix="": str(tmp_path / suffix))
    monkeypatch.setattr(
        cfg,
        "get_neo4j_env",
        lambda: ("bolt://unused", "neo4j", "unused", "neo4j"),
    )
    fake_torch = ModuleType("torch")
    fake_torch.Tensor = object
    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoTokenizer = _FakeTokenizer
    fake_transformers.AutoModel = _FakeModel
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    sys.modules.pop("auto_ingest.ingest.transcripts", None)
    module = importlib.import_module("auto_ingest.ingest.transcripts")
    return module


def test_stage_stats_and_timing(tr, monkeypatch):
    stats = tr.StageStats("x", alpha=0.5)
    assert stats.avg == 0.0
    stats.update(2.0)
    stats.update(4.0)
    assert stats.count == 2
    assert stats.total == 6.0
    assert stats.avg == 3.0
    assert stats.ema == 3.0
    assert "x: n=2" in stats.summary()

    ticks = iter([10.0, 12.5, 20.0, 21.0])
    monkeypatch.setattr(tr.time, "perf_counter", lambda: next(ticks))
    ok_stats = tr.StageStats("ok")
    with tr.TimedStage(ok_stats, "detail") as stage:
        pass
    assert stage.dt == 2.5
    assert ok_stats.count == 1

    failed_stats = tr.StageStats("failed")
    with pytest.raises(RuntimeError):
        with tr.TimedStage(failed_stats):
            raise RuntimeError("boom")
    assert failed_stats.count == 0


def test_identity_time_and_filename_helpers(tr):
    assert list(tr._chunks([1, 2, 3], 0)) == [[1], [2], [3]]
    assert list(tr._chunks([1, 2, 3, 4], 3)) == [[1, 2, 3], [4]]
    assert tr.stable_id("a", "b") == tr.stable_id("a", "b")
    assert tr.stable_id("a") != tr.stable_id("b")

    utc = tr._parse_any_iso_or_epoch(1_700_000_000)
    assert tr._parse_any_iso_or_epoch(1_700_000_000_000) == utc
    assert tr._parse_any_iso_or_epoch("2026-01-02T03:04:05Z").year == 2026
    assert utc.tzinfo == timezone.utc
    assert tr._parse_any_iso_or_epoch(None) is None
    assert tr._parse_any_iso_or_epoch("garbage") is None
    assert tr.iso(None) is None
    assert tr.iso(datetime(2026, 1, 1, tzinfo=timezone.utc)).startswith("2026-01-01")

    names = [
        "20260102030405",
        "2026_0102_030405",
        "20260102_030405",
        "2026-01-02_03-04-05",
        "2026_01_02_03_04_05",
        "/root/2026/01/02/foo_030405.wav",
    ]
    assert all(tr.parse_key_datetime_utc_from_string(name) for name in names)
    assert tr.parse_key_datetime_utc_from_string("nothing") is None
    assert tr.canonicalize_key("20260102030405", "/x") == "2026_0102_080405"
    assert tr.canonicalize_key(" weird key! ", "/no/date") == "weird_key"
    assert tr.file_key_from_name("abc_large-v3_transcription.txt") == "abc"
    assert tr.file_key_from_name("abc_transcription_entities.csv") == "abc"
    assert tr.file_key_from_name("abc_speakers.rttm") == "abc"
    assert tr.file_key_from_name("abc_metadata.csv") == "abc"


def test_token_chunk_and_segment_embeddings(tr, monkeypatch):
    tokenizer = _FakeTokenizer()
    assert tr.chunk_by_tokens("", tokenizer) == []
    chunks = tr.chunk_by_tokens(
        "one two three four five", tokenizer, max_tokens=2, overlap=1
    )
    assert chunks and all(len(chunk.split()) <= 2 for chunk in chunks)

    monkeypatch.setattr(tr, "EMBED_DIM", 3)
    assert tr.embed_long_text_via_segments([], 2) == [0.0, 0.0, 0.0]
    monkeypatch.setattr(
        tr,
        "embed_texts",
        lambda texts, batch_size, **kw: [[3.0, 0, 0], [0, 4.0, 0]],
    )
    vector = tr.embed_long_text_via_segments(["a", "b"], 2)
    assert np.linalg.norm(vector) == pytest.approx(1.0)
    assert tr.transcript_embedding_v2_from_segments([]) == [0.0, 0.0, 0.0]
    weighted = tr.transcript_embedding_v2_from_segments(
        [
            {"start": 0, "end": 1, "embedding": [1.0, 0, 0]},
            {"start": 1, "end": 4, "embedding": [0, 1.0, 0]},
            {"start": 4, "end": 5, "embedding": None},
        ]
    )
    assert np.linalg.norm(weighted) == pytest.approx(1.0)
    assert weighted[1] > weighted[0]


def test_entities_words_utterances_and_overlap(tr):
    entities = tr.aggregate_entities(
        [
            {"text": " Alice ", "label": "Person", "score": 0.8, "start": 1, "end": 2},
            {"text": "Alice", "label": "Person", "score": 1.0, "start": 3, "end": 4},
            {"text": "Paris", "label": "Place"},
        ]
    )
    alice = next(entity for entity in entities if entity["text"] == "Alice")
    assert alice["count"] == 2
    assert alice["avg_score"] == pytest.approx(0.9)
    assert tr.overlap(0, 2, 1, 3) == 1
    assert tr.overlap(0, 1, 2, 3) == 0

    segments = [
        {
            "id": "s1",
            "start": 0,
            "end": 2,
            "text": "hello world",
            "words": [
                {"word": " world ", "start": 1, "end": 1.5},
                {"word": "hello", "start": 0.1, "end": 0.6},
                {"word": ""},
            ],
        },
        {"id": "s2", "start": 3, "end": 4, "text": "later", "words": []},
    ]
    assert [word["text"] for word in tr.words_from_segments(segments)] == [
        "hello",
        "world",
    ]
    rttm = [(0, 0.9, "A"), (0.9, 2.1, "B")]
    utterances = tr.utterances_from_rttm_with_words(rttm, segments)
    assert [item["speaker_label"] for item in utterances] == ["A", "B"]
    assert tr.utterances_from_rttm_with_words(rttm, [{"words": []}]) == []
    dominant = tr.utterances_from_rttm_dominant_segment(rttm, segments)
    assert dominant[0]["speaker_label"] == "B"
    assert dominant[1]["speaker_label"] == "UNKNOWN"

    speaker_map = {"A": {"id": "a"}, "B": {"id": "b"}, "UNKNOWN": {"id": "u"}}
    best, edges = tr.compute_segment_speaker_overlaps(rttm, segments, speaker_map, 0.1)
    assert best["s1"]["speaker_id"] == "b"
    assert {edge["speaker_id"] for edge in edges} == {"a", "b"}


def test_json_csv_entity_and_rttm_loaders(tr, tmp_path):
    json_path = tmp_path / "sample.txt"
    json_path.write_text(
        json.dumps(
            {
                "text": " hello ",
                "segments": [{"start": 0, "end": 1}],
                "language": "en",
                "metadata": {"started_at": "2026-01-01T00:00:00Z"},
                "ended_at": 1_767_225_601,
            }
        ),
        encoding="utf-8",
    )
    loaded = tr.load_transcription_json_txt(str(json_path))
    assert loaded["text"] == "hello"
    assert loaded["file_started_at"].year == 2026
    bad_json = tmp_path / "bad.txt"
    bad_json.write_text("{", encoding="utf-8")
    assert tr.load_transcription_json_txt(str(bad_json)) is None
    assert tr.load_transcription_json_txt(str(tmp_path / "missing.txt")) is None

    csv_path = tmp_path / "transcription.csv"
    csv_path.write_text(
        "Text,StartTime,EndTime,AbsoluteStart,AbsoluteEnd\n"
        "hello,0,1,2026-01-01T00:00:00Z,2026-01-01T00:00:01Z\n"
        "world,bad,2,2026-01-01T00:00:01Z,2026-01-01T00:00:02Z\n",
        encoding="utf-8",
    )
    csv_doc = tr.load_transcription_csv(str(csv_path))
    assert csv_doc["text"] == "hello world"
    assert len(csv_doc["segments"]) == 2
    assert csv_doc["segments"][1]["start"] == 0.0
    assert tr.load_transcription_csv(str(tmp_path / "missing.csv")) is None

    entity_path = tmp_path / "entities.csv"
    entity_path.write_text(
        "Text,Label,Score,StartTime,EndTime\n"
        "Alice,Person,0.9,1,2\nBob,Person,bad,3,4\n",
        encoding="utf-8",
    )
    entity_rows = tr.load_entities_csv(str(entity_path))
    assert entity_rows[0]["text"] == "Alice"
    assert entity_rows[1]["score"] == 0.0
    assert tr.load_entities_csv(str(tmp_path / "none.csv")) == []

    rttm_path = tmp_path / "sample.rttm"
    rttm_path.write_text(
        "# comment\n"
        "bad line\n"
        "SPEAKER f 1 2.0 1.5 <NA> <NA> SPK2 <NA>\n"
        "SPEAKER f 1 0.0 1.0 <NA> <NA> UNKNOWN <NA>\n"
        "SPEAKER f 1 9.0 -1 <NA> <NA> BAD <NA>\n",
        encoding="utf-8",
    )
    assert tr.load_rttm(str(rttm_path)) == [
        (0.0, 1.0, "UNKNOWN"),
        (2.0, 3.5, "SPK2"),
    ]
    assert tr.load_rttm(str(tmp_path / "missing.rttm")) == []


def test_model_selection_rttm_index_and_discovery(tr, monkeypatch, tmp_path):
    audio = tmp_path / "audio"
    nested = audio / "nested"
    nested.mkdir(parents=True)
    monkeypatch.setattr(tr, "AUDIO_BASE", audio)
    good = audio / "20260102030405_large-v3_transcription.txt"
    other = nested / "20260102030405_small_transcription.txt"
    good.write_text(json.dumps({"segments": [1, 2, 3]}), encoding="utf-8")
    other.write_text(json.dumps({"segments": [1]}), encoding="utf-8")
    assert tr.extract_model_tag_from_json_txt(str(good)) == "large-v3"
    assert tr.model_rank("") == 10_000
    assert tr.model_rank("made-up-large") == 100
    assert tr.model_rank("unknown-model") == 9999
    assert tr.select_best_json([], []) is None
    assert tr.select_best_json([str(other), str(good)], []) == str(good)

    rttm_dir = tmp_path / "rttms"
    deep = rttm_dir / "deep"
    deep.mkdir(parents=True)
    shallow = rttm_dir / "20260102030405_speakers.rttm"
    deep_file = deep / "20260102030405_speakers.rttm"
    shallow.write_text("", encoding="utf-8")
    deep_file.write_text("", encoding="utf-8")
    index = tr.index_rttm_dirs([str(rttm_dir), str(tmp_path / "missing")])
    key = "2026_0102_080405"
    assert tr.pick_rttm_for_key(key, index) == str(shallow.resolve())
    assert tr.pick_rttm_for_key("not-there", index) is None

    scan = tmp_path / "scan"
    scan.mkdir()
    (scan / "20260102030405_large-v3_transcription.txt").write_text("{}", encoding="utf-8")
    (scan / "20260102030405_transcription.csv").write_text("Text\nhello\n", encoding="utf-8")
    (scan / "20260102030405_transcription_entities.csv").write_text("Text,Label\n", encoding="utf-8")
    (scan / "20260102030405.wav").write_bytes(b"x")
    (scan / "20260102030405_metadata.csv").write_text("Frame,Lat,Long\n", encoding="utf-8")
    monkeypatch.setattr(tr, "SCAN_ROOTS", [str(scan), str(tmp_path / "absent")])
    monkeypatch.setattr(tr, "DASHCAM_ROOT", str(scan))
    monkeypatch.setattr(tr, "OLD_DASHCAM_ROOT", str(tmp_path / "old-absent"))
    monkeypatch.setattr(tr, "RTTM_DIRS", [str(rttm_dir)])
    record = tr.discover_keys()[key]
    assert record["json_all"] and record["csv_all"]
    assert record["entities"] and record["media_all"] and record["meta_all"]
    assert record["rttm"] == str(shallow.resolve())


def test_segment_validation_and_geo_helpers(tr):
    segments, stats = tr.validate_and_clean_segments(
        [
            {"start": 0, "end": 1, "text": "ok"},
            {"start": 2, "end": 1, "text": "bad"},
            {"start": "x", "end": 3, "text": "bad"},
            {"start": 4, "end": 5, "text": "  "},
        ]
    )
    assert [segment["text"] for segment in segments] == ["ok"]
    assert stats["dropped"] == 3
    assert tr._bbox_ok(35.0, -80.0, (33, 38, -83, -70)) is True
    assert tr._bbox_ok(50.0, -80.0, (33, 38, -83, -70)) is False
    assert tr._repair_latlon(35.0, 80.0, (33, 38, -83, -70), True, True)[1] == -80.0
    assert tr._repair_latlon(-80.0, 35.0, (33, 38, -83, -70), False, True)[:2] == (35.0, -80.0)
    assert tr._repair_latlon(1000, 1000, (33, 38, -83, -70), True, True)[0] is None


def test_dashcam_metadata_quality_pipeline(tr, tmp_path):
    path = tmp_path / "clip_metadata.csv"
    path.write_text(
        "Frame,Lat,Long,Speed,Timestamp\n"
        "0,35.0,80.0,30,2026-01-01T00:00:00Z\n"
        "30,35.01,-80.01,200,2026-01-01T00:00:01Z\n"
        "60,-80.02,35.02,20,2026-01-01T00:00:02Z\n"
        "90,bad,-80.03,10,2026-01-01T00:00:03Z\n",
        encoding="utf-8",
    )
    rows, quality = tr.parse_dashcam_metadata_csv(
        str(path),
        fps=30.0,
        bbox=(33, 38, -83, -70),
        max_speed_mph=120,
        lon_auto_west=True,
        allow_latlon_swap=True,
        downsample_sec=1.0,
        min_keep_ratio=0.5,
        skip_when_bad=False,
    )
    assert len(rows) == 2
    assert quality["rows_total"] == 4
    assert quality["rows_kept"] == 2
    assert quality["rows_speed_dropped"] == 1
    assert quality["rows_invalid"] == 1
    assert quality["lon_flips"] == 1
    assert quality["latlon_swaps"] == 1
    assert quality["quality_ok"] is True

    skipped, bad_quality = tr.parse_dashcam_metadata_csv(
        str(path),
        fps=30.0,
        bbox=(33, 38, -83, -70),
        max_speed_mph=10,
        lon_auto_west=False,
        allow_latlon_swap=False,
        downsample_sec=1.0,
        min_keep_ratio=0.9,
        skip_when_bad=True,
    )
    assert skipped == []
    assert bad_quality["quality_ok"] is False
