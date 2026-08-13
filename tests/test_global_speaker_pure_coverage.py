from __future__ import annotations

import importlib
import sys
from types import ModuleType

import numpy as np
import pytest


def load_module(monkeypatch, tmp_path):
    neo = ModuleType("neo4j")
    neo.GraphDatabase = object
    exc = ModuleType("neo4j.exceptions")
    exc.Neo4jError = RuntimeError
    monkeypatch.setitem(sys.modules, "neo4j", neo)
    monkeypatch.setitem(sys.modules, "neo4j.exceptions", exc)
    import auto_ingest_config as cfg

    monkeypatch.setattr(cfg, "get_fileserver_path", lambda suffix="": str(tmp_path / suffix))
    monkeypatch.setattr(cfg, "get_neo4j_env", lambda: ("bolt://x", "u", "p", "neo4j"))
    sys.modules.pop("auto_ingest.diarize.link_global_speakers", None)
    return importlib.import_module("auto_ingest.diarize.link_global_speakers")


def test_audio_index_cache_normalization_and_discovery(monkeypatch, tmp_path):
    module = load_module(monkeypatch, tmp_path)
    audio = tmp_path / "audio"
    audio.mkdir()
    nested = audio / "nested"
    nested.mkdir()
    good = audio / "2026_0813_123456.wav"
    good.write_bytes(b"x")
    side = nested / "2026_0813_123456_transcription.wav"
    side.write_bytes(b"x")
    cache = tmp_path / "idx.json"
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *a, **k: type("R", (), {"stdout": str(good) + "\n" + str(side)})(),
    )
    idx = module.build_audio_index([audio], {".wav"}, cache_path=cache)
    assert module._norm_stem("2026_0813_123456_medium_transcription_speakers.wav") == "2026_0813_123456"
    assert idx["2026_0813_123456"][0] == good
    assert cache.exists()
    loaded = module.build_audio_index([audio], {".wav"}, cache_path=cache)
    assert loaded["2026_0813_123456"][0] == good
    module.set_audio_index(idx)
    assert module._index_lookup("2026_0813_123456") == good
    assert module.discover_audio_for_key("2026_0813_123456") == good
    assert module.discover_audio_for_key("missing") is None
    module.set_audio_index(None)
    monkeypatch.setattr(module, "AUDIO_BASE", audio)
    monkeypatch.setattr(module, "ALT_AUDIO_BASES", [])
    assert module.discover_audio_for_key("2026_0813_123456") == good


def test_vector_identity_regex_priority_and_assignment(monkeypatch, tmp_path):
    module = load_module(monkeypatch, tmp_path)
    assert module.stable_id("a") == module.stable_id("a") and module.stable_id("a") != module.stable_id("b")
    assert module.cosine(np.array([1.0, 0]), np.array([1.0, 0])) == pytest.approx(1)
    assert module.cosine(np.zeros(2), np.ones(2)) == 0
    assert np.allclose(module.unit(np.array([3.0, 4.0])), [0.6, 0.8])
    assert np.allclose(module.unit(np.zeros(2)), [0, 0])
    import re

    assert module._regex_ok("Scott", re.compile("scott", re.I))
    assert not module._regex_ok(None, re.compile("x"))

    class Records(list):
        def single(self):
            return self[0] if self else None

    class Sess:
        def __init__(self, rows):
            self.rows = rows
            self.writes = []

        def run(self, q, **kw):
            if "RETURN g.id AS gid" in q:
                return Records(self.rows)
            if "RETURN g.embedding AS emb" in q:
                return Records([{"emb": [1.0, 0.0], "w": 2}])
            self.writes.append((q, kw))
            return Records([])

    session = Sess([{"gid": "old", "name": "Scott", "aliases": [], "emb": [1, 0]}])
    assert module.ensure_priority_gs(session, "scott") == "old"
    session = Sess([])
    gid = module.ensure_priority_gs(session, "scott")
    assert gid and session.writes
    assert module.ensure_priority_gs(session, None) is None

    called = []
    monkeypatch.setattr(module, "update_existing_gs_with_assignments", lambda *a, **k: called.append((a, k)))
    centroids = {
        "a": np.array([1.0, 0.0]),
        "b": np.array([0.9, 0.1]),
        "c": np.array([-1.0, 0.0]),
    }
    weights = {"a": 3, "b": 2, "c": 1}
    session = Sess([])
    best, attached = module.assign_to_priority_first(session, "g", centroids, weights, {}, 0.8, 2)
    assert attached == ["a", "b"] and set(best) == {"a", "b"} and called
    assert module.assign_to_priority_first(session, "", centroids, weights, {}, 0.8, 2) == ({}, [])
    assert module.assign_to_priority_first(session, "g", {}, weights, {}, 0.8, 2) == ({}, [])


def test_audio_cache_snip_quality_and_cap(monkeypatch, tmp_path):
    module = load_module(monkeypatch, tmp_path)
    cache = tmp_path / "audio.json"
    module.save_audio_cache(cache, {"k": "/x.wav"})
    assert module.load_audio_cache(cache, False) == {"k": "/x.wav"}
    cache.write_text("{", encoding="utf-8")
    assert module.load_audio_cache(cache, False) == {}
    assert module.load_audio_cache(tmp_path / "missing", False) == {}
    assert module.load_audio_cache(cache, True) == {}
    tensor = module.torch.ones(16000)
    snip = module.fixed_snip(tensor, 16000, 0.2, 0.4, 0.0, 0.5)
    assert len(snip) > 0
    rms, snr = module.rms_and_snr_db(tensor)
    assert rms > 0 and np.isfinite(snr)
    rows = [("a", float(i), float(i) + 1, "x", 1.0) for i in range(5)] + [
        ("b", float(i), float(i) + 1, "x", 1.0) for i in range(2)
    ]
    capped = module.cap_snips_per_file(rows, 2, 4, module.random.Random(1))
    assert sum(row[0] == "a" for row in capped) <= 2 and len(capped) <= 4


def test_emb_cache_roundtrip(monkeypatch, tmp_path):
    module = load_module(monkeypatch, tmp_path)
    cache = module.EmbCache(tmp_path / "e.sqlite", False, "speechbrain", "m", 1.6, 16000)
    assert cache.get("s", "f", 0, 1) is None
    vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    cache.set("s", "f", 0, 1, vec, 0.2, 12.0)
    got = cache.get("s", "f", 0, 1)
    assert np.allclose(got[0], vec)
    assert got[1] == pytest.approx(0.2) and got[2] == pytest.approx(12)
    cache.conn.close()
