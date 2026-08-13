from __future__ import annotations

import importlib
import random
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


@pytest.fixture()
def lg(monkeypatch, tmp_path):
    import auto_ingest.backend as backend
    import auto_ingest_config as cfg

    monkeypatch.setattr(backend, "_BACKEND", "onnx")
    monkeypatch.setattr(cfg, "get_fileserver_path", lambda suffix="": str(tmp_path / suffix))
    monkeypatch.setattr(
        cfg,
        "get_neo4j_env",
        lambda: ("bolt://unused", "neo4j", "unused", "neo4j"),
    )

    torch = ModuleType("torch")
    torch.Tensor = object

    def inference_mode():
        return lambda fn: fn

    torch.inference_mode = inference_mode
    torchaudio = ModuleType("torchaudio")
    torchaudio.functional = SimpleNamespace()
    soundfile = ModuleType("soundfile")
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torchaudio", torchaudio)
    monkeypatch.setitem(sys.modules, "soundfile", soundfile)
    sys.modules.pop("auto_ingest.diarize.link_global_speakers", None)
    module = importlib.import_module("auto_ingest.diarize.link_global_speakers")
    yield module
    sys.modules.pop("auto_ingest.diarize.link_global_speakers", None)


def test_stem_and_audio_path_helpers(lg, tmp_path):
    audio = tmp_path / "audio"
    audio.mkdir()
    wav = audio / "2026_0102_030405.wav"
    wav.write_bytes(b"wav")
    assert lg._norm_stem("2026_0102_030405_medium_transcription_speakers.rttm") == "2026_0102_030405"
    assert lg._norm_stem("abc_R") == "abc"
    assert lg._is_audio_path(wav)
    assert lg._looks_like_audio_dir(audio)
    assert not lg._is_audio_path(tmp_path / "missing.wav")

    lg.set_audio_index({"2026_0102_030405": [wav]})
    assert lg._index_lookup("2026_0102_030405") == wav
    assert lg._index_lookup("missing") is None
    lg.set_audio_index(None)
    assert lg._index_lookup("x") is None


def test_build_audio_index_walk_cache_and_refresh(lg, monkeypatch, tmp_path):
    base = tmp_path / "base"
    nested = base / "audio"
    nested.mkdir(parents=True)
    (nested / "a.wav").write_bytes(b"a")
    (nested / "b.txt").write_text("b", encoding="utf-8")
    cache = tmp_path / "index.json"
    monkeypatch.setattr(
        lg.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(stdout=""),
    )
    idx = lg.build_audio_index([base, tmp_path / "absent"], {".wav"}, cache)
    assert "a" in idx
    assert idx["a"][0].name == "a.wav"
    assert cache.exists()

    loaded = lg.build_audio_index([tmp_path / "unused"], {".wav"}, cache)
    assert loaded["a"][0].name == "a.wav"

    cache.write_text("not-json", encoding="utf-8")
    rebuilt = lg.build_audio_index([base], {".wav"}, cache)
    assert "a" in rebuilt


def test_discover_audio_uses_index_and_fallback(lg, tmp_path):
    base = tmp_path / "audio"
    base.mkdir()
    exact = base / "clip.wav"
    exact.write_bytes(b"x")
    lg.AUDIO_BASE = base
    lg.ALT_AUDIO_BASES = []
    lg.set_audio_index({"clip": [exact]})
    assert lg.discover_audio_for_key("clip") == exact
    lg.set_audio_index({})
    assert lg.discover_audio_for_key("missing") is None
    lg.set_audio_index(None)
    assert lg.discover_audio_for_key("clip") == exact

    exact.unlink()
    relaxed = base / "clip_front.wav"
    relaxed.write_bytes(b"x")
    sidecar = base / "clip_transcription.wav"
    sidecar.write_bytes(b"x")
    assert lg.discover_audio_for_key("clip") == relaxed


def test_stable_cosine_unit_and_regex(lg):
    assert lg.stable_id("a") == lg.stable_id("a")
    assert lg.stable_id("a") != lg.stable_id("b")
    assert lg.cosine(np.array([1.0, 0]), np.array([1.0, 0])) == 1.0
    assert lg.cosine(np.zeros(2), np.ones(2)) == 0.0
    assert np.linalg.norm(lg.unit(np.array([3.0, 4.0]))) == pytest.approx(1.0)
    assert np.array_equal(lg.unit(np.zeros(2)), np.zeros(2))
    import re

    assert lg._regex_ok("Scott", re.compile("scott", re.I))
    assert not lg._regex_ok(None, re.compile("x"))
    assert not lg._regex_ok("x", None)


def test_embedding_cache_roundtrip_refresh_and_corruption(lg, tmp_path):
    path = tmp_path / "emb.sqlite"
    cache = lg.EmbCache(path, False, "speechbrain", "model", 1.6, 16000)
    assert cache.get("s", "f", 0, 1) is None
    emb = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    cache.set("s", "f", 0, 1, emb, 0.2, 9.0)
    got, rms, snr = cache.get("s", "f", 0, 1)
    assert np.array_equal(got, emb)
    assert rms == pytest.approx(0.2)
    assert snr == pytest.approx(9.0)
    key1 = cache._mk_key("s", "f", 0, 1)
    key2 = cache._mk_key("s", "f", 0, 2)
    assert key1 != key2

    refresh = lg.EmbCache(path, True, "speechbrain", "model", 1.6, 16000)
    assert refresh.get("s", "f", 0, 1) is None
    no_db = lg.EmbCache(None, False, "speechbrain", "model", 1.6, 16000)
    no_db.set("s", "f", 0, 1, emb, 0, 0)
    assert no_db.get("s", "f", 0, 1) is None

    cache.conn.execute("UPDATE emb_cache SET dim=99 WHERE key=?", (key1,))
    cache.conn.commit()
    assert cache.get("s", "f", 0, 1) is None


def test_cap_snips_per_file(lg):
    items = [
        ("f1", 0.0, 1.0, "a", 0.9),
        ("f1", 1.0, 3.0, "b", 0.8),
        ("f1", 3.0, 4.0, "c", 0.7),
        ("f2", 0.0, 2.0, "d", 1.0),
    ]
    chosen = lg.cap_snips_per_file(items, max_per_file=2, max_total=3, rng=random.Random(1))
    assert len(chosen) == 3
    assert sum(1 for x in chosen if x[0] == "f1") <= 2


def test_fetch_global_and_linked_speakers(lg):
    class Session:
        def __init__(self):
            self.calls = 0

        def run(self, query):
            self.calls += 1
            if "GlobalSpeaker" in query and "SAME_PERSON" not in query:
                return [
                    {"gid": "g1", "emb": [3.0, 4.0]},
                    {"gid": "empty", "emb": []},
                ]
            return [{"sid": "s1", "gid": "g1"}, {"sid": "s2", "gid": "g2"}]

    session = Session()
    confirmed = lg.fetch_global_speaker_embs(session, include_tentative=False)
    assert set(confirmed) == {"g1"}
    assert np.linalg.norm(confirmed["g1"]) == pytest.approx(1.0)
    all_status = lg.fetch_global_speaker_embs(session, include_tentative=True)
    assert set(all_status) == {"g1"}
    assert lg.fetch_already_linked_speakers(session) == {"s1": "g1", "s2": "g2"}


def test_assign_locals_to_globals_exact(lg):
    locals_ = {
        "s1": np.array([1.0, 0.0], dtype=np.float32),
        "s2": np.array([0.0, 1.0], dtype=np.float32),
        "s3": np.array([-1.0, 0.0], dtype=np.float32),
    }
    globals_ = {
        "g1": np.array([1.0, 0.0], dtype=np.float32),
        "g2": np.array([0.0, 1.0], dtype=np.float32),
    }
    assignments, best = lg.assign_locals_to_globals(
        locals_, globals_, thresh=0.8, k=2, use_faiss=False,
        index_type="flatip", hnsw_m=8, hnsw_ef=16,
    )
    assert assignments == {"g1": ["s1"], "g2": ["s2"]}
    assert best["s1"][0] == "g1"
    assert "s3" not in best
    assert lg.assign_locals_to_globals({}, globals_, 0.8, 2, False, "flatip", 8, 16) == ({}, {})
    assert lg.assign_locals_to_globals(locals_, {}, 0.8, 2, False, "flatip", 8, 16) == ({}, {})
