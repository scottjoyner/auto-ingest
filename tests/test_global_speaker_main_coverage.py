from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.ml


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


class Session:
    def __init__(self):
        self.runs = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def run(self, query, **kwargs):
        self.runs.append((query, kwargs))
        return []


class Driver:
    def __init__(self):
        self.sessions = []

    def session(self, **kwargs):
        session = Session()
        self.sessions.append((kwargs, session))
        return session


def args(tmp_path, **overrides):
    base = dict(
        state_file="",
        min_seg=0.1,
        min_prop=0.1,
        include_unknown=True,
        limit_speakers=0,
        source_level="auto",
        speaker_batch=100,
        items_per_speaker=10,
        max_speakers=0,
        skip_already_linked=False,
        include_no_audio=False,
        exclude_non_speech=True,
        audio_index=str(tmp_path / "audio-index.json"),
        audio_index_refresh=False,
        audio_cache=str(tmp_path / "audio-cache.json"),
        cache_refresh=False,
        emb_cache=str(tmp_path / "emb.sqlite"),
        emb_refresh=False,
        backend="speechbrain",
        snip_len=1.0,
        write_snips=False,
        snips_dir=None,
        holdout=True,
        max_per_file=3,
        max_snips=8,
        pad=0.0,
        min_rms=0.01,
        min_snr_db=1.0,
        weight_quality=True,
        global_prefilter=False,
        global_include_tentative=False,
        global_thresh=0.7,
        global_k=4,
        global_index="flatip",
        global_m=8,
        global_ef=16,
        dry_run=False,
        quarantine_min=0.5,
        priority_name=None,
        priority_thresh=0.8,
        priority_max_attach=100,
        faiss_prefilter=False,
        faiss_k=8,
        faiss_index="flatip",
        faiss_m=8,
        faiss_ef=16,
        thresh=0.7,
        holdout_min=0.4,
        holdout_action="drop-members",
        singleton_tentative=True,
        store_embeddings=True,
        rank_and_label=False,
        promote=False,
        promote_min_weight=3.0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def install_common(monkeypatch, module, tmp_path, parsed_args, speakers):
    drv = Driver()
    monkeypatch.setattr(module, "parse_args", lambda: parsed_args)
    monkeypatch.setattr(module, "driver", lambda: drv)
    monkeypatch.setattr(module, "ensure_schema", lambda d: None)
    monkeypatch.setattr(module, "fetch_speakers_and_segments", lambda *a, **k: speakers)
    monkeypatch.setattr(module, "build_audio_index", lambda *a, **k: {})
    monkeypatch.setattr(module, "set_audio_index", lambda idx: None)
    monkeypatch.setattr(module, "load_audio_cache", lambda *a, **k: {})
    monkeypatch.setattr(module, "save_audio_cache", lambda *a, **k: None)

    class Cache:
        def __init__(self, *a, **k):
            self.values = {}

        def get(self, sid, key, start, end):
            return self.values.get((sid, key, start, end))

        def set(self, sid, key, start, end, emb, rms, snr):
            self.values[(sid, key, start, end)] = (np.asarray(emb), rms, snr)

    monkeypatch.setattr(module, "EmbCache", Cache)
    monkeypatch.setattr(module, "cap_snips_per_file", lambda rows, *a, **k: list(rows))
    monkeypatch.setattr(module, "fixed_snip", lambda *a, **k: module.torch.ones(16))
    monkeypatch.setattr(module, "rms_and_snr_db", lambda snip: (1.0, 20.0))
    monkeypatch.setattr(module, "_mark_no_audio", lambda *a, **k: None)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"x")
    monkeypatch.setattr(module, "discover_audio_for_key", lambda key: audio)
    monkeypatch.setattr(module, "load_audio", lambda *a, **k: (module.torch.ones(32), 16000))
    return drv


def test_main_embeds_clusters_validates_writes_and_postprocesses(monkeypatch, tmp_path):
    module = load_module(monkeypatch, tmp_path)
    state = tmp_path / "state.json"
    state.write_text(json.dumps(["already-done"]), encoding="utf-8")
    parsed = args(
        tmp_path,
        state_file=str(state),
        rank_and_label=True,
        promote=True,
        holdout=True,
        holdout_action="drop-members",
    )
    speakers = {
        "s1": {"items": [("k1", 0.0, 1.0, "spoken words", 1.0), ("k1", 1.0, 2.0, "more words", 0.9)]},
        "s2": {"items": [("k2", 0.0, 1.0, "spoken words", 1.0), ("k2", 1.0, 2.0, "more words", 0.9)]},
    }
    drv = install_common(monkeypatch, module, tmp_path, parsed, speakers)

    vectors = iter([
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([0.95, 0.05], dtype=np.float32),
        np.array([0.95, 0.05], dtype=np.float32),
    ])

    class Embedder:
        def __init__(self, backend):
            self.backend = backend

        def embed(self, snip, sr):
            return next(vectors)

    monkeypatch.setattr(module, "SpkEmbedder", Embedder)
    writes = []
    monkeypatch.setattr(module, "write_clusters_incremental", lambda *a, **k: writes.append((a, k)))
    ranked = []
    promoted = []
    monkeypatch.setattr(module, "compute_dominance_and_label_transcriptions", lambda sess: ranked.append(sess))
    monkeypatch.setattr(module, "promote_by_evidence", lambda *a, **k: promoted.append((a, k)))

    module.main()

    assert writes
    groups = writes[0][0][1]
    assert groups == [["s1", "s2"]]
    assert ranked and promoted
    saved = set(json.loads(state.read_text(encoding="utf-8")))
    assert {"already-done", "s1", "s2"} <= saved
    assert drv.sessions


def test_main_global_prefilter_assigns_all_and_exits(monkeypatch, tmp_path):
    module = load_module(monkeypatch, tmp_path)
    parsed = args(tmp_path, global_prefilter=True, skip_already_linked=True)
    speakers = {"s1": {"items": [("k", 0.0, 1.0, "spoken words", 1.0)]}}
    install_common(monkeypatch, module, tmp_path, parsed, speakers)

    class Embedder:
        def __init__(self, backend): pass
        def embed(self, snip, sr): return np.array([1.0, 0.0], dtype=np.float32)

    monkeypatch.setattr(module, "SpkEmbedder", Embedder)
    monkeypatch.setattr(module, "fetch_global_speaker_embs", lambda *a, **k: {"g1": np.array([1.0, 0.0])})
    monkeypatch.setattr(module, "fetch_already_linked_speakers", lambda sess: {})
    monkeypatch.setattr(
        module,
        "assign_locals_to_globals",
        lambda *a, **k: ({"g1": ["s1"]}, {"s1": ("g1", 0.99)}),
    )
    updated = []
    monkeypatch.setattr(module, "update_existing_gs_with_assignments", lambda *a, **k: updated.append((a, k)))
    monkeypatch.setattr(module, "write_clusters_incremental", lambda *a, **k: (_ for _ in ()).throw(AssertionError("local clustering should not run")))

    module.main()
    assert updated


def test_main_marks_missing_audio_and_returns_without_centroids(monkeypatch, tmp_path):
    module = load_module(monkeypatch, tmp_path)
    parsed = args(tmp_path)
    speakers = {
        "empty": {"items": []},
        "missing": {"items": [("missing-key", 0.0, 1.0, "spoken words", 1.0)]},
    }
    drv = install_common(monkeypatch, module, tmp_path, parsed, speakers)
    monkeypatch.setattr(module, "discover_audio_for_key", lambda key: None)

    class Embedder:
        def __init__(self, backend): pass
        def embed(self, snip, sr): raise AssertionError("no audio should reach embedder")

    monkeypatch.setattr(module, "SpkEmbedder", Embedder)
    marked = []
    monkeypatch.setattr(module, "_mark_no_audio", lambda driver, ids: marked.extend(ids))
    monkeypatch.setattr(module, "write_clusters_incremental", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no centroids")))

    module.main()
    assert set(marked) == {"empty", "missing"}
    assert drv is not None
