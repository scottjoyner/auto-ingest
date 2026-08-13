from __future__ import annotations

import importlib
import sys
from types import ModuleType

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


class Result(list):
    def single(self):
        return self[0] if self else None


def test_global_fetch_and_exact_assignment(monkeypatch, tmp_path):
    module = load_module(monkeypatch, tmp_path)

    class Session:
        def run(self, query, **kw):
            if "SAME_PERSON" in query:
                return Result([{"sid": "s1", "gid": "g1"}])
            return Result([{"gid": "g1", "emb": [1.0, 0.0]}, {"gid": "g2", "emb": [0.0, 1.0]}, {"gid": "z", "emb": []}])

    embeddings = module.fetch_global_speaker_embs(Session(), False)
    assert set(embeddings) == {"g1", "g2"}
    assert module.fetch_already_linked_speakers(Session()) == {"s1": "g1"}
    locals_ = {"a": np.array([0.9, 0.1]), "b": np.array([0.1, 0.9]), "c": np.array([-1.0, 0.0])}
    assignments, best = module.assign_locals_to_globals(locals_, embeddings, 0.7, 2, False, "flatip", 8, 16)
    assert assignments == {"g1": ["a"], "g2": ["b"]}
    assert set(best) == {"a", "b"}


def test_update_existing_global_speaker_confirmed_and_tentative(monkeypatch, tmp_path):
    module = load_module(monkeypatch, tmp_path)

    class Session:
        def __init__(self, me=False):
            self.me = me
            self.writes = []

        def run(self, query, **kw):
            if "RETURN g.embedding AS emb" in query:
                return Result([{"emb": [1.0, 0.0], "w": 2.0, "me": self.me, "pid": "scott" if self.me else None, "lbl": "Scott" if self.me else None}])
            self.writes.append((query, kw))
            return Result([])

    centroids = {"a": np.array([1.0, 0.0]), "b": np.array([0.9, 0.1])}
    session = Session(me=True)
    module.update_existing_gs_with_assignments(session, {"g": ["a", "b"]}, centroids, {"a": 2.0, "b": 1.0}, "test", 0.5)
    assert any(write[1].get("status") == "confirmed" for write in session.writes)
    assert any("s.is_me=true" in query for query, _ in session.writes)

    session = Session(me=False)
    opposite = {"x": np.array([1.0, 0.0]), "y": np.array([-1.0, 0.0])}
    module.update_existing_gs_with_assignments(session, {"g": ["x", "y"]}, opposite, {"x": 1, "y": 1}, "test", 0.9)
    assert any(write[1].get("status") == "tentative" for write in session.writes)


def test_write_clusters_dry_run_and_persist(monkeypatch, tmp_path):
    module = load_module(monkeypatch, tmp_path)

    class Session:
        def __init__(self, any_me=False):
            self.any_me = any_me
            self.writes = []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def run(self, query, **kw):
            if "count(s) > 0 AS b" in query:
                return Result([{"b": self.any_me}])
            if "RETURN s.person_id AS pid" in query:
                return Result([{"pid": "scott", "lbl": "Scott"}])
            if "RETURN g.embedding AS emb" in query:
                return Result([])
            self.writes.append((query, kw))
            return Result([])

    class Driver:
        def __init__(self, session):
            self._session = session

        def session(self, **kw):
            return self._session

    centroids = {"a": np.array([1.0, 0.0]), "b": np.array([0.9, 0.1]), "c": np.array([0.0, 1.0])}
    scores = {("a", "b"): 0.95}
    weights = {key: 1 for key in centroids}
    session = Session()
    module.write_clusters_incremental(Driver(session), [["a", "b"], ["c"]], scores, centroids, weights, "m", 0.8, True, False, True)
    assert not session.writes
    session = Session(any_me=True)
    module.write_clusters_incremental(Driver(session), [["a", "b"]], scores, centroids, weights, "m", 0.8, True, True, False)
    assert any("GlobalSpeaker" in query for query, _ in session.writes)
    assert any("s.is_me=true" in query for query, _ in session.writes)


def test_cluster_edge_cases(monkeypatch, tmp_path):
    module = load_module(monkeypatch, tmp_path)
    assert module.cluster_by_threshold_faiss({}, 0.8, 2) == ([], {})
    assert module.cluster_by_threshold_faiss({"a": np.array([1.0, 0.0])}, 0.8, 2) == ([['a']], {})
