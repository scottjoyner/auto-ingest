"""Hermetic recovery test for canonical cleaned-intermediate replay (#12)."""
import hashlib
from pathlib import Path

import pytest

from auto_ingest.ingest_import import import_all
from auto_ingest.ingest_write import write_intermediate


class ReplayGraph:
    """Minimal committed-batch graph double; reject unexpected Cypher."""

    def __init__(self):
        self.nodes = {}
        self.edges = set()
        self.calls = 0
        self.fail_at = 3

    def session(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, query, *, rows):
        self.calls += 1
        assert len(rows) == 1
        if self.calls == self.fail_at:
            raise RuntimeError("injected batch failure")
        q = " ".join(query.split())
        parent_query = "UNWIND $rows AS r MERGE (n:Transcription {id: r.key}) SET n += r.props"
        child_query = ("UNWIND $rows AS r MERGE (n:Segment {id: r.key}) SET n += r.props "
                       "WITH n, r MATCH (p:Transcription {id: r.parent}) "
                       "MERGE (p)-[:HAS_SEGMENT]->(n)")
        assert q in (parent_query, child_query)
        for row in rows:
            label = "Transcription" if q == parent_query else "Segment"
            identity = (label, row["key"])
            self.nodes.setdefault(identity, {}).update(row["props"])
            if label == "Segment":
                parent = ("Transcription", row["parent"])
                assert parent in self.nodes
                self.edges.add((parent, "HAS_SEGMENT", identity))


def test_intermediate_replay_recovers_after_committed_batch(tmp_path):
    key = "fixture/clip"
    source_hash = hashlib.sha256(b"synthetic media fixture").hexdigest()
    parent = [{"id": "clip", "source_sha256": source_hash}]
    children = [
        {"id": "s1", "parent": "clip", "text": "one"},
        {"id": "s2", "parent": "clip", "text": "two"},
    ]
    paths = [
        Path(write_intermediate(str(tmp_path), key, "segments", children)),
        Path(write_intermediate(str(tmp_path), key, "transcription", parent)),
    ]
    original = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    graph = ReplayGraph()
    with pytest.raises(RuntimeError, match="injected batch failure"):
        import_all(str(tmp_path), graph, key=key, batch=1)
    assert set(graph.nodes) == {("Transcription", "clip"), ("Segment", "s1")}
    assert len(graph.edges) == 1

    graph.fail_at = None
    expected = {f"{key}/transcription": 1, f"{key}/segments": 2}
    assert import_all(str(tmp_path), graph, key=key, batch=1) == expected
    assert import_all(str(tmp_path), graph, key=key, batch=1) == expected
    assert graph.nodes == {
        ("Transcription", "clip"): {"source_sha256": source_hash},
        ("Segment", "s1"): {"parent": "clip", "text": "one"},
        ("Segment", "s2"): {"parent": "clip", "text": "two"},
    }
    assert graph.edges == {
        (("Transcription", "clip"), "HAS_SEGMENT", ("Segment", s))
        for s in ("s1", "s2")
    }
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in tmp_path.iterdir()} == original
