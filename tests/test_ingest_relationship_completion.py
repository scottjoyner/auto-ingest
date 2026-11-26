"""Hermetic explicit-transaction regression for linked imports."""
import pytest

from auto_ingest.ingest_import import import_intermediate
from auto_ingest.ingest_write import write_intermediate


class Graph:
    def __init__(self, failure=None):
        self.failure = failure
        self.parents = {"p": 1, "q": 1}
        self.nodes = set()
        self.edges = set()
        self.commits = 0
        self.rollbacks = 0
        self.calls = 0

    def session(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def begin_transaction(self):
        return Transaction(self)

class Transaction:
    def __init__(self, graph):
        self.g = graph
        self.nodes = set(graph.nodes)
        self.edges = set(graph.edges)
        self.committed = False
        self.consumed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if not self.committed:
            self.g.rollbacks += 1

    def run(self, query, *, rows):
        assert " ".join(query.split()) == (
            "UNWIND $rows AS r MERGE (n:Segment {id: r.key}) SET n += r.props "
            "WITH n, r MATCH (p:Transcription {id: r.parent}) "
            "MERGE (p)-[:HAS_SEGMENT]->(n) "
            "RETURN r.row_index AS row_index, count(*) AS matches")
        self.g.calls += 1
        self.acks = []
        for r in rows:
            self.nodes.add(r["key"])
            n = self.g.parents.get(r["parent"], 0)
            if n:
                self.edges.add((r["parent"], r["key"]))
                self.acks.append({"row_index": r["row_index"], "matches": n})
        if self.g.failure == "wrong_index":
            self.acks[-1]["row_index"] = 99
        elif self.g.failure == "duplicate_ack":
            self.acks.append(dict(self.acks[0]))
        return self

    def __iter__(self):
        return iter(reversed(self.acks))

    def consume(self):
        if self.g.failure == "consume":
            raise OSError("deferred failure")
        self.consumed = True

    def commit(self):
        assert self.consumed
        if self.g.failure == "commit":
            raise OSError("commit failure")
        self.g.nodes = self.nodes
        self.g.edges = self.edges
        self.g.commits += 1
        self.committed = True

def run(graph, path, batch=2):
    return import_intermediate(graph, path, batch=batch, label="Segment",
        key_field="id", parent_label="Transcription", parent_field="id",
        edge_label="HAS_SEGMENT")

@pytest.fixture
def artifact(tmp_path):
    return write_intermediate(str(tmp_path), "fixture", "segments",
        [{"id": "s1", "parent": "p"}, {"id": "s2", "parent": "q"}])

@pytest.mark.parametrize("failure", ["missing", "ambiguous", "wrong_index",
    "duplicate_ack", "consume", "commit"])
def test_rejected_batch_rolls_back_and_replays(artifact, failure):
    graph = Graph(failure)
    if failure == "missing":
        del graph.parents["q"]
    elif failure == "ambiguous":
        graph.parents["q"] = 2
    with pytest.raises((RuntimeError, OSError)):
        run(graph, artifact)
    assert graph.nodes == graph.edges == set()
    assert graph.commits == 0 and graph.rollbacks == 1
    graph.parents["q"] = 1
    graph.failure = None
    assert run(graph, artifact) == run(graph, artifact) == 2
    assert graph.nodes == {"s1", "s2"}
    assert graph.edges == {("p", "s1"), ("q", "s2")}
    assert graph.commits == 2

def test_prior_committed_batch_retained(artifact):
    graph = Graph()
    del graph.parents["q"]
    with pytest.raises(RuntimeError, match="Incomplete or ambiguous"):
        run(graph, artifact, batch=1)
    assert graph.nodes == {"s1"} and graph.edges == {("p", "s1")}
    assert graph.commits == graph.rollbacks == 1
    graph.parents["q"] = 1
    assert run(graph, artifact, batch=1) == 2
    assert graph.nodes == {"s1", "s2"} and len(graph.edges) == 2
