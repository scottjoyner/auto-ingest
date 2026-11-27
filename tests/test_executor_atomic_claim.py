"""Exercise actual executor query functions with a lock-aware graph double."""
import ast
import datetime
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

import pytest


@pytest.fixture
def funcs():
    source = Path(__file__).resolve().parents[1] / "summarize_from_segments.py"
    names = {"q_pick_ready", "q_attach_run", "q_finish_run"}
    fns = [n for n in ast.parse(source.read_text()).body
           if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = dict(datetime=datetime, json=json, List=List, Dict=Dict, Any=Any)
    exec(compile(ast.Module(body=fns, type_ignores=[]), str(source), "exec"), ns)
    return ns


class Result:
    def __init__(self, records, failure=False):
        self.records = records
        self.failure = failure

    def __iter__(self):
        return iter(self.records)

    def single(self, *, strict):
        assert strict
        if len(self.records) != 1:
            raise RuntimeError("cardinality")
        return self.records[0]

    def consume(self):
        if self.failure:
            raise OSError("deferred")


class Graph:
    def __init__(self):
        self.status = "READY"
        self.owner = None
        self.runs = {}
        self.lock = threading.Lock()
        self.failure = False

    def run(self, query, *, limit, excluded):
        assert "SET " not in query
        assert "status:'READY'" in query
        assert limit == 1
        assert "WHERE NOT task.id IN $excluded" in query
        return Result([{"id": "task"}] if self.status == "READY" and "task" not in excluded else [])

    def begin_transaction(self):
        return Transaction(self)


class Transaction:
    def __init__(self, graph):
        self.g = graph
        self.committed = False

    def __enter__(self):
        self.g.lock.acquire()
        self.state = self.g.status, self.g.owner, self.g.runs.copy()
        return self

    def __exit__(self, *exc):
        if not self.committed:
            self.g.status, self.g.owner, self.g.runs = self.state
        self.g.lock.release()

    def run(self, query, **params):
        q = " ".join(query.split())
        assert "SET task._executor_lock = true REMOVE task._executor_lock WITH task WHERE" in q
        assert params["tid"] == "task"
        rid = params["rid"]
        if "CREATE (r:Run" in q:
            assert "WHERE task.status = 'READY' AND task.active_run_id IS NULL" in q
            assert "SET task.status = 'RUNNING', task.active_run_id = $rid" in q
            if self.g.status != "READY" or self.g.owner is not None:
                return Result([])
            self.g.status, self.g.owner = "RUNNING", rid
            self.g.runs[rid] = "RUNNING"
            return Result([{"rid": rid}], self.g.failure)
        assert "WHERE task.status = 'RUNNING' AND task.active_run_id = $rid" in q
        assert "WHERE r.status = 'RUNNING'" in q
        assert "REMOVE task.active_run_id" in q
        if (self.g.status, self.g.owner, self.g.runs.get(rid)) != ("RUNNING", rid, "RUNNING"):
            return Result([])
        self.g.status = self.g.runs[rid] = params["tstatus"]
        self.g.owner = None
        return Result([{"tid": "task", "rid": rid, "status": self.g.status}], self.g.failure)

    def commit(self):
        self.committed = True


def attach(funcs, graph, rid):
    return funcs["q_attach_run"](graph, "task", {
        "id": rid, "started_at": "2026-11-26T00:00:00Z",
        "status": "RUNNING", "manifest": {}})


def test_selection_exclusion_is_read_only(funcs):
    graph = Graph()
    assert funcs["q_pick_ready"](graph, 1, excluded=["task"]) == []
    assert graph.status == "READY" and graph.owner is None and not graph.runs
    assert funcs["q_pick_ready"](graph, 1) == ["task"]


def test_two_workers_one_claim(funcs):
    graph = Graph()
    both_read = threading.Barrier(2)

    def worker(rid):
        assert funcs["q_pick_ready"](graph, 1) == ["task"]
        both_read.wait(timeout=2)
        try:
            return attach(funcs, graph, rid)
        except RuntimeError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(worker, ["a", "b"]))
    winners = [r for r in results if r is not None]
    assert len(winners) == 1
    assert graph.status == "RUNNING" and graph.owner == winners[0]
    assert graph.runs == {winners[0]: "RUNNING"}


def test_stale_and_duplicate_finalization_rejected(funcs):
    graph = Graph()
    graph.runs["old"] = "RUNNING"
    attach(funcs, graph, "current")
    with pytest.raises(RuntimeError):
        funcs["q_finish_run"](graph, "task", "old", False, {})
    assert graph.owner == "current" and graph.status == "RUNNING"
    funcs["q_finish_run"](graph, "task", "current", True, {})
    assert graph.owner is None and graph.status == "DONE"
    with pytest.raises(RuntimeError):
        funcs["q_finish_run"](graph, "task", "current", False, {})
    assert graph.status == "DONE"


@pytest.mark.parametrize("stage", ["claim", "finish"])
def test_deferred_failure_rolls_back(funcs, stage):
    graph = Graph()
    if stage == "finish":
        attach(funcs, graph, "current")
    graph.failure = True
    with pytest.raises(OSError):
        if stage == "claim":
            attach(funcs, graph, "current")
        else:
            funcs["q_finish_run"](graph, "task", "current", True, {})
    if stage == "claim":
        assert graph.status == "READY" and graph.owner is None and not graph.runs
    else:
        assert graph.status == "RUNNING" and graph.owner == "current"
        assert graph.runs == {"current": "RUNNING"}
