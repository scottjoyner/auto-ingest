"""Opt-in real Neo4j + loopback HTTP recovery acceptance; no production I/O."""
import ast
import datetime
import json
import os
import pathlib
import subprocess
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional

import requests
from neo4j import GraphDatabase

import verify_disposable_ingest_e2e as fixture
from auto_ingest import executor_evidence
from auto_ingest.executor_http_report import report
from auto_ingest.executor_reconcile import inspect_tasks


def child():
    import time
    cfg = json.load(sys.stdin)
    assert cfg["uri"].startswith("bolt://127.0.0.1:")
    source = fixture.ROOT / "summarize_from_segments.py"
    names = {"q_pick_ready", "q_get_task", "q_attach_run", "q_finish_run",
             "ensure_artifacts", "execute_task_minimal", "execute_ready",
             "run_acceptance", "acc_http_ok"}
    nodes = [n for n in ast.parse(source.read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names
    driver = GraphDatabase.driver(cfg["uri"], auth=tuple(cfg["auth"]),
                                  connection_timeout=3, connection_acquisition_timeout=5,
                                  max_transaction_retry_time=0)
    ns = dict(globals(), time=time, ARTIFACTS_DIR=cfg["artifacts"],
              neo_driver=lambda: driver)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), ns)
    ns["ACCEPTANCE_FUNCS"] = {"http_ok": ns["acc_http_ok"]}

    def stop(*args, **kwargs):
        os._exit(77)

    mode = cfg["mode"]
    if mode == "before_get":
        requests.get = stop
    elif mode == "before_outcome":
        executor_evidence.publish_http_outcome = stop
    elif mode == "before_finalize":
        ns["q_finish_run"] = stop
    else:
        assert mode == "normal"
    try:
        ns["execute_ready"](cfg.get("limit", 1))
    finally:
        driver.close()


def execute_fixture(driver, root, connection):
    hits = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.path)
            self.send_response(503 if self.path == "/bad" else 200)
            self.end_headers()
            self.wfile.write(b"synthetic")

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    artifacts = root / "artifacts"

    def q(query, **params):
        with driver.session() as session:
            return session.run(query, **params).data()

    def worker(mode, expected, limit=1):
        data = json.dumps(dict(connection, artifacts=str(artifacts), mode=mode, limit=limit))
        result = subprocess.run(
            [sys.executable, str(pathlib.Path(__file__).resolve()), "--child"],
            input=data, text=True, capture_output=True, timeout=20,
            env=dict(os.environ, NO_PROXY="127.0.0.1", no_proxy="127.0.0.1"))
        # Do not emit child logs, which could include connection details.
        assert result.returncode == expected, ("child exit", mode, result.returncode)

    def state(tid):
        rows = q("MATCH (t:Task {id:$tid})-[:HAS_RUN]->(r:Run) "
                 "RETURN t.status AS status,t.active_run_id AS owner,"
                 "r.id AS rid,r.status AS rs,r.manifest_json AS manifest", tid=tid)
        assert len(rows) == 1
        return rows[0]

    try:
        assert q("MATCH (n) RETURN count(n) AS n") == [{"n": 0}]
        for mode, endpoint, expected_hits, expected_class, expected_status in (
            ("before_get", "/before", 0, "HTTP_INTENT_ONLY_UNCERTAIN", "RUNNING"),
            ("before_outcome", "/after-get", 1, "HTTP_INTENT_ONLY_UNCERTAIN", "RUNNING"),
            ("before_finalize", "/after-outcome", 1, "HTTP_LOCAL_RESPONSE_OBSERVED", "RUNNING"),
            ("normal", "/ok", 1, "HTTP_LOCAL_RESPONSE_OBSERVED", "DONE"),
            ("normal", "/bad", 1, "HTTP_LOCAL_RESPONSE_OBSERVED", "FAILED"),
        ):
            tid = str(uuid.uuid4())
            url = base + endpoint
            acceptance = json.dumps([{"type": "http_ok", "args": {"url": url}}])
            q("CREATE (:Task {id:$tid,status:'READY',updated_at:datetime(),"
              "acceptance_json:$acceptance})", tid=tid, acceptance=acceptance)
            before = len(hits)
            worker(mode, 0 if mode == "normal" else 77)
            row = state(tid)
            assert row["status"] == row["rs"] == expected_status
            assert row["owner"] == (row["rid"] if expected_status == "RUNNING" else None)
            evidence = report(artifacts / tid, task_id=tid, run_id=row["rid"], url=url)
            assert evidence["classification"] == expected_class
            assert not evidence["replay_allowed"] and not evidence["remote_completion_verified"]
            if expected_class == "HTTP_LOCAL_RESPONSE_OBSERVED":
                assert evidence["status"] == (503 if endpoint == "/bad" else 200)
            else:
                assert evidence["status"] is None
            graph = inspect_tasks(driver, [tid], now=datetime.datetime.now(datetime.timezone.utc),
                                  stale_after=datetime.timedelta(hours=1))[0]
            assert not graph["replay_allowed"]
            assert graph["classification"] == (
                "RUNNING_OBSERVED" if expected_status == "RUNNING" else "NOT_RUNNING")
            if expected_status != "RUNNING":
                assert json.loads(row["manifest"])["acceptance_results"][0]["passed"] == (
                    expected_status == "DONE")
            # Restart worker against the same graph; no READY task or replay.
            worker("normal", 0)
            assert state(tid) == row
            assert len(hits) - before == expected_hits
            print("PASS", mode, endpoint, expected_status, expected_class,
                  "GETs", expected_hits, flush=True)
        # Real stored corruption must not become an empty success contract.
        for invalid in ("{", "null", '[{"type":"http_ok","type":"file_exists"}]'):
            tid = str(uuid.uuid4())
            q("CREATE (:Task {id:$tid,status:'READY',updated_at:datetime(),"
              "acceptance_json:$acceptance})", tid=tid, acceptance=invalid)
            snapshot_query = ("MATCH (t:Task {id:$tid}) OPTIONAL MATCH (t)-[:HAS_RUN]->(r:Run) "
                              "RETURN properties(t) AS task,count(r) AS runs")
            before = q(snapshot_query, tid=tid)
            assert before[0]["runs"] == 0
            before_hits = len(hits)
            for _ in range(2):
                worker("normal", 1)
                assert q(snapshot_query, tid=tid) == before
                assert len(hits) == before_hits
                assert not (artifacts / tid).exists()
            # Remove only the synthetic task in this disposable graph so the
            # next independent case can be selected.
            q("MATCH (t:Task {id:$tid}) DELETE t", tid=tid)
            print("PASS corrupt persisted contract: two rejected worker attempts, "
                  "unchanged READY task, no run/artifact/GET", flush=True)
        print("REAL_GRAPH_CORRUPT_CONTRACT_PASS 3/3", flush=True)
        assert q("MATCH (r:Run) RETURN count(r) AS n") == [{"n": 5}]
        assert len(hits) == 4
        print("REAL_GRAPH_HTTP_RECOVERY_PASS 5/5", flush=True)
        # Older malformed task is skipped only within the bounded invocation.
        bad, good = str(uuid.uuid4()), str(uuid.uuid4())
        url = base + "/mixed"
        q("CREATE (:Task {id:$bad,status:'READY',updated_at:datetime('2026-01-01T00:00:00Z'),"
          "acceptance_json:'{'}) "
          "CREATE (:Task {id:$good,status:'READY',updated_at:datetime('2026-01-02T00:00:00Z'),"
          "acceptance_json:$acceptance})", bad=bad, good=good,
          acceptance=json.dumps([{"type": "http_ok", "args": {"url": url}}]))
        before = q(snapshot_query, tid=bad)
        worker("normal", 1, limit=1)
        assert q(snapshot_query, tid=bad) == before
        assert q(snapshot_query, tid=good)[0]["runs"] == 0
        assert len(hits) == 4
        worker("normal", 1, limit=2)
        row = state(good)
        assert row["status"] == row["rs"] == "DONE" and row["owner"] is None
        assert json.loads(row["manifest"])["acceptance_results"][0]["passed"]
        assert report(artifacts / good, task_id=good, run_id=row["rid"],
                      url=url)["classification"] == "HTTP_LOCAL_RESPONSE_OBSERVED"
        assert q(snapshot_query, tid=bad) == before and not (artifacts / bad).exists()
        assert hits.count("/mixed") == 1 and len(hits) == 5
        worker("normal", 1, limit=2)
        assert state(good) == row and q(snapshot_query, tid=bad) == before
        assert len(hits) == 5
        assert q("MATCH (r:Run) RETURN count(r) AS n") == [{"n": 6}]
        print("REAL_GRAPH_MIXED_BATCH_PASS budget1 blocked, budget2 valid DONE, "
              "restart no replay, corrupt unchanged", flush=True)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def main():
    connection = {}

    class Capture:
        @staticmethod
        def driver(uri, **kwargs):
            assert uri.startswith("bolt://127.0.0.1:")
            connection.update(uri=uri, auth=kwargs["auth"])
            return GraphDatabase.driver(uri, **kwargs)

    # Reuse the established resource guard, pinned cached image, disposable
    # credentials, loopback bind, deadline and teardown from the earlier fixture.
    fixture.GraphDatabase = Capture
    fixture.execute_fixture = lambda driver, root: execute_fixture(driver, root, connection)
    fixture.main()


if __name__ == "__main__":
    if sys.argv[1:] == ["--child"]:
        child()
    else:
        assert not sys.argv[1:]
        main()
