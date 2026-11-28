"""Verify an installed wheel using an exclusively disposable loopback Neo4j.

The caller must create and remove a fresh fixture database. No external model
endpoint is used. Run outside the repository, with the wheel installed into
the active environment or an explicitly provided isolated target directory.
"""
import argparse
import hashlib
import http.server
import importlib.metadata as md
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import threading
import time

from packaging.requirements import Requirement


def verify_operator_report(root, env, target):
    """Exercise the actual packaged operator command, without network or DB."""
    from auto_ingest.executor_evidence import publish_http_intent, publish_http_outcome

    directory = root / "operator-fixture"
    directory.mkdir()
    identity = {"task_id": "fixture", "run_id": "11111111-1111-4111-8111-111111111111",
                "url": "http://127.0.0.1:1/fixture?marker=synthetic-private"}
    # Deliberately strip all database/model environment variables.
    clean_env = {"PATH": env["PATH"], "HOME": str(root),
                 "PYTHONDONTWRITEBYTECODE": "1", "NO_PROXY": "*"}
    if target:
        clean_env["PYTHONPATH"] = str(target)

    def invoke():
        result = subprocess.run(
            [sys.executable, "-m", "auto_ingest.executor_http_report",
             "--artifact-dir", str(directory)],
            input=json.dumps(identity), cwd=root, env=clean_env,
            capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, "installed operator report failed"
        assert "synthetic-private" not in result.stdout
        data = json.loads(result.stdout)
        assert data["replay_allowed"] is False
        assert data["remote_completion_verified"] is False
        assert data["task_id"] == identity["task_id"] and data["run_id"] == identity["run_id"]
        assert data["target_sha256"] == hashlib.sha256(identity["url"].encode()).hexdigest()
        return data

    assert invoke()["classification"] == "HTTP_NO_EVIDENCE"
    publish_http_intent(directory, **identity)
    assert invoke()["classification"] == "HTTP_INTENT_ONLY_UNCERTAIN"
    publish_http_outcome(directory, **identity, status=503)
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    data = invoke()
    assert data["classification"] == "HTTP_LOCAL_RESPONSE_OBSERVED" and data["status"] == 503
    assert before == {p.name: p.read_bytes() for p in directory.iterdir()}
    print("INSTALLED_OPERATOR_REPORT_PASS 3/3", flush=True)
    return 3



def verify_operation_cli(root, env, target, q, port, password):
    """Actual installed combined report against the disposable graph.

    Fixture Community admin is not a production read-only RBAC assertion.
    """
    from auto_ingest.executor_evidence import (
        publish_http_intent,
        publish_local_output_receipt,
        publish_run_output,
    )

    directory = root / "combined-operation-fixture"
    directory.mkdir()
    identity = {"task_id": "report-fixture", "run_id": "22222222-2222-4222-8222-222222222222",
                "url": "http://127.0.0.1:1/report?marker=synthetic-private"}
    contract = json.dumps([{"type": "http_ok", "args": {"url": identity["url"]}}])
    q("CREATE (t:Task {id:$tid,status:'RUNNING',active_run_id:$rid,acceptance_json:$contract})"
      "-[:HAS_RUN]->(:Run {id:$rid,status:'RUNNING'})",
      tid=identity["task_id"], rid=identity["run_id"], contract=contract)
    publish_http_intent(directory, **identity)
    output = ("fixture\nRUN: " + identity["run_id"] + "\n").encode()
    local = {"task_id": identity["task_id"], "run_id": identity["run_id"], "output": output}
    publish_run_output(directory, **local)
    publish_local_output_receipt(directory, **local)
    clean = {"PATH": env["PATH"], "HOME": str(root), "PYTHONDONTWRITEBYTECODE": "1",
             "NO_PROXY": "*", "EXECUTOR_REPORT_USER": "neo4j",
             "EXECUTOR_REPORT_PASSWORD": password}
    if target:
        clean["PYTHONPATH"] = str(target)

    def invoke():
        return subprocess.run(
            [sys.executable, "-m", "auto_ingest.executor_operation_cli",
             "--artifact-dir", str(directory), "--bolt-port", str(port), "--database", "neo4j",
             "--confirm-read-only-admission"], cwd=root, env=clean,
            input=json.dumps(identity), capture_output=True, text=True, timeout=15)

    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    graph_query = ("MATCH (t:Task {id:$tid})-[:HAS_RUN]->(r:Run) "
                   "RETURN properties(t) AS task,properties(r) AS run")
    graph_before = q(graph_query, tid=identity["task_id"])
    failure = invoke()
    assert failure.returncode == 2 and not failure.stdout
    assert "no replay" in failure.stderr
    q("CREATE INDEX FOR (t:Task) ON (t.id)")
    q("CALL db.awaitIndexes(20)")
    success = invoke()
    assert success.returncode == 0, "combined installed CLI failed"
    result = json.loads(success.stdout)
    assert result["graph_classification"] == "RUNNING_OWNER_MATCH"
    assert result["local_output"] == "LOCAL_OUTPUT_BYTES_MATCH"
    assert result["http"]["classification"] == "HTTP_INTENT_ONLY_UNCERTAIN"
    assert result["replay_allowed"] is False and result["activation_allowed"] is False
    assert result["remote_completion_verified"] is False and result["atomic_snapshot"] is False
    assert identity["url"] not in success.stdout and password not in success.stdout + success.stderr
    assert before == {p.name: p.read_bytes() for p in directory.iterdir()}
    assert graph_before == q(graph_query, tid=identity["task_id"])
    print("INSTALLED_COMBINED_OPERATION_PASS 2/2", flush=True)
    return 2


def main():
    if not __debug__:
        raise RuntimeError("installed acceptance requires Python assertions enabled")
    ap = argparse.ArgumentParser()
    ap.add_argument("--wheel", required=True, type=pathlib.Path)
    ap.add_argument("--manifest", required=True, type=pathlib.Path)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--installed-target", type=pathlib.Path)
    args = ap.parse_args()
    assert re.fullmatch(r"[a-f0-9]{40}", args.commit)
    wheel = args.wheel.resolve(strict=True)
    assert wheel.is_file() and wheel.suffix == ".whl"
    manifest = args.manifest.resolve()
    assert not manifest.exists(), "manifest already exists"
    port = int(os.environ["MUSE_FIXTURE_BOLT_PORT"])
    password = os.environ["MUSE_FIXTURE_PASSWORD"]
    assert 1 <= port <= 65535 and re.fullmatch(r"[a-f0-9]{32}", password)
    uri = f"bolt://127.0.0.1:{port}"
    target = args.installed_target.resolve(strict=True) if args.installed_target else None
    if target:
        sys.path.insert(0, str(target))
        dist = next(d for d in md.distributions(path=[str(target)])
                    if d.metadata["Name"] == "auto-ingest")
    else:
        dist = md.distribution("auto-ingest")
    import auto_ingest
    from auto_ingest.executor_evidence import inspect_local_output
    from neo4j import GraphDatabase
    installed = pathlib.Path(auto_ingest.__file__).resolve()
    assert installed.is_relative_to(target or pathlib.Path(dist.locate_file("")).resolve())
    assert not installed.is_relative_to(pathlib.Path(__file__).resolve().parents[1])
    deps = {}
    for raw in dist.requires:
        req = Requirement(raw)
        if req.marker and not req.marker.evaluate({"extra": ""}):
            continue
        version = md.version(req.name)
        assert req.specifier.contains(version), req.name
        deps[req.name] = version
    assert set(deps) == {"pydantic", "PyYAML", "neo4j", "requests"}
    print("DEPENDENCIES_PASS", deps, flush=True)

    responses, calls = [], []
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            size = int(self.headers["Content-Length"])
            if not 0 < size < 100000:
                self.send_error(400)
                return
            req = json.loads(self.rfile.read(size))
            calls.append(self.path)
            if self.path != "/v1/chat/completions" or req.get("model") != "fixture-model" or not responses:
                self.send_error(400)
                return
            data = json.dumps({"choices": [{"message": {"content": responses.pop(0)}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    driver = GraphDatabase.driver(uri, auth=("neo4j", password),
                                  connection_timeout=3, max_transaction_retry_time=0)
    try:
        deadline = time.monotonic() + 65
        while True:
            try:
                driver.verify_connectivity()
                break
            except Exception:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(1)
        def q(cypher, **params):
            with driver.session() as session:
                return session.run(cypher, **params).data()
        assert q("MATCH (n) RETURN count(n) AS n") == [{"n": 0}], "fixture not empty"
        db_version = driver.get_server_info().agent
        q("CREATE (t:Transcription {id:'clip',notes:'original'})-[:HAS_SEGMENT]->"
          "(:Segment {id:'s1',text:'Check the file',start:0.0,end:1.0})")
        with tempfile.TemporaryDirectory(prefix="installed-acceptance-") as tmp:
            root = pathlib.Path(tmp)
            artifacts = root / "artifacts"
            env = {"PATH": os.environ["PATH"], "HOME": tmp,
                   "NEO4J_URI": uri, "NEO4J_USER": "neo4j", "NEO4J_PASSWORD": password,
                   "LLM_ENDPOINT": f"http://127.0.0.1:{server.server_port}/v1",
                   "LLM_MODEL": "fixture-model", "OLLAMA_HOST": "http://127.0.0.1:1",
                   "OLLAMA_USE_CLI": "0", "OLLAMA_TIMEOUT": "3",
                   "SUMMARY_FAILOPEN": "copy", "ARTIFACTS_DIR": str(artifacts),
                   "PYTHONDONTWRITEBYTECODE": "1", "NO_PROXY": "*"}
            if target:
                env["PYTHONPATH"] = str(target)
            def cli(*argv):
                result = subprocess.run([sys.executable, "-m", "summarize_from_segments", *argv],
                                        cwd=tmp, env=env, capture_output=True, text=True, timeout=30)
                assert result.returncode == 0, "installed CLI failed"
            def counts():
                return [q("MATCH (n:" + label + ") RETURN count(n) AS n")[0]["n"]
                        for label in ("Summary", "Task", "Run")]
            responses.append('{"summary":"","bullets":[]}')
            cli("--id", "clip", "--write-notes")
            assert counts() == [0, 0, 0] and not artifacts.exists()
            assert q("MATCH (t:Transcription) RETURN t.notes AS notes") == [{"notes": "original"}]
            assert len(calls) == 1 and not responses
            print("NEGATIVE_PASS", flush=True)
            good = {"summary": "Check the file", "bullets": []}
            task = {"title": "Check file", "description": "Review file", "priority": "MEDIUM",
                    "due": None, "confidence": 0.8, "acceptance": [
                        {"type": "file_exists", "args": {"path": "artifacts/{TASK_ID}/output.txt"}}]}
            responses.extend([json.dumps(good), json.dumps({"tasks": [task]})])
            cli("--id", "clip", "--write-notes")
            assert counts() == [1, 1, 0] and q("MATCH (t:Task) RETURN t.status AS status") == [{"status": "REVIEW"}]
            assert len(calls) == 3 and not responses
            cli("--approve-all", "1", "--execute-ready", "1")
            rows = q("MATCH (t:Task)-[:HAS_RUN]->(r:Run) RETURN t.id AS id,t.status AS status,"
                     "t.active_run_id AS owner,r.id AS rid,r.status AS rs")
            assert len(rows) == 1 and rows[0]["status"] == rows[0]["rs"] == "DONE" and rows[0]["owner"] is None
            row = rows[0]
            d = artifacts / row["id"]
            data = (d / ("run-" + row["rid"] + ".txt")).read_bytes()
            receipt = (d / ("run-" + row["rid"] + ".json")).read_bytes()
            assert inspect_local_output(receipt, data, task_id=row["id"], run_id=row["rid"])["classification"] == "LOCAL_OUTPUT_BYTES_MATCH"
            cli("--execute-ready", "1")
            assert counts() == [1, 1, 1] and len(calls) == 3
            print("POSITIVE_PASS", flush=True)
            operator_cases = verify_operator_report(root, env, target)
            combined_cases = verify_operation_cli(root, env, target, q, port, password)
        report = {"schema": 1, "commit": args.commit, "wheel": wheel.name,
                  "wheel_bytes": wheel.stat().st_size, "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
                  "python": sys.version.split()[0], "dependencies": deps, "neo4j": db_version,
                  "negative_counts": [0, 0, 0], "terminal_counts": [1, 1, 1],
                  "fixture_http_calls": len(calls), "model_inference_calls": 0,
                  "local_receipts_verified": 1, "operator_report_cases": operator_cases,
                  "combined_operation_report_cases": combined_cases,
                  "production_read_only_rbac_verified": False,
                  "acceptance": "PASS",
                  "production_activation_allowed": False}
        with manifest.open("x", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)
            f.write("\n")
        print("INSTALLED_WHEEL_ACCEPTANCE_PASS", flush=True)
    finally:
        driver.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


if __name__ == "__main__":
    main()
