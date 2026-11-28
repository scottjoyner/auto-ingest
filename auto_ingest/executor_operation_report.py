"""Bounded, read-only reconciliation of one exact task, run and HTTP target.

Caller supplies an admitted driver and trusted artifact directory. No discovery,
dispatch, mutation, or replay. Graph and file observations are not atomic.
"""
from __future__ import annotations

import json
import os

from auto_ingest.executor_cutover import _index_only
from auto_ingest.executor_evidence import (
    _http_identity,
    _http_path,
    inspect_http_observation,
    inspect_local_output,
)
from auto_ingest.executor_http_report import _read_evidence

_TASK = """MATCH (t:Task {id:$task_id}) USING INDEX t:Task(id)
RETURN elementId(t) AS identity, t.status AS status, t.active_run_id AS owner,
       t.acceptance_json AS acceptance LIMIT 3"""
_RUN = """MATCH (t) WHERE elementId(t) = $identity
OPTIONAL MATCH (t)-[:HAS_RUN]->(r:Run)
WHERE r.id = $run_id OR r.id = t.active_run_id OR r.status = 'RUNNING'
RETURN r.id AS rid, r.status AS status LIMIT 3"""


def _rows(tx, query, **params):
    result = tx.run(query, **params)
    rows = list(result)
    result.consume()
    if len(rows) > 3:
        raise RuntimeError("unexpected graph cardinality")
    return rows


def reconcile_operation(driver, directory, *, task_id, run_id, url):
    """Bind exact graph ownership and local evidence; always deny activation/replay.

    Requires Task(id) index and independent resource admission. The supplied
    URL must occur in the task's HTTP acceptance contract. File reads are
    bounded by the existing receipt limit; an oversized output fails closed.
    """
    identity = _http_identity(task_id, run_id, url)
    result = {"task_id": task_id, "run_id": run_id,
              "target_sha256": identity["target_sha256"],
              "graph_classification": "GRAPH_UNVERIFIED",
              "local_output": "NOT_READ", "http": None,
              "observation_authenticity": "UNVERIFIED",
              "atomic_snapshot": False, "remote_completion_verified": False,
              "activation_allowed": False, "replay_allowed": False}
    with driver.session(default_access_mode="READ") as session:
        with session.begin_transaction(timeout=5) as tx:
            plan = tx.run("EXPLAIN " + _TASK, task_id=task_id).consume().plan
            if not _index_only(plan):
                raise RuntimeError("Task(id) index-only lookup not verified")
            tasks = _rows(tx, _TASK, task_id=task_id)
            if len(tasks) != 1:
                result["graph_classification"] = ("TASK_NOT_FOUND" if not tasks else
                                                  "AMBIGUOUS_TASK")
                return result
            task = tasks[0]
            if not isinstance(task["identity"], str) or not task["identity"]:
                return result
            contract = task["acceptance"]
            if not isinstance(contract, str) or len(contract) > 16384:
                result["graph_classification"] = "HTTP_CONTRACT_UNVERIFIED"
                return result
            try:
                def unique(pairs):
                    obj = {}
                    for k, v in pairs:
                        if k in obj:
                            raise ValueError("duplicate contract key")
                        obj[k] = v
                    return obj
                checks = json.loads(contract, object_pairs_hook=unique)
                matches = isinstance(checks, list) and any(
                    isinstance(x, dict) and x.get("type") == "http_ok"
                    and isinstance(x.get("args"), dict) and x["args"].get("url") == url
                    for x in checks)
            except (ValueError, TypeError, RecursionError):
                matches = False
            if not matches:
                result["graph_classification"] = "HTTP_CONTRACT_UNVERIFIED"
                return result
            runs = _rows(tx, _RUN, identity=task["identity"], run_id=run_id)
            if len(runs) != 1 or runs[0]["rid"] != run_id:
                result["graph_classification"] = "RUN_LINK_UNVERIFIED"
                return result
            status, owner, run_status = task["status"], task["owner"], runs[0]["status"]
            if status == run_status == "RUNNING" and owner == run_id:
                result["graph_classification"] = "RUNNING_OWNER_MATCH"
            elif status in ("DONE", "FAILED") and status == run_status and owner is None:
                result["graph_classification"] = "TERMINAL_RUN_MATCH"
            else:
                result["graph_classification"] = "GRAPH_STATE_MISMATCH"
                return result
            tx.commit()
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        local = _read_evidence(fd, f"run-{run_id}.txt")
        receipt = _read_evidence(fd, f"run-{run_id}.json")
        intent = _read_evidence(fd, _http_path(".", identity, "intent").name)
        outcome = _read_evidence(fd, _http_path(".", identity, "outcome").name)
    finally:
        os.close(fd)
    result["local_output"] = (inspect_local_output(receipt, local, task_id=task_id,
                             run_id=run_id)["classification"] if local is not None
                             else "LOCAL_OUTPUT_UNVERIFIED")
    result["http"] = inspect_http_observation(intent, outcome, task_id=task_id,
                                              run_id=run_id, url=url)
    return result
