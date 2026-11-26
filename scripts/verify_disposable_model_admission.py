"""Opt-in model admission acceptance on a disposable graph.

Reuse the pinned image, isolation, resource guard and teardown of the
media E2E harness. Model response fixtures are controlled, not inference.
"""
import ast
import json
import pathlib
import re
import sys
from typing import Any, Dict, List, Optional

import verify_disposable_ingest_e2e as fixture


def execute_fixture(driver, root):
    def q(cy, **params):
        with driver.session() as s:
            return s.run(cy, **params).data()

    assert q("MATCH (n) RETURN count(n) AS n") == [{"n": 0}]
    q("CREATE (:Transcription {id:'clip', notes:'original notes'})")
    q("MATCH (t:Transcription {id:'clip'}) CREATE (t)-[:HAS_SEGMENT]->(:Segment {id:'s1',text:'Check the file',start:0.0,end:1.0})")
    source = fixture.ROOT / "summarize_from_segments.py"
    names = {"_coerce_json_dict", "_normalize_summary_obj", "_failopen_summary_from_text",
             "summarize_as_json", "extract_tasks_json", "process_one", "chunk",
             "fetch_one", "fetch_segments_for", "build_transcript", "_serialize_tasks_for_db",
             "write_back", "q_get_task", "q_pick_ready", "q_attach_run", "q_finish_run",
             "q_approve_all_review", "approve_all", "execute_ready", "ensure_artifacts",
             "execute_task_minimal", "replace_placeholders", "acc_file_exists", "run_acceptance"}
    nodes = [n for n in ast.parse(source.read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names
    ns = dict(vars(fixture), SUMMARY_FAILOPEN="skip", SUMMARIZE_INSTR="summary",
              TASKS_INSTR="tasks", ARTIFACTS_DIR=str(root/"artifacts"),
              neo_driver=lambda: driver)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),"exec"),ns)
    ns["ACCEPTANCE_FUNCS"] = {"file_exists":ns["acc_file_exists"]}

    def snapshot():
        return (q("MATCH (s:Summary) RETURN count(s) AS n"),
                q("MATCH (t:Task) RETURN count(t) AS n"),
                q("MATCH (r:Run) RETURN count(r) AS n"),
                q("MATCH (t:Transcription {id:'clip'}) RETURN t.notes AS notes"),
                q("MATCH ()-[r:HAS_SUMMARY|GENERATED_TASK|HAS_RUN]->() RETURN count(r) AS n"))

    initial = snapshot()
    assert initial == ([{"n":0}],[{"n":0}],[{"n":0}],[{"notes":"original notes"}],[{"n":0}])
    good = {"summary":"Check the file","bullets":[]}
    task = {"title":"Check file","description":"Review the file","priority":"MEDIUM",
            "due":None,"confidence":0.8,"acceptance":[
                {"type":"file_exists","args":{"path":"artifacts/{TASK_ID}/output.txt"}}]}

    def run_case(raw_responses, mode="skip", parts=None):
        calls = []
        pending = iter(raw_responses)
        def model(*args):
            calls.append(args)
            return ns["_coerce_json_dict"](next(pending))
        ns["ollama_chat_json"] = model
        ns["SUMMARY_FAILOPEN"] = mode
        original_chunk = ns["chunk"]
        if parts is not None:
            ns["chunk"] = lambda text: parts
        try:
            with driver.session() as s:
                result = ns["process_one"](s,{"id":"clip"},False,False,True,skip_refetch=True)
        finally:
            ns["chunk"] = original_chunk
        return result, len(calls)

    invalid_batches = [
        ("malformed_json", ['{"summary":"unterminated'], "skip", None, 1),
        ("empty_summary", [json.dumps({"summary":"","bullets":[]})], "skip", None, 1),
        ("copy_fallback", ["{}"], "copy", None, 1),
        ("invalid_partial", [json.dumps(good),"{}"], "skip", ["a","b"], 2),
        ("mixed_task_batch", [json.dumps(good)]+[json.dumps({"tasks":[task,{}]})]*3,"skip",None,4),
        ("external_acceptance", [json.dumps(good)]+[json.dumps({"tasks":[dict(task,acceptance=[{"type":"http_ok","args":{"url":"https://example.invalid"}}]) ]})]*3,"skip",None,4),
        ("duplicate_keys", ['{"summary":"a","summary":"b","bullets":[]}'], "skip", None, 1)
    ]
    for name,raw,mode,parts,n in invalid_batches:
        sid,calls=run_case(raw,mode,parts)
        assert sid is None and calls==n
        assert snapshot()==initial, name
        assert not (root/"artifacts").exists()
        print("NEGATIVE_PASS",name,"calls",calls,"summary/task/run/edges=0/0/0/0; notes unchanged",flush=True)

    sid,calls=run_case([json.dumps(good),json.dumps({"summary":"replacement","tasks":[task]})])
    assert sid and calls==2
    assert snapshot()==([{"n":1}],[{"n":1}],[{"n":0}],[{"notes":"Check the file"}],[{"n":2}])
    rows=q("MATCH (t:Transcription {id:'clip'})-[:HAS_SUMMARY]->(s:Summary)-[:GENERATED_TASK]->(k:Task) RETURN s.text AS text,k.status AS status,k.id AS id,k.acceptance_json AS acceptance")
    assert len(rows)==1 and rows[0]["status"]=="REVIEW" and rows[0]["text"]=="Check the file"
    assert json.loads(rows[0]["acceptance"])==task["acceptance"]
    print("POSITIVE_PASS valid response -> exactly 1 Summary/1 REVIEW Task/0 Run; original summary preserved",flush=True)
    with driver.session() as s:
        assert ns["approve_all"](s,1)==1
    ns["execute_ready"](1)
    records=q("MATCH (k:Task)-[:HAS_RUN]->(r:Run) RETURN k.id AS id,k.status AS status,k.active_run_id AS owner,r.id AS rid,r.status AS rs")
    assert len(records)==1 and records[0]["status"]==records[0]["rs"]=="DONE" and records[0]["owner"] is None
    x=records[0];d=root/"artifacts"/x["id"]
    data=(d/("run-"+x["rid"]+".txt")).read_bytes()
    receipt=(d/("run-"+x["rid"]+".json")).read_bytes()
    assert fixture.inspect_local_output(receipt,data,task_id=x["id"],run_id=x["rid"])["classification"]=="LOCAL_OUTPUT_BYTES_MATCH"
    ns["execute_ready"](1)
    assert q("MATCH (r:Run) RETURN count(r) AS n")==[{"n":1}]
    print("POSITIVE_PASS approved fixture -> DONE; local receipt verified; repeat no additional Run",flush=True)
    print("MODEL_ADMISSION_REAL_GRAPH_PASS 7 negative + 2 positive groups",flush=True)


if __name__=="__main__":
    fixture.execute_fixture=execute_fixture
    fixture.main()
