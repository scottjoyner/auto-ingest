"""Exercise the actual production admission functions without model or DB access."""
import ast
import json
import pathlib
import re
import unittest
from contextlib import nullcontext
from typing import Any, Dict, List, Optional


def harness(responses=(), *, copy=False, parts=None):
    source = pathlib.Path(__file__).resolve().parents[1] / "summarize_from_segments.py"
    names = {"_coerce_json_dict", "_normalize_summary_obj", "_failopen_summary_from_text",
             "summarize_as_json", "extract_tasks_json", "process_one"}
    nodes = [n for n in ast.parse(source.read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names
    calls, writes = [], []
    pending = iter(responses)
    def model(*args):
        calls.append(args)
        return next(pending)
    class Driver:
        def session(self):
            return nullcontext(object())
    row = {"t": {"id": "clip"}, "segments": [{"text": "check the file"}], "utterances": []}
    ns = dict(json=json,re=re,Any=Any,Dict=Dict,List=List,Optional=Optional,
              SUMMARY_FAILOPEN="copy" if copy else "skip", SUMMARIZE_INSTR="summary",
              TASKS_INSTR="tasks", chunk=lambda text: parts if parts is not None else [text],
              ollama_chat_json=model,neo_driver=Driver,fetch_one=lambda *a,**k:row,
              build_transcript=lambda *a,**k:"check the file",
              write_back=lambda *a,**k:writes.append((a,k)) or "summary-id")
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),"exec"),ns)
    return ns,calls,writes


GOOD={"summary":"Check the file","bullets":[]}
TASK={"title":"Check file","description":"Review the file","priority":"MEDIUM",
      "due":None,"confidence":0.8,
      "acceptance":[{"type":"file_exists","args":{"path":"artifacts/{TASK_ID}/output.txt"}}]}


class ModelAdmissionTests(unittest.TestCase):
    def test_strict_json(self):
        ns,_,_=harness()
        parse=ns["_coerce_json_dict"]
        for bad in ('{"summary":"unterminated', 'prose {"summary":"ok"}', '{"summary":"ok"} trailing',
                    '{"summary":"a","summary":"b"}','{"confidence":NaN}','[]'):
            with self.subTest(bad=bad):
                self.assertEqual(parse(bad),{})
        self.assertEqual(parse('{"summary":"ok","bullets":[]}'),{"summary":"ok","bullets":[]})
        self.assertEqual(parse('```json\n{"summary":"ok"}\n```'),{"summary":"ok"})

    def test_empty_and_wrong_types(self):
        for bad in ({},None,[],{"summary":"","bullets":[]},{"summary":"  ","bullets":[]},
                    {"summary":None,"bullets":[]},{"summary":42,"bullets":[]},
                    {"summary":"ok","bullets":"not a list"},{"summary":"ok","bullets":[3]},
                    {"summary":"ok"}):
            with self.subTest(bad=bad):
                ns,calls,writes=harness([bad])
                self.assertIsNone(ns["process_one"](object(),{"id":"clip"},False,False,True))
                self.assertEqual(len(calls),1)
                self.assertEqual(writes,[])

    def test_copy_diagnostic_never_extracts_or_writes(self):
        ns,calls,writes=harness([{}],copy=True)
        self.assertIsNone(ns["process_one"](object(),{"id":"clip"},False,False,True))
        self.assertEqual(len(calls),1)
        self.assertEqual(writes,[])
        ns,calls,_=harness([{}],copy=True)
        copied=ns["summarize_as_json"]("check the file")
        self.assertEqual(copied["summary_source"],"TRANSCRIPT_COPY_DIAGNOSTIC")
        self.assertEqual(copied["summary"],"check the file")
        self.assertIsNone(ns["extract_tasks_json"](copied))
        self.assertEqual(len(calls),1)

    def test_bad_partial_stops_reduce(self):
        ns,calls,writes=harness([GOOD,{"summary":"","bullets":[]}],parts=["a","b"])
        self.assertIsNone(ns["process_one"](object(),{"id":"clip"},False,False,True))
        self.assertEqual(len(calls),2)
        self.assertEqual(writes,[])

    def test_bad_task_batch_no_write(self):
        for bad in ({}, {"tasks":None}, {"tasks":[TASK,{}]},
                    {"tasks":[dict(TASK,confidence=float("nan"))]},
                    {"tasks":[dict(TASK,confidence=True)]},
                    {"tasks":[dict(TASK,acceptance=[{"type":"http_ok","args":{"url":"https://example.com"}}])]},
                    {"tasks":[dict(TASK,acceptance=[])]}):
            with self.subTest(bad=bad):
                ns,calls,writes=harness([GOOD,bad,bad,bad])
                self.assertIsNone(ns["process_one"](object(),{"id":"clip"},False,False,True))
                self.assertEqual(len(calls),4)
                self.assertEqual(writes,[])

    def test_valid_summary_and_task_reaches_write_back(self):
        ns,calls,writes=harness([GOOD,{"summary":"invented replacement","tasks":[TASK]}])
        self.assertEqual(ns["process_one"](object(),{"id":"clip"},False,False,True),"summary-id")
        self.assertEqual(len(calls),2)
        self.assertEqual(len(writes),1)
        self.assertEqual(writes[0][0][2],dict(GOOD,tasks=[TASK],summary_source="MODEL_OUTPUT"))

    def test_explicit_empty_task_list_is_valid(self):
        ns,calls,writes=harness([GOOD,{"tasks":[]}])
        self.assertEqual(ns["process_one"](object(),{"id":"clip"},False,False,False),"summary-id")
        self.assertEqual(writes[0][0][2]["tasks"],[])

    def test_empty_transcript_no_call(self):
        ns,calls,_=harness([])
        self.assertEqual(ns["summarize_as_json"]("  ")["summary_source"],"INVALID_MODEL_OUTPUT")
        self.assertEqual(calls,[])
