from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from auto_ingest.content import build_summaries as bs


class _Response:
    def __init__(self, status=200, body=b'{"response":"ok"}'):
        self.status = status
        self._body = body

    def read(self):
        return self._body


class _Connection:
    responses = []
    calls = []

    def __init__(self, host, port, timeout):
        self.host = host
        self.port = port
        self.timeout = timeout

    def request(self, method, path, body=None, headers=None):
        self.calls.append((method, path, body, headers))

    def getresponse(self):
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _state(**overrides):
    value = {
        "auto_throttle": False,
        "sleep_min": 0.0,
        "sleep_max": 0.0,
        "prompt_chars": 1000,
        "last_duration": 0.0,
        "consec_fail": 0,
    }
    value.update(overrides)
    return value


def test_ollama_client_url_and_success(monkeypatch):
    _Connection.calls = []
    _Connection.responses = [_Response(body=b'{"response":"hello"}')]
    monkeypatch.setattr(bs.http.client, "HTTPConnection", _Connection)
    client = bs.OllamaClient("http://example.test:1234/base")
    assert client._host == "example.test"
    assert client._port == 1234
    assert client._base_path == "/base"
    assert client.generate("m", "p", options={"x": 1}) == "hello"
    method, path, body, headers = _Connection.calls[0]
    assert method == "POST"
    assert path == "/base/api/generate"
    assert json.loads(body)["options"] == {"x": 1}
    assert headers["Content-Type"] == "application/json"


def test_ollama_client_https_default_port_retry_and_failures(monkeypatch):
    _Connection.responses = [
        _Response(status=500, body=b"bad"),
        _Response(body=b'{"text":"fallback"}'),
    ]
    monkeypatch.setattr(bs.http.client, "HTTPSConnection", _Connection)
    monkeypatch.setattr(bs.time, "sleep", lambda _n: None)
    client = bs.OllamaClient("https://secure.test")
    assert client._is_https is True
    assert client._port == 443
    assert client.generate("m", "p", retries=2) == "fallback"

    _Connection.responses = [_Response(body=b"{}")]
    with pytest.raises(RuntimeError, match="after retries"):
        client.generate("m", "p", retries=1)


def test_stem_text_csv_and_json_extract_helpers(tmp_path):
    good = tmp_path / "2026_0102_030405_large-v3_transcription.txt"
    good.write_text(" hello ", encoding="utf-8")
    assert bs._detect_stem(good) == "2026_0102_030405"
    assert bs._detect_stem(tmp_path / "ordinary.txt") is None
    assert bs._read_text_file(good) == "hello"
    assert bs._read_text_file(tmp_path / "missing") is None

    csv_path = tmp_path / "x.csv"
    csv_path.write_text("Text,Other\nhello,x\nworld,y\n", encoding="utf-8")
    assert bs._read_csv_concat_text(csv_path) == "hello\nworld"
    csv_fallback = tmp_path / "y.csv"
    csv_fallback.write_text("A,B\none,two\n", encoding="utf-8")
    assert bs._read_csv_concat_text(csv_fallback) == "one two"
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    assert bs._read_csv_concat_text(empty) is None
    assert bs._read_csv_concat_text(tmp_path / "none.csv") is None

    assert bs._extract_first_json_object('{"a":{"b":1}} trailing') == '{"a":{"b":1}}'
    assert bs._extract_first_json_object('```json\n{"a":"}"}\n```') == '{"a":"}"}'
    assert bs._extract_first_json_object("no object") is None


def test_payload_and_task_normalization():
    payload = bs._normalize_payload("summary text")
    assert payload["summary"] == "summary text"
    assert payload["version"] == "1.0"
    assert payload["tasks"] == []
    payload = bs._normalize_payload(["bad", {"summary": "ok", "tasks": "bad"}])
    assert payload["summary"] == "ok"
    assert payload["tasks"] == []
    payload = bs._normalize_payload(3)
    assert payload["summary"] == ""

    task = {
        "title": "  Do thing  ",
        "description": " details ",
        "labels": ["A", "A", "", 3],
        "priority": "INVALID",
        "owner_hint": " DevOps ",
        "agent": {"name": " Bot ", "confidence": 5, "rationale": " why "},
        "plan": [
            {
                "step": "2",
                "action": "act",
                "tool": "tool",
                "operation": "op",
                "inputs": {"x": 1},
                "expected_output": "done",
            },
            "skip",
        ],
    }
    normalized = bs._normalize_tasks(["skip", task])
    assert len(normalized) == 1
    assert normalized[0]["title"] == "Do thing"
    assert normalized[0]["priority"] == "medium"
    assert normalized[0]["agent"]["confidence"] == 1.0
    assert normalized[0]["plan"][0]["step"] == 2
    assert bs._normalize_tasks("bad") == []
    assert bs._normalize_tasks([{"title": "", "description": ""}]) == []


def test_choose_discover_sleep_json_vtt(monkeypatch, tmp_path):
    stem = "2026_0102_030405"
    preferred = tmp_path / f"{stem}_large-v3_transcription.txt"
    fallback = tmp_path / f"{stem}.txt"
    preferred.write_text("x", encoding="utf-8")
    fallback.write_text("y", encoding="utf-8")
    assert bs._choose_best_path(tmp_path, stem, [fallback, preferred]) == preferred
    assert bs._choose_best_path(tmp_path, stem, []) == preferred
    preferred.unlink()
    fallback.unlink()
    assert bs._choose_best_path(tmp_path, stem, []) is None

    nested = tmp_path / "nested"
    nested.mkdir()
    transcript = nested / f"{stem}.txt"
    transcript.write_text("x", encoding="utf-8")
    groups = bs._discover_stems([tmp_path, tmp_path / "missing"])
    assert any(group[1] == stem for group in groups)

    sleeps = []
    monkeypatch.setattr(bs.time, "sleep", sleeps.append)
    state = {"auto_throttle": True, "sleep_min": 0.1, "sleep_max": 1.0, "consec_fail": 2}
    bs._auto_sleep(state, True, 4)
    assert state["consec_fail"] == 0
    assert sleeps[-1] == 1.0
    bs._auto_sleep(state, False, 0)
    assert state["consec_fail"] == 1
    state["auto_throttle"] = False
    before = len(sleeps)
    bs._auto_sleep(state, True, 0)
    assert len(sleeps) == before

    doc = tmp_path / "doc.json"
    doc.write_text('{"a":1}', encoding="utf-8")
    assert bs._read_json_file(doc) == {"a": 1}
    doc.write_text("[]", encoding="utf-8")
    assert bs._read_json_file(doc) is None
    assert bs._read_json_file(tmp_path / "no.json") is None

    vtt = tmp_path / "x.vtt"
    vtt.write_text(
        "WEBVTT\n\n1\n00:00:01.000 --> 00:00:02.000\nHello\n"
        "00:03.000 --> 00:04.000\nWorld\n",
        encoding="utf-8",
    )
    assert bs._read_vtt_to_text(vtt) == "Hello\nWorld"
    assert bs._read_vtt_to_text(tmp_path / "missing.vtt") is None


def test_enrich_agent_and_plan_routes_domains():
    tasks = [
        {"title": "deploy cluster", "description": "", "labels": ["Kubernetes"], "owner_hint": ""},
        {"title": "generic", "description": "", "labels": [], "owner_hint": "Unknown"},
        {
            "title": "keep",
            "description": "",
            "labels": [],
            "owner_hint": "",
            "agent": {"name": "Custom"},
            "plan": [{"step": 1}],
        },
    ]
    result = bs._enrich_tasks_with_agent_plan(tasks)
    assert result[0]["agent"]["name"] == "DevOpsAgent"
    assert result[0]["plan"][0]["step"] == 1
    assert result[1]["agent"]["name"]
    assert result[1]["plan"]
    assert result[2]["agent"]["name"] == "Custom"
    assert result[2]["plan"] == [{"step": 1}]


class _Client:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.prompts = []

    def generate(self, model, prompt, options=None):
        self.prompts.append((model, prompt, options))
        item = self.outputs.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_process_transcript_new_summary_and_tasks(monkeypatch, tmp_path):
    monkeypatch.setattr(bs.time, "sleep", lambda _n: None)
    stem = "2026_0102_030405"
    path = tmp_path / f"{stem}.txt"
    path.write_text("A transcript that contains deploy work.", encoding="utf-8")
    raw = json.dumps(
        {
            "summary": "Summary",
            "key_points": ["one"],
            "tasks": [
                {
                    "title": "Deploy service",
                    "description": "Ship it",
                    "labels": ["DevOps"],
                }
            ],
        }
    )
    client = _Client([raw])
    state = _state()
    bs.process_transcript(path, client, "model", {"n": 1}, state)
    summary = json.loads((tmp_path / f"{stem}_summary.json").read_text(encoding="utf-8"))
    tasks = json.loads((tmp_path / f"{stem}_tasks.json").read_text(encoding="utf-8"))
    assert summary["summary"] == "Summary"
    assert summary["_meta"]["source_transcript"] == str(path)
    assert tasks["tasks"][0]["agent"]["name"] == "DevOpsAgent"
    assert client.prompts


def test_process_transcript_existing_summary_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(bs.time, "sleep", lambda _n: None)
    stem = "2026_0102_030405"
    path = tmp_path / f"{stem}.txt"
    path.write_text("text", encoding="utf-8")
    summary_path = tmp_path / f"{stem}_summary.json"
    tasks_path = tmp_path / f"{stem}_tasks.json"
    summary_path.write_text(
        json.dumps(
            {
                "summary": "Existing",
                "tasks": [{"title": "Pay invoice", "description": "Do it", "labels": ["invoice"]}],
            }
        ),
        encoding="utf-8",
    )
    client = _Client([])
    bs.process_transcript(path, client, "model", {}, _state())
    assert tasks_path.exists()
    assert client.prompts == []

    tasks_path.unlink()
    summary_path.write_text(json.dumps({"summary": "Need action", "tasks": []}), encoding="utf-8")
    client = _Client([json.dumps({"tasks": [{"title": "Call", "description": "someone"}]})])
    bs.process_transcript(path, client, "model", {}, _state())
    assert json.loads(tasks_path.read_text(encoding="utf-8"))["tasks"][0]["title"] == "Call"

    # Fast skip when both sidecars already exist.
    no_calls = _Client([])
    bs.process_transcript(path, no_calls, "model", {}, _state())
    assert no_calls.prompts == []


def test_process_transcript_dry_run_failures_and_bad_json(monkeypatch, tmp_path):
    monkeypatch.setattr(bs.time, "sleep", lambda _n: None)
    stem = "2026_0102_030405"
    path = tmp_path / f"{stem}.txt"
    path.write_text("text", encoding="utf-8")
    state = _state()
    bs.process_transcript(path, _Client([]), "m", {}, state, dry_run=True)
    assert not (tmp_path / f"{stem}_summary.json").exists()

    bs.process_transcript(path, _Client([RuntimeError("offline")]), "m", {}, state)
    assert state["consec_fail"] == 1

    state["consec_fail"] = 0
    bs.process_transcript(path, _Client(["definitely not json"]), "m", {}, state)
    assert (tmp_path / f"{stem}_summary.json.bad.txt").exists()

    bad_name = tmp_path / "ordinary.txt"
    bad_name.write_text("x", encoding="utf-8")
    assert bs.process_transcript(bad_name, _Client([]), "m", {}, state) is None


def test_process_json_csv_vtt_input_loading(monkeypatch, tmp_path):
    monkeypatch.setattr(bs.time, "sleep", lambda _n: None)
    stem = "2026_0102_030405"
    inputs = [
        (tmp_path / f"{stem}.json", json.dumps({"segments": [{"text": "one"}, {"text": "two"}]})),
        (tmp_path / f"{stem}.csv", "Text\nhello\n"),
        (tmp_path / f"{stem}.vtt", "WEBVTT\n\n00:00.000 --> 00:01.000\nhello\n"),
    ]
    for path, body in inputs:
        path.write_text(body, encoding="utf-8")
        client = _Client([json.dumps({"summary": "ok", "tasks": []})])
        bs.process_transcript(path, client, "m", {}, _state(), overwrite=True)
        assert client.prompts
        # remove generated files so each suffix exercises the loader path
        (tmp_path / f"{stem}_summary.json").unlink(missing_ok=True)
        (tmp_path / f"{stem}_tasks.json").unlink(missing_ok=True)
