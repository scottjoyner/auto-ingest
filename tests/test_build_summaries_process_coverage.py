from __future__ import annotations

import json

from auto_ingest.content import build_summaries as b


class Client:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def generate(self, model, prompt, options=None):
        self.calls.append((model, prompt, options))
        if self.error:
            raise self.error
        return self.response


def state(**kw):
    value = {
        "auto_throttle": False,
        "sleep_min": 0.0,
        "sleep_max": 1.0,
        "prompt_chars": 100,
        "last_duration": 0.0,
        "consec_fail": 0,
    }
    value.update(kw)
    return value


def test_process_rejects_unstemmed_and_skips_existing(tmp_path, monkeypatch):
    client = Client("{}")
    current = state()
    monkeypatch.setattr(b, "_auto_sleep", lambda *a: None)
    b.process_transcript(tmp_path / "plain.txt", client, "m", {}, current)
    assert not client.calls
    path = tmp_path / "2026_0813_123456_transcription.txt"
    path.write_text("hello", encoding="utf-8")
    (tmp_path / "2026_0813_123456_summary.json").write_text("{}", encoding="utf-8")
    (tmp_path / "2026_0813_123456_tasks.json").write_text("{}", encoding="utf-8")
    b.process_transcript(path, client, "m", {}, current)
    assert not client.calls


def test_process_derives_tasks_from_existing_summary_without_model(tmp_path, monkeypatch):
    path = tmp_path / "2026_0813_123456_transcription.txt"
    path.write_text("hello", encoding="utf-8")
    summary = tmp_path / "2026_0813_123456_summary.json"
    summary.write_text(
        json.dumps({"summary": "x", "tasks": [{"title": "deploy", "labels": ["DevOps"]}]}),
        encoding="utf-8",
    )
    client = Client("{}")
    monkeypatch.setattr(b, "_auto_sleep", lambda *a: None)
    b.process_transcript(path, client, "m", {}, state())
    assert not client.calls
    doc = json.loads((tmp_path / "2026_0813_123456_tasks.json").read_text())
    assert doc["tasks"][0]["agent"]["name"] == "DevOpsAgent"
    assert doc["_meta"]["derived_from"] == str(summary)


def test_process_existing_summary_model_tasks_success_dryrun_and_failures(tmp_path, monkeypatch):
    path = tmp_path / "2026_0813_123456_transcription.txt"
    path.write_text("hello", encoding="utf-8")
    summary = tmp_path / "2026_0813_123456_summary.json"
    summary.write_text(json.dumps({"summary": "summary", "key_points": ["one"], "tasks": []}), encoding="utf-8")
    monkeypatch.setattr(b, "_auto_sleep", lambda *a: None)

    dry = Client("{}")
    b.process_transcript(path, dry, "m", {}, state(), dry_run=True)
    assert not dry.calls
    client = Client('prefix {"tasks":[{"title":"pay invoice","labels":["invoice"]}]} tail')
    b.process_transcript(path, client, "m", {"x": 1}, state())
    assert client.calls and "summary" in client.calls[0][1].lower()
    task = json.loads((tmp_path / "2026_0813_123456_tasks.json").read_text())["tasks"][0]
    assert task["agent"]["name"] == "FinanceAgent"

    (tmp_path / "2026_0813_123456_tasks.json").unlink()
    client = Client("not json")
    b.process_transcript(path, client, "m", {}, state())
    assert (tmp_path / "2026_0813_123456_tasks.json.bad.txt").exists()

    (tmp_path / "2026_0813_123456_tasks.json.bad.txt").unlink()
    client = Client(error=RuntimeError("boom"))
    current = state()
    b.process_transcript(path, client, "m", {}, current)
    assert current["consec_fail"] == 1


def test_process_full_generation_success_bad_json_empty_and_truncation(tmp_path, monkeypatch):
    path = tmp_path / "2026_0813_123456_transcription.txt"
    path.write_text("A" * 200, encoding="utf-8")
    monkeypatch.setattr(b.time, "sleep", lambda n: None)
    payload = {"summary": "ok", "tasks": [{"title": "nda contract", "labels": ["contract"]}], "topics": ["t"]}
    client = Client("noise " + json.dumps(payload) + " tail")
    current = state(prompt_chars=40, sleep_min=0.1)
    b.process_transcript(path, client, "model", {"temperature": 0}, current)
    assert len(client.calls[0][1]) < len(b.PROMPT_SCHEMA) + 200 + len(b.PROMPT_END)
    summary = json.loads((tmp_path / "2026_0813_123456_summary.json").read_text())
    tasks = json.loads((tmp_path / "2026_0813_123456_tasks.json").read_text())
    assert summary["_meta"]["schema"] == "summary.sidecar.v1"
    assert tasks["tasks"][0]["agent"]["name"] == "LegalAgent"

    client = Client("garbage")
    b.process_transcript(path, client, "m", {}, state(), overwrite=True)
    assert (tmp_path / "2026_0813_123456_summary.json.bad.txt").exists()

    empty = tmp_path / "2026_0813_223456_transcription.txt"
    empty.write_text("   ", encoding="utf-8")
    client = Client("{}")
    b.process_transcript(empty, client, "m", {}, state())
    assert not client.calls
    client = Client("{}")
    b.process_transcript(path, client, "m", {}, state(), overwrite=True, dry_run=True)
    assert not client.calls
    client = Client(error=RuntimeError("boom"))
    current = state()
    b.process_transcript(path, client, "m", {}, current, overwrite=True)
    assert current["consec_fail"] == 1


def test_main_invalid_roots_and_limit(monkeypatch, tmp_path):
    root = tmp_path / "r"
    root.mkdir()
    for stamp in ("2026_0813_123456", "2026_0813_223456"):
        (root / f"{stamp}_transcription.txt").write_text("hello", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["prog", "--roots", str(root), "--dry-run", "--limit", "1"])
    b.main()
    monkeypatch.setattr("sys.argv", ["prog", "--roots", str(tmp_path / "missing")])
    b.main()
