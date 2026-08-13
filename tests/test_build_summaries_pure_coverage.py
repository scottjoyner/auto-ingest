from __future__ import annotations

import json

import pytest

from auto_ingest.content import build_summaries as b


def test_stem_and_text_readers(tmp_path):
    p = tmp_path / "2026_0813_123456_transcription.txt"
    p.write_text(" hello \n", encoding="utf-8")
    assert b._detect_stem(p) == "2026_0813_123456"
    assert b._detect_stem(tmp_path / "no-date.txt") is None
    assert b._read_text_file(p) == "hello"
    assert b._read_text_file(tmp_path / "missing") is None

    csvp = tmp_path / "x.csv"
    csvp.write_text("Text,Other\nhello,a\nworld,b\n", encoding="utf-8")
    assert b._read_csv_concat_text(csvp) == "hello\nworld"
    fallback = tmp_path / "f.csv"
    fallback.write_text("A,B\none,two\n", encoding="utf-8")
    assert b._read_csv_concat_text(fallback) == "one two"
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    assert b._read_csv_concat_text(empty) is None


def test_json_extraction_and_payload_normalization():
    raw = 'prefix {"a":{"b":"} escaped \\" ok"},"c":1} suffix'
    obj = b._extract_first_json_object(raw)
    assert obj and json.loads(obj)["c"] == 1
    assert json.loads(b._extract_first_json_object('```json\n{"x":1}\n```')) == {"x": 1}
    assert b._extract_first_json_object("none") is None

    payload = b._normalize_payload("summary only")
    assert payload["summary"] == "summary only"
    assert payload["tasks"] == [] and payload["version"] == "1.0"
    payload = b._normalize_payload([None, {"summary": "x", "tasks": [{"title": "do"}]}])
    assert payload["summary"] == "x" and payload["tasks"][0]["title"] == "do"
    assert b._normalize_payload(4)["summary"] == ""


def test_task_normalization_covers_validation_limits_and_plan():
    assert b._normalize_tasks(None) == []
    tasks = b._normalize_tasks(
        [
            None,
            {},
            {
                "title": "  Ship it  ",
                "description": " desc ",
                "labels": [" A ", "A", "", 2],
                "priority": "nonsense",
                "owner_hint": "Ops",
                "agent": {"name": "Bot", "confidence": "2", "rationale": "why"},
                "plan": [
                    {"step": "bad", "action": "a", "tool": "t", "operation": "op", "inputs": {"x": 1}, "expected_output": "ok"},
                    "bad",
                    {"step": 3, "action": "b", "inputs": "bad"},
                ],
            },
            {"description": "desc-only", "priority": "urgent", "agent": {"confidence": "bad"}},
        ]
    )
    assert len(tasks) == 2
    first = tasks[0]
    assert first["title"] == "Ship it"
    assert first["labels"] == ["A", "2"]
    assert first["priority"] == "medium"
    assert first["agent"]["confidence"] == 1.0
    assert first["plan"][0]["step"] == 1 and first["plan"][0]["inputs"] == {"x": 1}
    assert first["plan"][1]["step"] == 3 and first["plan"][1]["inputs"] == {}
    assert tasks[1]["priority"] == "urgent" and tasks[1]["agent"]["confidence"] == 0.0


def test_discovery_choice_throttle_and_json_vtt(tmp_path, monkeypatch):
    root = tmp_path / "root"
    directory = root / "day"
    directory.mkdir(parents=True)
    preferred = directory / "2026_0813_123456_transcription.txt"
    preferred.write_text("x", encoding="utf-8")
    alt = directory / "2026_0813_123456.vtt"
    alt.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHello\n1\nWorld\n", encoding="utf-8")
    buckets = b._discover_stems([root, tmp_path / "missing"])
    assert len(buckets) == 1 and buckets[0][1] == "2026_0813_123456"
    assert b._choose_best_path(directory, buckets[0][1], buckets[0][2]) == preferred
    assert b._choose_best_path(directory, "none", []) is None

    state = {"auto_throttle": True, "sleep_min": 0.1, "sleep_max": 1.0}
    slept = []
    monkeypatch.setattr(b.time, "sleep", lambda n: slept.append(n))
    b._auto_sleep(state, False, 2)
    assert state["consec_fail"] == 1 and slept[-1] == pytest.approx(0.9)
    b._auto_sleep(state, True, 0)
    assert state["consec_fail"] == 0 and slept[-1] == pytest.approx(0.1)

    jp = tmp_path / "j.json"
    jp.write_text('{"x":1}', encoding="utf-8")
    assert b._read_json_file(jp) == {"x": 1}
    jp.write_text("[]", encoding="utf-8")
    assert b._read_json_file(jp) is None
    assert b._read_json_file(tmp_path / "bad.json") is None
    assert b._read_vtt_to_text(alt) == "Hello\nWorld"


def test_agent_enrichment_loaders_atomic_and_tasks_document(tmp_path):
    tasks = [
        {"title": "deploy kubernetes cluster", "description": "", "labels": ["DevOps"], "owner_hint": "", "agent": {}, "plan": []},
        {"title": "generic", "description": "", "labels": [], "owner_hint": "Someone", "agent": {"name": "Custom", "confidence": 0.7}, "plan": []},
    ]
    out = b._enrich_tasks_with_agent_plan(tasks)
    assert out[0]["agent"]["name"] == "DevOpsAgent" and out[0]["plan"][0]["step"] == 1
    assert out[1]["agent"]["name"] == "Custom" and out[1]["plan"]

    csvp = tmp_path / "t.csv"
    csvp.write_text("text\na\nb\n", encoding="utf-8")
    assert b._load_transcript_text(csvp) == "a\nb"
    vtt = tmp_path / "t.vtt"
    vtt.write_text("WEBVTT\n00:00.000 --> 00:01.000\nhello\n", encoding="utf-8")
    assert b._load_transcript_text(vtt) == "hello"
    for payload, expected in [
        ({"text": " hello "}, "hello"),
        ({"transcript": " tr "}, "tr"),
        ({"segments": [{"text": " a "}, {"text": "b"}, None]}, "ab"),
    ]:
        p = tmp_path / "t.json"
        p.write_text(json.dumps(payload), encoding="utf-8")
        assert b._load_transcript_text(p) == expected
    txt = tmp_path / "plain.txt"
    txt.write_text("plain", encoding="utf-8")
    assert b._load_transcript_text(txt) == "plain"

    dest = tmp_path / "out.json"
    b._atomic_json(dest, {"a": 1})
    assert json.loads(dest.read_text()) == {"a": 1}
    assert not (tmp_path / "out.json.tmp").exists()
    doc = b._tasks_document(out, dest, "model", derived_from=txt)
    assert doc["_meta"]["schema"] == "tasks.sidecar.v1"
    assert doc["_meta"]["derived_from"] == str(txt)
