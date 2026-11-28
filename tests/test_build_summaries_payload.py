"""Hermetic contract for the content-summary JSON payload normalizer.

Two failures are pinned here, both found on ``feat/pipeline-staging``:

* The module could not be imported at all -- ``_discover_stems`` uses
  ``Tuple`` in annotations but never imported it, so importing
  ``auto_ingest.content`` raised ``NameError``.
* ``_normalize_tasks`` returned from inside its loop, so a summary payload
  with several tasks silently kept only the first one.

Neither test needs Ollama, a model, a graph or any network access.
"""
from __future__ import annotations


def test_content_package_imports_without_a_model():
    import auto_ingest.content as content

    assert callable(content.run_build_summaries)


def test_all_tasks_in_a_payload_survive_normalization():
    from auto_ingest.content.build_summaries import _normalize_tasks

    tasks = [
        {"title": "First", "priority": "high"},
        {"title": "Second", "priority": "low"},
        {"title": "Third", "description": "third detail"},
    ]

    normalized = _normalize_tasks(tasks)

    assert [t["title"] for t in normalized] == ["First", "Second", "Third"]
    assert [t["priority"] for t in normalized] == ["high", "low", "medium"]


def test_later_tasks_keep_their_agent_and_plan_metadata():
    from auto_ingest.content.build_summaries import _normalize_tasks

    tasks = [
        {"title": "First"},
        {
            "title": "Second",
            "agent": {"name": "ops", "confidence": 0.8, "rationale": "fits"},
            "plan": [{"step": 1, "action": "check", "tool": "shell"}],
        },
    ]

    normalized = _normalize_tasks(tasks)

    assert len(normalized) == 2
    assert normalized[1]["agent"]["name"] == "ops"
    assert normalized[1]["plan"][0]["action"] == "check"


def test_invalid_entries_are_still_skipped_without_truncating_the_rest():
    from auto_ingest.content.build_summaries import _normalize_tasks

    tasks = [
        "not a task",
        {"title": ""},
        {"title": "Kept"},
        {"description": "also kept"},
    ]

    normalized = _normalize_tasks(tasks)

    assert [t["title"] for t in normalized] == ["Kept", ""]


def test_normalize_payload_keeps_the_whole_task_list():
    from auto_ingest.content.build_summaries import _normalize_payload

    payload = {
        "summary": "s",
        "tasks": [{"title": "A"}, {"title": "B"}, {"title": "C"}],
    }

    normalized = _normalize_payload(payload)

    assert [t["title"] for t in normalized["tasks"]] == ["A", "B", "C"]
    assert normalized["version"] == "1.0"
