"""Regression tests for the summary/task sidecar normalizer.

Hermetic and deterministic: these call the pure ``_normalize_tasks`` helper
directly. No Ollama server, no model load, no network, no filesystem writes.

``_normalize_tasks`` used to ``return out`` after appending the *first* valid
task, so a model that emitted several actions silently produced a tasks sidecar
containing only one. It also had a second, dead normalisation loop that would
have duplicated every task had the early return been simply deleted.
"""

from auto_ingest.content.build_summaries import _normalize_tasks


def _task(title, **overrides):
    task = {"title": title, "description": f"description for {title}"}
    task.update(overrides)
    return task


def test_normalize_tasks_keeps_every_valid_task():
    """No silent truncation: all emitted actions survive normalisation."""
    tasks = [_task(f"task-{i}") for i in range(4)]
    out = _normalize_tasks(tasks)
    assert [t["title"] for t in out] == [f"task-{i}" for i in range(4)]


def test_normalize_tasks_skips_invalid_but_keeps_all_valid():
    """Invalid entries are dropped without truncating later valid ones."""
    tasks = [
        {"title": "", "description": ""},   # dropped: no title/description
        _task("valid-1"),
        "not-a-dict",                       # dropped: not an object
        _task("valid-2"),
        None,                               # dropped: not an object
        _task("valid-3"),
    ]
    out = _normalize_tasks(tasks)
    assert [t["title"] for t in out] == ["valid-1", "valid-2", "valid-3"]


def test_normalize_tasks_preserves_full_shape_without_duplicates():
    """Full agent/plan schema survives, and no task is emitted twice."""
    out = _normalize_tasks([
        _task(
            "ship-it",
            labels=["DevOps", "DevOps", "Urgent"],
            priority="HIGH",
            agent={"name": "DevOpsAgent", "confidence": "0.9", "rationale": "r"},
            plan=[{"action": "a", "tool": "t", "operation": "o"}],
        ),
        _task("follow-up"),
    ])
    assert [t["title"] for t in out] == ["ship-it", "follow-up"]
    first = out[0]
    assert first["priority"] == "high"
    assert first["labels"] == ["DevOps", "Urgent"]
    assert first["agent"]["name"] == "DevOpsAgent"
    assert first["agent"]["confidence"] == 0.9
    assert first["plan"][0]["step"] == 1


def test_normalize_tasks_non_list_is_empty():
    assert _normalize_tasks(None) == []
    assert _normalize_tasks({"title": "one"}) == []
