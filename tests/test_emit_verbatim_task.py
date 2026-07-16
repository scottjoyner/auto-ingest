"""Tests for the auto-ingest verbatim-task producer (the ingest side of the
return-contract chain: auto-ingest -> delegate_task(opencode-cli) -> auto-assign)."""
import importlib.util
import os

import pytest

_EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "examples")


def _load(modname, path):
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def producer():
    return _load("emit_verbatim_task", os.path.join(_EXAMPLES, "emit_verbatim_task.py"))


def test_build_verbatim_task_shape(producer):
    task = producer.build_verbatim_task(
        title="fix a typo in the ingest tagger",
        instruction="Reply with exactly the word PONG",
        expected="PONG",
    )
    assert task["status"] == "READY"
    assert task["payload"]["return_format"] == "verbatim"
    assert task["payload"]["contract"] == "verbatim"
    assert "PONG" in task["description"]
    # consumers should know result.output is machine-usable, not prose
    assert task["payload"]["source_repo"] == "auto-ingest"


def test_verbatim_title_routes_to_delegated_tier(producer):
    # title must classify to tool-small so the auto-assist adapter routes it
    # through run_hermes_delegated() (when HERMES_DELEGATE_OPENCODE_TIERS=tool-small)
    assert producer.classifies_to_tool_small("fix a typo in the ingest tagger") is True
    assert producer.classifies_to_tool_small("architect a new ingestion pipeline") is False


def test_json_return_format_builds_object_contract(producer):
    task = producer.build_verbatim_task(
        title="fix a one-liner in the parser",
        instruction="classify this",
        expected={"label": "audio"},
        return_format="json",
    )
    assert task["payload"]["return_format"] == "json"
    assert "'result'" in task["description"]
    assert producer.classifies_to_tool_small(task["title"]) is True
