from dataclasses import dataclass

from auto_ingest.pipeline_contract import plan_hash, plan_payload


@dataclass(frozen=True)
class _Task:
    name: str
    command: tuple[str, ...]
    timeout_sec: int


def test_python_interpreter_path_does_not_change_plan_identity():
    a = (_Task("stage", ("/usr/bin/python3", "-m", "thing"), 30),)
    b = (_Task("stage", ("/opt/venv/bin/python3.12", "-m", "thing"), 30),)
    assert plan_hash(a) == plan_hash(b)
    assert plan_payload(a)[0]["command"][0] == "<python>"


def test_semantic_command_change_changes_plan_identity():
    a = (_Task("stage", ("python3", "-m", "thing", "--limit", "10"), 30),)
    b = (_Task("stage", ("python3", "-m", "thing", "--limit", "11"), 30),)
    assert plan_hash(a) != plan_hash(b)


def test_timeout_change_changes_plan_identity():
    a = (_Task("stage", ("python3", "-m", "thing"), 30),)
    b = (_Task("stage", ("python3", "-m", "thing"), 31),)
    assert plan_hash(a) != plan_hash(b)
