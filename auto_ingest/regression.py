"""Replayable regression fixtures for production bug harvesting.

Fixtures are JSON documents containing named cases against an allowlisted adapter
registry. They never execute arbitrary shell commands or import arbitrary callables.
This makes it safe to turn a bug observed on another machine into a permanent CI
fixture with the exact input and expected behavior.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from auto_ingest.artifacts import artifact_relative_path, build_identity, sha256_bytes
from auto_ingest.file_queue import QueueJob
from auto_ingest.pipeline_contract import plan_hash
from auto_ingest.orchestration import Task

FIXTURE_VERSION = 1


class RegressionFixtureError(ValueError):
    pass


@dataclass(frozen=True)
class CaseResult:
    name: str
    target: str
    passed: bool
    actual: Any = None
    error: str | None = None


def _artifact_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    source = str(payload.get("source_text", "")).encode("utf-8")
    identity = build_identity(
        source_hash=sha256_bytes(source),
        stage=str(payload["stage"]),
        stage_version=str(payload["stage_version"]),
        config=dict(payload.get("config") or {}),
        model=dict(payload.get("model") or {}) or None,
    )
    return {
        "artifact_id": identity.artifact_id,
        "relative_path": str(artifact_relative_path(identity, str(payload.get("suffix", "")))),
    }


def _queue_job(payload: Mapping[str, Any]) -> dict[str, Any]:
    return QueueJob.from_dict(dict(payload)).to_dict()


def _pipeline_plan(payload: Mapping[str, Any]) -> dict[str, Any]:
    rows = payload.get("tasks")
    if not isinstance(rows, list):
        raise RegressionFixtureError("pipeline.plan input.tasks must be a list")
    tasks = tuple(
        Task(
            str(row["name"]),
            tuple(str(value) for value in row["command"]),
            int(row["timeout_sec"]),
        )
        for row in rows
    )
    return {"plan_hash": plan_hash(tasks)}


ADAPTERS: dict[str, Callable[[Mapping[str, Any]], Any]] = {
    "artifact.identity": _artifact_identity,
    "queue.job": _queue_job,
    "pipeline.plan": _pipeline_plan,
}


def _matches(actual: Any, expected: Mapping[str, Any]) -> bool:
    if "equals" in expected:
        return actual == expected["equals"]
    if "contains" in expected:
        wanted = expected["contains"]
        if not isinstance(actual, Mapping) or not isinstance(wanted, Mapping):
            return False
        return all(actual.get(key) == value for key, value in wanted.items())
    raise RegressionFixtureError("expect must define either 'equals' or 'contains'")


def run_case(case: Mapping[str, Any]) -> CaseResult:
    name = str(case.get("name") or "unnamed")
    target = str(case.get("target") or "")
    adapter = ADAPTERS.get(target)
    if adapter is None:
        return CaseResult(name, target, False, error=f"unknown target: {target!r}")
    payload = case.get("input", {})
    if not isinstance(payload, Mapping):
        return CaseResult(name, target, False, error="input must be an object")
    expected_error = case.get("raises")
    try:
        actual = adapter(payload)
    except Exception as exc:
        if expected_error and type(exc).__name__ == str(expected_error):
            return CaseResult(name, target, True, error=type(exc).__name__)
        return CaseResult(name, target, False, error=f"{type(exc).__name__}: {exc}")
    if expected_error:
        return CaseResult(name, target, False, actual=actual, error=f"expected {expected_error}")
    expect = case.get("expect")
    if not isinstance(expect, Mapping):
        return CaseResult(name, target, False, actual=actual, error="expect must be an object")
    try:
        passed = _matches(actual, expect)
    except Exception as exc:
        return CaseResult(name, target, False, actual=actual, error=str(exc))
    return CaseResult(name, target, passed, actual=actual, error=None if passed else "expectation mismatch")


def run_fixture(path: str | Path) -> list[CaseResult]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("version") != FIXTURE_VERSION:
        raise RegressionFixtureError(f"fixture version must be {FIXTURE_VERSION}")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise RegressionFixtureError("fixture cases must be a non-empty list")
    return [run_case(case) for case in cases]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m auto_ingest.regression")
    parser.add_argument("fixtures", nargs="+", help="Regression fixture JSON files")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    all_results: list[CaseResult] = []
    for fixture in args.fixtures:
        all_results.extend(run_fixture(fixture))
    if args.json:
        print(json.dumps([result.__dict__ for result in all_results], indent=2, default=str))
    else:
        for result in all_results:
            status = "PASS" if result.passed else "FAIL"
            detail = f" ({result.error})" if result.error else ""
            print(f"{status} {result.name} [{result.target}]{detail}")
    return 0 if all(result.passed for result in all_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
