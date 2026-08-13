"""Persisted execution-plan contracts for resumable pipelines.

A partially completed job must not silently resume under a different task graph.
The deterministic plan hash binds task names, semantic commands, timeouts, and
declared versions. Host-specific Python executable paths are normalized so the
same logical plan can fail over across machines without a false drift alarm.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Iterable


class PipelinePlanDrift(RuntimeError):
    pass


@dataclass(frozen=True)
class PlannedTask:
    name: str
    command: tuple[str, ...]
    timeout_sec: int
    version: str = "1"


def _normalize_command(command: Iterable[object]) -> list[str]:
    values = [str(v) for v in command]
    if not values:
        raise ValueError("pipeline task command must be non-empty")
    first = values[0]
    base = os.path.basename(first).lower()
    if re.fullmatch(r"python(?:3(?:\.\d+)?)?", base):
        values[0] = "<python>"
    return values


def plan_payload(tasks: Iterable[object]) -> list[dict]:
    payload: list[dict] = []
    for task in tasks:
        name = str(getattr(task, "name"))
        command = _normalize_command(getattr(task, "command"))
        timeout = int(getattr(task, "timeout_sec"))
        version = str(getattr(task, "version", "1"))
        payload.append(
            {
                "name": name,
                "command": command,
                "timeout_sec": timeout,
                "version": version,
            }
        )
    if len({row["name"] for row in payload}) != len(payload):
        raise ValueError("pipeline task names must be unique")
    return payload


def plan_hash(tasks: Iterable[object]) -> str:
    raw = json.dumps(plan_payload(tasks), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def bind_plan(driver, job_key: str, tasks: Iterable[object]) -> str:
    """Bind a deterministic plan to a job, rejecting drift after progress."""
    digest = plan_hash(tasks)
    with driver.session() as session:
        rec = session.run(
            """
            MATCH (j:IngestJob {key:$key})
            WITH j,
                 coalesce(j.completed_tasks,[]) AS completed,
                 j.plan_hash AS existing
            WITH j, completed, existing,
                 CASE
                   WHEN existing IS NULL THEN 'bind'
                   WHEN existing=$digest THEN 'same'
                   WHEN size(completed)=0 THEN 'rebind'
                   ELSE 'drift'
                 END AS decision
            FOREACH (_ IN CASE WHEN decision IN ['bind','rebind'] THEN [1] ELSE [] END |
                SET j.plan_hash=$digest,
                    j.plan_bound_at=timestamp(),
                    j.updated_at=timestamp()
            )
            RETURN decision, existing, completed
            """,
            key=job_key,
            digest=digest,
        ).single()
    if rec is None:
        raise KeyError(f"ingest job not found: {job_key}")
    if rec["decision"] == "drift":
        raise PipelinePlanDrift(
            f"pipeline plan changed after committed progress for {job_key}; "
            f"existing={rec['existing']} new={digest} completed={list(rec['completed'])}"
        )
    return digest
