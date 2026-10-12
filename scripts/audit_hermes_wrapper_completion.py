#!/usr/bin/env python3
"""Read-only Hermes no-agent script wrapper completion preflight.

Never executes wrappers, model inference, scheduler actions, or graph queries.
Avoids emitting private script paths, job prompts, argv, command contents,
log content, host addresses, and tokens. All results remain deployment NO-GO.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

MAX_BYTES = 65536
MAX_JOBS_BYTES = 262144
KNOWN = {
    "kg-arxiv-ingest": ("kg-arxiv-ingest.py", "arxiv_kg_bridge.py"),
    "kg-paper768-backfill": ("kg-paper768-backfill.py", "backfill_paper_768.py"),
    "kg-signal-bridge": ("kg-signal-bridge.py", "signal_kg_bridge.py"),
}
WEEKLY = "neo4j enrichment semantic re-embed (weekly)"


def read_bounded(path: Path, budget: int) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError("untrusted or unavailable audit file")
    if not 0 < path.stat().st_size <= budget:
        raise ValueError("file size outside audit budget")
    with path.open("rb") as fh:
        data = fh.read(budget + 1)
    if len(data) > budget:
        raise ValueError("file grew past budget")
    return data


def inspect(jobs: dict[str, Any], wrappers: Path) -> dict[str, Any]:
    if not isinstance(jobs, dict):
        raise ValueError("Hermes jobs payload must be object")
    if wrappers.is_symlink() or not wrappers.is_dir():
        raise ValueError("untrusted wrapper root")
    rows = jobs.get("jobs", [])
    if isinstance(rows, dict):
        rows = list(rows.values())
    if not isinstance(rows, list) or len(rows) > 512:
        raise ValueError("unbounded or invalid jobs")
    found: list[dict[str, Any]] = []
    relevant = set(KNOWN) | {WEEKLY}
    for job in rows:
        if not isinstance(job, dict):
            continue
        name = job.get("name")
        if not isinstance(name, str) or name.lower() not in relevant:
            continue
        if type(job.get("enabled")) is not bool:
            raise ValueError("enabled job has invalid bool state")
        normalized = name.lower()
        observation = {
            "job_name": name,
            "enabled": job["enabled"],
            "last_status_observed": (
                job.get("last_status") if job.get("last_status") in ("ok", "error")
                else "unknown"
            ),
            "job_mode": "unverified",
            "configured_script_basename": None,
            "wrapper_file_observed": False,
            "wrapper_sha256_prefix": None,
            "expected_caller_referenced": False,
            "shared_ssd_referenced": False,
            "launch_kind": "unverified",
            "child_wait_or_completion_check_detected": False,
            "child_completion_attested": False,
            "owner_or_approved_digest_attested": False,
            "can_promote": False,
            "can_dispatch": False,
        }
        expected = KNOWN.get(normalized)
        if (
            expected
            and job.get("no_agent") is True
            and job.get("script") == expected[0]
        ):
            observation["job_mode"] = "configured_no_agent_script"
            observation["configured_script_basename"] = expected[0]
            wrapper = wrappers / expected[0]
            try:
                raw = read_bounded(wrapper, MAX_BYTES)
                parsed = ast.parse(raw.decode("utf-8"))
                observation["wrapper_file_observed"] = True
                observation["wrapper_sha256_prefix"] = hashlib.sha256(
                    raw
                ).hexdigest()[:16]
                observation["expected_caller_referenced"] = (
                    expected[1].encode() in raw
                )
                observation["shared_ssd_referenced"] = b"SSD_4TB" in raw
                def calls_named(*allowed: str) -> list[ast.Call]:
                    return [
                        x for x in ast.walk(parsed)
                        if isinstance(x, ast.Call)
                        and isinstance(x.func, ast.Attribute)
                        and x.func.attr in allowed
                    ]
                detached = calls_named("Popen")
                blocking = calls_named("call", "run", "check_call")
                waits = calls_named("wait", "communicate")
                has_detached_without_wait = bool(detached) and not waits
                observation["launch_kind"] = (
                    "detached_popen_no_completion_wait"
                    if has_detached_without_wait
                    else "blocking_child_call_candidate" if blocking
                    else "other_or_unverified_wrapper"
                )
                observation["child_wait_or_completion_check_detected"] = (
                    bool(blocking) or bool(waits)
                )
            except (OSError, ValueError, UnicodeError, SyntaxError):
                observation["launch_kind"] = "missing_or_invalid_wrapper"
        elif normalized == WEEKLY and job.get("no_agent") is not True:
            observation["job_mode"] = "agent_driven_no_direct_script"
        found.append(observation)
    return {
        "schema": "hermes-no-agent-wrapper-completion-audit-v1",
        "source_kind": "unsigned_local_source_metadata_only",
        "total_known_jobs": len(found),
        "enabled_known_jobs": sum(x["enabled"] for x in found),
        "configured_no_agent_script_jobs": sum(
            x["job_mode"] == "configured_no_agent_script" for x in found
        ),
        "detached_without_completion_wait": sum(
            x["launch_kind"] == "detached_popen_no_completion_wait" for x in found
        ),
        "agent_driven_jobs": sum(
            x["job_mode"] == "agent_driven_no_direct_script" for x in found
        ),
        "jobs": found,
        "private_prompt_or_script_content_included": False,
        "runtime_child_execution_verified": False,
        "source_digest_independently_approved": False,
        "operator_change_approved": False,
        "can_change_scheduler": False,
        "can_deploy": False,
        "can_dispatch": False,
        "status": "BLOCKED_UNVERIFIED_HERMES_CHILD_COMPLETION",
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hermes-jobs", type=Path, required=True)
    p.add_argument("--wrapper-root", type=Path, required=True)
    args = p.parse_args()
    try:
        jobs = json.loads(read_bounded(args.hermes_jobs, MAX_JOBS_BYTES))
        out = inspect(jobs, args.wrapper_root)
    except (OSError, ValueError, UnicodeError, TypeError) as exc:
        print("Hermes wrapper audit refused: " + type(exc).__name__, file=sys.stderr)
        return 2
    print(json.dumps(out, sort_keys=True))
    return 1  # Configured wrapper is not verified child execution or an admission.


if __name__ == "__main__":
    raise SystemExit(main())
