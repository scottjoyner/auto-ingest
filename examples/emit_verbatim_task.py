#!/usr/bin/env python3
"""
auto-ingest -> Hermes delegation -> auto-assign/auto-router: verbatim task example.

This is the *producer* side of the return-contract chain. auto-ingest is the
source of all swarm processing/tasks; when a task needs a programmatically
consumable answer (a token, a JSON object) rather than a prose summary, it
emits a task whose title is engineered to route through the auto-assist
adapter's ``run_hermes_delegated()`` path. The adapter then solves it via
``delegate_task(provider="opencode-cli", return_format="verbatim"|"json")`` and
writes the machine-usable value into ``task.result.output``, which
auto-assign / auto-router consume downstream.

End-to-end chain
----------------
    auto-ingest (this script)  --creates-->  AssistX Task (status=READY)
                                                    |
    auto-assist hermes_agent_adapter  --claims, delegates to opencode-cli,
                                       return_format=verbatim, completes with
                                       the token in result.output
                                                    |
    auto-assign / auto-router  --consume result.output (no parsing of prose)

Prerequisites
-------------
* The auto-assist adapter must run with delegation enabled for the tier this
  task classifies to (``tool-small`` by default here):

      HERMES_DELEGATE_OPENCODE_TIERS="tool-small" HERMES_DELEGATE_RETURN_FORMAT="verbatim" \\

* AssistX reachable at ASSISTX_API_URL (default http://localhost:8000).

Usage
-----
    # Dry-run: just print the task payload that would be emitted.
    python examples/emit_verbatim_task.py --dry-run \
        --title "fix a typo in the ingest tagger" \
        --instruction "Reply with exactly the word PONG" \
        --expected PONG

    # Real: create the task in AssistX (standalone READY task, notifies auto-assign).
    python examples/emit_verbatim_task.py \
        --title "fix a typo in the ingest tagger" \
        --instruction "Reply with exactly the word PONG" \
        --expected PONG

    # Then watch it complete and print the verbatim result:
    python examples/emit_verbatim_task.py --collect <task_id>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Optional

# Keywords the auto-assist adapter's classify_model_tier() maps to the
# "tool-small" tier (see _MODEL_TIER_KEYWORDS in hermes_agent_adapter.py).
# A title containing one of these routes the task through run_hermes_delegated()
# when HERMES_DELEGATE_OPENCODE_TIERS includes "tool-small".
_TOOL_SMALL_KEYWORDS = ["fix", "typo", "script", "tool", "rename", "small edit", "one-liner"]


def build_verbatim_task(
    title: str,
    instruction: str,
    expected: str,
    return_format: str = "verbatim",
    kind: str = "swarm_task",
) -> Dict[str, Any]:
    """Build an AssistX task payload that expects a machine-usable (verbatim) answer.

    The description embeds the verbatim contract so the delegated opencode-cli
    session returns exactly ``expected`` (or a JSON object when
    ``return_format="json"``). The title is expected to classify to the
    ``tool-small`` tier in the auto-assist adapter.
    """
    if return_format not in ("verbatim", "json", "summary"):
        return_format = "verbatim"

    if return_format == "verbatim":
        ask = (
            f"{instruction}\n\nReturn ONLY the exact value '{expected}' and nothing "
            f"else -- no preamble, no markdown fences, no commentary."
        )
    elif return_format == "json":
        ask = (
            f"{instruction}\n\nReturn a single JSON object (no markdown fences) that "
            f"includes the field 'result' set to: {json.dumps(expected)}"
        )
    else:
        ask = instruction

    return {
        "title": title,
        "kind": kind,
        "status": "READY",
        "description": ask,
        "payload": {
            "return_format": return_format,
            "expected": expected,
            "source_repo": "auto-ingest",
            "contract": "verbatim",  # signal to consumers: result.output is machine-usable
        },
    }


def classifies_to_tool_small(title: str) -> bool:
    """Best-effort check that the title routes to the delegated tier."""
    low = title.lower()
    return any(kw in low for kw in _TOOL_SMALL_KEYWORDS)


# ---------------------------------------------------------------------------
# AssistX transport (standalone create + collect)
# ---------------------------------------------------------------------------
def _assistx_base() -> str:
    return os.getenv("ASSISTX_API_URL", "http://localhost:8000").rstrip("/")


def _assistx_auth() -> tuple:
    return (os.getenv("ASSISTX_AUTH_USER", "admin"), os.getenv("ASSISTX_AUTH_PASS", "change-me"))


def emit_via_api(task: Dict[str, Any], transcription_id: Optional[str] = None) -> Dict[str, Any]:
    """Create the task in AssistX.

    Uses the transcription-task endpoint when a ``transcription_id`` is given
    (auto-ingest's natural path), otherwise falls back to the standalone
    Neo4jClient.create_task_with_context() if assistx is importable.
    """
    import requests  # local import so --dry-run needs no network deps

    if transcription_id:
        url = f"{_assistx_base()}/api/transcriptions/{transcription_id}/task"
        body = {"title": task["title"], "kind": task["kind"], "payload": task["payload"]}
        resp = requests.post(url, json=body, auth=_assistx_auth(), timeout=30)
        resp.raise_for_status()
        return resp.json()

    # Standalone READY task that also notifies auto-assign.
    try:
        from assistx.neo4j_client import Neo4jClient  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on assistx on path
        raise RuntimeError(
            "Standalone emit requires assistx on PYTHONPATH (or pass --transcription-id). "
            f"Import failed: {exc}"
        ) from exc
    neo = Neo4jClient()
    try:
        res = neo.create_task_with_context(
            title=task["title"],
            task_type=task["kind"],
            status="READY",
            kind=task["kind"],
            payload=task["payload"],
        )
    finally:
        neo.close()
    return res


def collect_result(task_id: str, poll: int = 5, timeout: int = 600) -> Dict[str, Any]:
    """Poll AssistX until the task is DONE/FAILED and return result.output."""
    import requests

    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = requests.get(f"{_assistx_base()}/api/tasks/{task_id}", auth=_assistx_auth(), timeout=30)
        resp.raise_for_status()
        doc = resp.json()
        status = doc.get("status")
        if status in ("DONE", "FAILED"):
            return {
                "status": status,
                "output": (doc.get("result") or {}).get("output"),
                "model": (doc.get("result") or {}).get("model"),
            }
        time.sleep(poll)
    return {"status": "TIMEOUT", "output": None}


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--title", default="fix a typo in the ingest tagger")
    p.add_argument("--instruction", default="Reply with exactly the word PONG")
    p.add_argument("--expected", default="PONG")
    p.add_argument("--return-format", default="verbatim", choices=["verbatim", "json", "summary"])
    p.add_argument("--transcription-id", default=None, help="Attach to an existing Transcription")
    p.add_argument("--dry-run", action="store_true", help="Only print the task payload")
    p.add_argument("--collect", metavar="TASK_ID", help="Poll a task id and print its result.output")
    args = p.parse_args(argv)

    if args.collect:
        out = collect_result(args.collect)
        print(json.dumps(out, indent=2))
        return 0

    task = build_verbatim_task(
        title=args.title,
        instruction=args.instruction,
        expected=args.expected,
        return_format=args.return_format,
    )

    if not classifies_to_tool_small(task["title"]):
        print(
            "WARNING: title does not contain a tool-small keyword; the auto-assist "
            "adapter may not route it through run_hermes_delegated(). Add one of: "
            f"{_TOOL_SMALL_KEYWORDS}",
            file=sys.stderr,
        )

    if args.dry_run:
        print(json.dumps(task, indent=2))
        return 0

    res = emit_via_api(task, transcription_id=args.transcription_id)
    task_id = res.get("task_id")
    print(f"Emitted task {task_id}. The auto-assist adapter will delegate it to opencode-cli")
    print(f"and complete it with the verbatim value in result.output.")
    print(f"Collect with: python {os.path.basename(__file__)} --collect {task_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
