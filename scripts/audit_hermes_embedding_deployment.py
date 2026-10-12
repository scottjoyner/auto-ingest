#!/usr/bin/env python3
"""Read-only Hermes/Auto-Ingest rollout inventory; always denies deployment.

Job prompts can contain sensitive context. This tool NEVER echoes prompts,
command lines, endpoint addresses, filenames outside an approved basename
allowlist, schedules, tokens, or file contents. It does not use network, cron,
Signal, model inference, Neo4j, or subprocess commands. A scheduled-agent
prompt is not a cryptographically bound execution path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

MAX_JOBS_BYTES = 256 * 1024
MAX_SCRIPT_BYTES = 256 * 1024
EXPECTED = {
    "arxiv_kg_bridge.py",
    "backfill_paper_768.py",
    "signal_kg_bridge.py",
    "enrich_bridge_semantic.py",
}
KNOWN_JOBS = {
    "kg-arxiv-ingest",
    "kg-paper768-backfill",
    "kg-signal-bridge",
    "neo4j enrichment semantic re-embed (weekly)",
}


def _read_bounded(file: Path, *, limit: int) -> bytes:
    if file.is_symlink() or not file.is_file():
        raise ValueError("untrusted symlink or missing source")
    if not 0 < file.stat().st_size <= limit:
        raise ValueError("source outside byte budget")
    with file.open("rb") as handle:
        body = handle.read(limit + 1)
    if len(body) > limit:
        raise ValueError("source exceeds byte budget")
    return body


def inspect_jobs(obj: Any) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise ValueError("scheduler must be JSON object")
    rows = obj.get("jobs")
    if isinstance(rows, dict):
        rows = list(rows.values())
    if not isinstance(rows, list) or len(rows) > 512:
        raise ValueError("malformed or excessive scheduler entries")
    known = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = row.get("name")
        if not isinstance(name, str) or name.lower() not in KNOWN_JOBS:
            continue
        enabled = row.get("enabled")
        if type(enabled) is not bool:
            raise ValueError("scheduled-agent job enabled state invalid")
        # We do NOT inspect prompt text to infer actual command authority.
        known.append({
            "known_job_name": name,
            "enabled": enabled,
            "execution_path_attested": False,
            "source_digest_attested": False,
            "process_or_runtime_owner_attested": False,
        })
    return {
        "total_scheduler_records": len(rows),
        "known_embedding_job_count": len(known),
        "known_jobs": known,
        "enabled_execution_paths_unbound": sum(
            1 for item in known if item["enabled"]
        ),
        "scheduler_job_prompts_disclosed": False,
    }


def inspect_roots(roots: list[Path]) -> dict[str, Any]:
    if not isinstance(roots, list) or not 1 <= len(roots) <= 5:
        raise ValueError("supply bounded exact checkout roots")
    copies = []
    for n, root in enumerate(roots):
        if root.is_symlink() or not root.is_dir():
            raise ValueError("untrusted checkout root")
        records = []
        for name in sorted(EXPECTED):
            file = root / "scripts" / name
            if file.is_symlink() or not file.is_file():
                records.append({
                    "script": name, "exists": False, "central_client_import": False,
                    "legacy_endpoint_present": False,
                })
                continue
            data = _read_bounded(file, limit=MAX_SCRIPT_BYTES)
            records.append({
                "script": name,
                "exists": True,
                "sha256_prefix": hashlib.sha256(data).hexdigest()[:16],
                "central_client_import": (
                    b"from scripts.embedding_client import " in data
                ),
                "legacy_endpoint_present": (
                    b"LM_STUDIO_URL =" in data or b"EMBED_URL =" in data
                ),
            })
        shared = root / "scripts" / "embedding_client.py"
        registry = root / "embedding_endpoints.json"
        client_present = shared.is_file() and not shared.is_symlink()
        dimension_pin = False
        try:
            reg = json.loads(_read_bounded(registry, limit=MAX_SCRIPT_BYTES))
            dimension_pin = (
                isinstance(reg, dict)
                and reg.get("embedding_dimensions") == 768
                and reg.get("model") == "text-embedding-nomic-embed-text-v1.5"
            )
        except (OSError, ValueError, UnicodeError):
            pass
        copies.append({
            "checkout_index": n,
            "known_scripts": records,
            "shared_client_present": client_present,
            "nomic_768_registry_pinned": dimension_pin,
        })
    fingerprints = {
        entry["script"]: set() for entry in copies[0]["known_scripts"]
    }
    for copy in copies:
        for entry in copy["known_scripts"]:
            if entry["exists"]:
                fingerprints[entry["script"]].add(entry["sha256_prefix"])
    return {
        "checkout_count": len(copies),
        "checkouts": copies,
        "script_copies_differ": any(len(values) > 1 for values in fingerprints.values()),
        "all_copies_migrated": all(
            x["shared_client_present"]
            and x["nomic_768_registry_pinned"]
            and all(
                entry["exists"] and entry["central_client_import"]
                and not entry["legacy_endpoint_present"]
                for entry in x["known_scripts"]
            )
            for x in copies
        ),
    }


def collect(jobs: Path, roots: list[Path]) -> dict[str, Any]:
    scheduler = inspect_jobs(json.loads(_read_bounded(jobs, limit=MAX_JOBS_BYTES)))
    source = inspect_roots(roots)
    return {
        "schema": "hermes-embedding-deployment-review-v1",
        "source_kind": "unsigned_read_only_local_snapshot",
        "scheduler": scheduler,
        "source_copies": source,
        "status": "BLOCKED_UNBOUND_SCHEDULED_EXECUTION",
        "authenticated_embedding_owner_proven": False,
        "source_provenance_verified": False,
        "active_execution_path_verified": False,
        "operator_approval_present": False,
        "can_change_cron": False,
        "can_deploy": False,
        "can_dispatch": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hermes-jobs", required=True, type=Path)
    parser.add_argument("--checkout-root", required=True, type=Path, action="append")
    args = parser.parse_args()
    try:
        report = collect(args.hermes_jobs, args.checkout_root)
    except (OSError, ValueError, UnicodeError, TypeError) as exc:
        print("scheduler/source audit refused: " + type(exc).__name__, file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True))
    return 1  # NO-GO even when files align: agent job execution is unbound.


if __name__ == "__main__":
    raise SystemExit(main())
