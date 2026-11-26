"""Local output receipts for reconciliation, not execution authorization.

Receipts bind a run ID to exact local output bytes; they do not authenticate
the writer, attest remote side effects, establish worker liveness, or grant
replay. Verification is offline and consumes caller-supplied bytes only.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from pathlib import Path

_SCHEMA = "executor-local-output-v1"
_MAX_RECEIPT_BYTES = 16384


def _validate_identity(task_id, run_id):
    if not isinstance(task_id, str) or not task_id.strip() or len(task_id) > 256:
        raise ValueError("invalid task ID")
    if not isinstance(run_id, str) or str(uuid.UUID(run_id)) != run_id:
        raise ValueError("invalid canonical run UUID")


def _payload(task_id, run_id, output):
    _validate_identity(task_id, run_id)
    if not isinstance(output, bytes):
        raise TypeError("output must be bytes")
    return {
        "schema": _SCHEMA, "task_id": task_id, "run_id": run_id,
        "bytes": len(output), "sha256": hashlib.sha256(output).hexdigest(),
        "scope": "LOCAL_OUTPUT_ONLY",
    }


def publish_local_output_receipt(directory, *, task_id, run_id, output):
    """Publish a complete, create-only receipt beside the existing output.

    The caller must flush the output before publication. An existing run
    receipt is never replaced. A publication error is not permission to retry
    the operation. File and directory fsync are attempted before returning.
    """
    data = _payload(task_id, run_id, output)
    return _publish_create_only(Path(directory) / f"run-{run_id}.json",
                                (json.dumps(data, sort_keys=True) + "\n").encode("utf-8"))


def publish_run_output(directory, *, task_id, run_id, output):
    """Preserve exact run bytes without replacing any existing run output."""
    _payload(task_id, run_id, output)
    return _publish_create_only(Path(directory) / f"run-{run_id}.txt", output)


def _publish_create_only(target, output):
    parent = target.parent
    fd, temporary = tempfile.mkstemp(prefix=".run-receipt-", dir=parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(output)
            handle.flush()
            os.fsync(handle.fileno())
        # Same-directory hard link is atomic and refuses an existing target.
        os.link(temporary, target)
        dirfd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dirfd)
        finally:
            os.close(dirfd)
    finally:
        os.unlink(temporary)
    return target


def inspect_local_output(receipt, output, *, task_id, run_id):
    """Compare exact bytes against a supplied receipt; never authorize replay."""
    expected = _payload(task_id, run_id, output)
    classification = "LOCAL_OUTPUT_UNVERIFIED"
    if isinstance(receipt, bytes) and len(receipt) <= _MAX_RECEIPT_BYTES:
        try:
            # Duplicate JSON keys would make the receipt ambiguous.
            def unique(pairs):
                values = {}
                for key, value in pairs:
                    if key in values:
                        raise ValueError("duplicate receipt key")
                    values[key] = value
                return values

            actual = json.loads(receipt, object_pairs_hook=unique)
            marker = f"\nRUN: {run_id}\n".encode("utf-8")
            if (isinstance(actual, dict) and type(actual.get("bytes")) is int
                    and actual == expected and output.endswith(marker)):
                classification = "LOCAL_OUTPUT_BYTES_MATCH"
        except (ValueError, UnicodeError, RecursionError):
            pass
    return {"classification": classification, "replay_allowed": False}
