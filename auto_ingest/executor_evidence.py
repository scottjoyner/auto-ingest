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


def _http_identity(task_id, run_id, url):
    _validate_identity(task_id, run_id)
    if not isinstance(url, str) or not url:
        raise ValueError("invalid HTTP target")
    return {"schema": "executor-http-observation-v1", "task_id": task_id,
            "run_id": run_id, "method": "GET",
            "target_sha256": hashlib.sha256(url.encode("utf-8")).hexdigest(),
            "scope": "LOCAL_OBSERVATION_ONLY"}


def _http_bytes(payload):
    return (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")


def _http_path(directory, identity, kind):
    return Path(directory) / (
        f'run-{identity["run_id"]}-http-{identity["target_sha256"]}-{kind}.json')


def publish_http_intent(directory, *, task_id, run_id, url):
    """Persist intent before dispatch; existing intent is never replay authority."""
    identity = _http_identity(task_id, run_id, url)
    return _publish_create_only(_http_path(directory, identity, "intent"),
                                _http_bytes(dict(identity, state="INTENT")))


def publish_http_outcome(directory, *, task_id, run_id, url, status):
    """Record a locally observed response or uncertainty, not remote completion."""
    identity = _http_identity(task_id, run_id, url)
    intent = _http_bytes(dict(identity, state="INTENT"))
    # Require the matching create-only intent, with a bounded read.
    with _http_path(directory, identity, "intent").open("rb") as handle:
        if handle.read(len(intent) + 1) != intent:
            raise ValueError("HTTP intent mismatch")
    if status is not None and (type(status) is not int or not 100 <= status <= 599):
        raise ValueError("invalid HTTP status")
    payload = dict(identity, state="RESPONSE_OBSERVED" if status is not None else "UNCERTAIN",
                   status=status, intent_sha256=hashlib.sha256(intent).hexdigest())
    return _publish_create_only(_http_path(directory, identity, "outcome"),
                                _http_bytes(payload))


def inspect_http_observation(intent, outcome, *, task_id, run_id, url):
    """Classify caller-supplied bytes; no I/O, remote proof or replay authority.

    An absent intent is not evidence of non-dispatch. Exact canonical records
    establish only local observation consistency, not writer authentication.
    """
    identity = _http_identity(task_id, run_id, url)
    expected_intent = _http_bytes(dict(identity, state="INTENT"))
    result = {"classification": "HTTP_EVIDENCE_UNVERIFIED", "status": None,
              "remote_completion_verified": False, "replay_allowed": False}
    if intent is None and outcome is None:
        result["classification"] = "HTTP_NO_EVIDENCE"
        return result
    if not isinstance(intent, bytes) or intent != expected_intent:
        return result
    if outcome is None:
        result["classification"] = "HTTP_INTENT_ONLY_UNCERTAIN"
        return result
    if not isinstance(outcome, bytes) or len(outcome) > _MAX_RECEIPT_BYTES:
        return result
    try:
        actual = json.loads(outcome)
        if not isinstance(actual, dict):
            return result
        status = actual.get("status")
        if status is not None and (type(status) is not int or not 100 <= status <= 599):
            return result
        expected = dict(identity, state="RESPONSE_OBSERVED" if status is not None else "UNCERTAIN",
                        status=status, intent_sha256=hashlib.sha256(intent).hexdigest())
        # Byte equality also rejects duplicate keys, extra fields and noncanonical records.
        if outcome != _http_bytes(expected):
            return result
        result["classification"] = ("HTTP_LOCAL_RESPONSE_OBSERVED" if status is not None
                                    else "HTTP_OUTCOME_UNCERTAIN")
        result["status"] = status
    except (ValueError, UnicodeError, RecursionError):
        pass
    return result
