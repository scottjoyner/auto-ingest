"""Experimental synchronous Hermes arXiv child supervisor; NOT deployed or authorized.

Library only: intentionally NO __main__/CLI. Only an independently approved
scheduler integration may call supervise() after actual owner/auth/admission
review. Unit tests spawn harmless disposable synthetic scripts, never models.
The checksum arguments are caller-supplied pins, NOT signed authorizations.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
from pathlib import Path

APPROVED_MODEL = "text-embedding-nomic-embed-text-v1.5"
EXPECTED_DIMENSIONS = 768
SHA = re.compile(r"[0-9a-f]{64}\Z")


def _pinned(file: Path, pinned_sha: str, *, max_size: int) -> bool:
    if (
        not isinstance(pinned_sha, str)
        or not SHA.fullmatch(pinned_sha)
        or file.is_symlink()
        or not file.is_file()
        or not 0 < file.stat().st_size <= max_size
    ):
        return False
    with file.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return digest == pinned_sha


def _preflight(
    checkout: Path, python_executable: Path, source_sha: str,
    registry_sha: str,
) -> bool:
    if checkout.is_symlink() or not checkout.is_dir():
        return False
    script = checkout / "scripts" / "arxiv_kg_bridge.py"
    registry_file = checkout / "embedding_endpoints.json"
    client_file = checkout / "scripts" / "embedding_client.py"
    if not (
        _pinned(script, source_sha, max_size=262144)
        and _pinned(registry_file, registry_sha, max_size=262144)
        and client_file.is_file()
        and not client_file.is_symlink()
        and python_executable.is_file()
        and not python_executable.is_symlink()
        and os.access(python_executable, os.X_OK)
    ):
        return False
    try:
        data = json.loads(registry_file.read_text())
    except (OSError, ValueError, UnicodeError):
        return False
    return (
        isinstance(data, dict)
        and data.get("model") == APPROVED_MODEL
        and type(data.get("embedding_dimensions")) is int
        and data["embedding_dimensions"] == EXPECTED_DIMENSIONS
    )


def supervise(
    *,
    checkout: Path,
    python_executable: Path,
    reviewed_source_sha256: str,
    reviewed_registry_sha256: str,
    lock_file: Path,
    timeout_seconds: float,
) -> dict:
    """Synchronously run one pinned child and REAP it; reject overlapping jobs.

    Returns a SANITIZED local outcome, not a signed owner/worker receipt.
    Does not print script content, argv, model endpoint, prompts or output.
    It is NOT callable from an installed production entrypoint yet.
    """
    outcome = {
        "schema": "hermes-arxiv-supervisor-experiment-v1",
        "status": "DENIED_PREFLIGHT",
        "child_exit_code": None,
        "child_waited": False,
        "source_and_registry_checksum_match": False,
        "operator_approval_verified": False,
        "per_user_embedding_auth_verified": False,
        "source_signed_receipt": False,
        "can_approve": False,
        "can_dispatch": False,
    }
    if (
        type(timeout_seconds) not in (float, int)
        or not 0.1 <= timeout_seconds <= 3600
    ):
        outcome["status"] = "DENIED_TIMEOUT_BUDGET"
        return outcome
    if not _preflight(
        checkout, python_executable,
        reviewed_source_sha256, reviewed_registry_sha256,
    ):
        return outcome
    outcome["source_and_registry_checksum_match"] = True
    if (
        lock_file.is_symlink() or lock_file.parent.is_symlink()
        or not lock_file.parent.is_dir()
    ):
        outcome["status"] = "DENIED_LOCK_LOCATION"
        return outcome

    # Atomic, nonblocking single-host admission against overlapping fires.
    # This is NOT a fleet-wide admission or split-brain protection.
    with lock_file.open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            outcome["status"] = "DENIED_OVERLAP"
            return outcome
        proc = None
        try:
            # start_new_session makes timeouts kill the entire immediate
            # process group, not an unrelated job, then waits for reaping.
            proc = subprocess.Popen(
                [str(python_executable), str(checkout / "scripts" / "arxiv_kg_bridge.py")],
                cwd=str(checkout),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
                start_new_session=True,
            )
            try:
                code = proc.wait(timeout=timeout_seconds)
                outcome["child_waited"] = True
                outcome["child_exit_code"] = code
                outcome["status"] = "CHILD_EXIT_ZERO" if code == 0 else "CHILD_EXIT_NONZERO"
            except subprocess.TimeoutExpired:
                outcome["status"] = "CHILD_TIMEOUT_REAPED"
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=2)
                outcome["child_waited"] = True
        except (OSError, subprocess.SubprocessError):
            outcome["status"] = "CHILD_START_OR_WAIT_FAILED"
            if proc is not None and proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=2)
                    outcome["child_waited"] = True
                except (OSError, subprocess.SubprocessError):
                    outcome["status"] = "CHILD_REAP_UNCONFIRMED"
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return outcome
