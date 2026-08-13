"""Crash-safe filesystem commit primitives for generated artifacts.

Writers stage bytes into a sibling temporary file, fsync, validate the digest,
and atomically replace the final path. The final artifact is either absent or
complete; partially written final files are never exposed.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from auto_ingest.artifacts import sha256_file


@dataclass(frozen=True)
class CommitResult:
    path: str
    sha256: str
    bytes_written: int
    reused: bool


class ArtifactCommitError(RuntimeError):
    pass


def atomic_commit_bytes(
    final_path: str | Path,
    data: bytes,
    *,
    expected_sha256: str | None = None,
    before_replace: Callable[[Path], None] | None = None,
) -> CommitResult:
    """Atomically commit bytes to ``final_path`` and validate their digest.

    ``before_replace`` exists primarily for fault-injection testing and for
    callers that need a final pre-commit validation hook. If it raises, the
    temporary file is cleaned and the final path remains untouched.
    """
    final = Path(final_path)
    final.parent.mkdir(parents=True, exist_ok=True)

    if final.exists():
        digest = sha256_file(final)
        if expected_sha256 is None or digest == expected_sha256:
            return CommitResult(str(final), digest, final.stat().st_size, True)
        raise ArtifactCommitError(
            f"existing artifact digest mismatch at {final}: {digest} != {expected_sha256}"
        )

    fd, temp_name = tempfile.mkstemp(prefix=f".{final.name}.", suffix=".tmp", dir=final.parent)
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        digest = sha256_file(temp)
        if expected_sha256 is not None and digest != expected_sha256:
            raise ArtifactCommitError(
                f"staged artifact digest mismatch: {digest} != {expected_sha256}"
            )
        if before_replace is not None:
            before_replace(temp)
        os.replace(temp, final)
        # fsync the parent directory so the rename itself is durable.
        dir_fd = os.open(final.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
        return CommitResult(str(final), digest, len(data), False)
    except Exception:
        try:
            temp.unlink(missing_ok=True)
        finally:
            raise


def verify_artifact(path: str | Path, expected_sha256: str) -> bool:
    p = Path(path)
    return p.is_file() and sha256_file(p) == expected_sha256
