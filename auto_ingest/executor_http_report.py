"""Offline, bounded single-operation evidence report. Never dispatches or writes.

Usage: python -m auto_ingest.executor_http_report --artifact-dir DIRECTORY
Supply a JSON object with task_id, run_id and url on stdin. The directory
must be explicitly selected by the operator; no discovery or DB connection.
Exit 0 means a classification was produced, not that replay is safe.
"""
from __future__ import annotations

import argparse
import json
import os
import stat
import sys

from auto_ingest.executor_evidence import (
    _http_identity,
    _http_path,
    _MAX_RECEIPT_BYTES,
    inspect_http_observation,
)

MAX_INPUT_BYTES = 16384


def _read_evidence(directory_fd, name):
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        fd = os.open(name, flags, dir_fd=directory_fd)
    except FileNotFoundError:
        return None
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError("nonregular or multiply linked evidence")
        if before.st_size > _MAX_RECEIPT_BYTES:
            raise ValueError("evidence exceeds size limit")
        data = bytearray()
        while len(data) <= _MAX_RECEIPT_BYTES:
            chunk = os.read(fd, _MAX_RECEIPT_BYTES + 1 - len(data))
            if not chunk:
                break
            data.extend(chunk)
        after = os.fstat(fd)
        attrs = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink")
        if (len(data) != before.st_size or len(data) > _MAX_RECEIPT_BYTES
                or any(getattr(before, key) != getattr(after, key) for key in attrs)):
            raise ValueError("evidence changed during read")
        return bytes(data)
    finally:
        os.close(fd)


def report(directory, *, task_id, run_id, url):
    """Read only the two exact filenames in the caller-selected directory.

    Caller must supply a trusted parent path. The directory's final component
    and evidence entries cannot be symlinks. This is not an atomic snapshot
    across both records and does not establish writer authenticity.
    """
    identity = _http_identity(task_id, run_id, url)
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        intent = _read_evidence(fd, _http_path(".", identity, "intent").name)
        outcome = _read_evidence(fd, _http_path(".", identity, "outcome").name)
    finally:
        os.close(fd)
    result = inspect_http_observation(intent, outcome, task_id=task_id, run_id=run_id, url=url)
    return dict(result, task_id=task_id, run_id=run_id,
                target_sha256=identity["target_sha256"])


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate input key")
        result[key] = value
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True)
    args = parser.parse_args()
    try:
        data = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
        if len(data) > MAX_INPUT_BYTES:
            raise ValueError("input exceeds size limit")
        identity = json.loads(data, object_pairs_hook=_unique)
        if not isinstance(identity, dict) or set(identity) != {"task_id", "run_id", "url"}:
            raise ValueError("invalid input fields")
        result = report(args.artifact_dir, **identity)
    except (OSError, ValueError, TypeError, AttributeError, UnicodeError, RecursionError):
        print("HTTP evidence report unavailable; no replay authorization.", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
