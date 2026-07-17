"""Stage writer: emit cleaned, replayable intermediates (G5).

This module is the "cleaned intermediate" half of the DEEP_DIVE §3.2 / §3.3
design. An ingest *stage* (transcribe, metadata, summarize, ...) can write its
cleaned, normalized rows to a per-(key, stage) file instead of (or in addition
to) writing straight into the 21M-node Neo4j graph. A later, separate step
replays those files into the graph with *dynamic transactions* via
``auto_ingest.ingest_import`` — bounded, resumable, OOM-safe.

Why a plain file instead of a DB?
  * It is trivially resumable: a worker that dies mid-stage can just re-read
    the artifact it already produced; no partial-graph cleanup needed.
  * It is claim-friendly: the file lives next to the media key under ``out_dir``
    and is the unit a distributed worker produces/consumes (see
    ``ingest_claim``).
  * It is pure: no neo4j import at module level, so it is safe to import in any
    context (tests, workers, CI) without dragging in the driver.

Format
------
* ``json`` (default): ``<out_dir>/<key>.<stage>.json`` — a JSON list of dict
  rows. Human-readable, preserves types, easy to diff/claim.
* ``csv``: ``<out_dir>/<key>.<stage>.csv`` — flat rows; good for huge tables.

Idempotency
-----------
``write_intermediate`` *overwrites* the (key, stage) file, so re-running a stage
for the same key replaces its artifact rather than appending duplicates. This
matches the resumable model: the artifact represents "the cleaned output of
this stage for this key, as of the latest run".
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, Iterable, List, Optional

# Allowed formats. Anything else raises ValueError at call time.
SUPPORTED_FORMATS = ("json", "csv")


def _validate_fmt(fmt: str) -> None:
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(
            f"unsupported format {fmt!r}; expected one of {SUPPORTED_FORMATS}"
        )


def _path(out_dir: str, key: str, stage: str, fmt: str) -> str:
    _validate_fmt(fmt)
    # Keys may contain path separators (e.g. "dashcam/2026-07-14/CLT_1423_F").
    # Replace "/" so the artifact lands in out_dir as a single file and does
    # not try to create nested directories we didn't ask for.
    safe_key = key.replace("/", "__")
    return os.path.join(out_dir, f"{safe_key}.{stage}.{fmt}")


def write_intermediate(out_dir: str, key: str, stage: str,
                       records: Iterable[Dict[str, Any]], *, fmt: str = "json") -> str:
    """Write the cleaned ``records`` for ``(key, stage)`` to ``out_dir``.

    Idempotent: overwrites any existing artifact for this (key, stage). Records
    must be plain JSON-serializable dicts (no neo4j objects, no numpy arrays —
    the stage is responsible for producing graph-ready values).

    Returns the path written.
    """
    _validate_fmt(fmt)
    os.makedirs(out_dir, exist_ok=True)
    rows: List[Dict[str, Any]] = [dict(r) for r in records]
    path = _path(out_dir, key, stage, fmt)
    if fmt == "json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2, sort_keys=True)
    elif fmt == "csv":
        if not rows:
            # Still emit a header-less empty file so the artifact exists.
            open(path, "w", encoding="utf-8").close()
        else:
            fields: List[str] = []
            for r in rows:
                for k in r.keys():
                    if k not in fields:
                        fields.append(k)
            with open(path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                for r in rows:
                    writer.writerow(r)
    return path


def read_intermediate(out_dir: str, key: str, stage: str, *, fmt: str = "json") -> List[Dict[str, Any]]:
    """Load the cleaned rows previously written by ``write_intermediate``.

    Raises ``FileNotFoundError`` if no artifact exists for ``(key, stage)``.
    """
    _validate_fmt(fmt)
    path = _path(out_dir, key, stage, fmt)
    if not os.path.exists(path):
        raise FileNotFoundError(f"no intermediate for key={key!r} stage={stage!r} at {path}")
    if fmt == "json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    elif fmt == "csv":
        with open(path, "r", encoding="utf-8", newline="") as f:
            return [dict(r) for r in csv.DictReader(f)]
    raise ValueError(fmt)  # pragma: no cover - _validate_fmt already guards


def list_intermediates(out_dir: str, key: Optional[str] = None,
                       stage: Optional[str] = None) -> List[Dict[str, str]]:
    """Discover produced intermediates, for resume/claim logic.

    Returns a list of ``{"key", "stage", "fmt", "path"}`` dicts, newest-first by
    mtime. Filter by ``key`` (exact, pre-sanitization form) and/or ``stage``.
    If a ``(key, stage)`` exists in more than one format, both are returned.
    """
    if not os.path.isdir(out_dir):
        return []
    results: List[Dict[str, str]] = []
    for name in os.listdir(out_dir):
        # Match <safe_key>.<stage>.<fmt> with a known fmt suffix.
        for fmt in SUPPORTED_FORMATS:
            if not name.endswith(f".{fmt}"):
                continue
            body = name[: -(len(fmt) + 1)]  # drop ".<fmt>"
            if "." not in body:
                continue
            key_part, stage_part = body.rsplit(".", 1)
            # Reverse the "/" -> "__" sanitization done at write time.
            real_key = key_part.replace("__", "/")
            if key is not None and real_key != key:
                continue
            if stage is not None and stage_part != stage:
                continue
            results.append({
                "key": real_key,
                "stage": stage_part,
                "fmt": fmt,
                "path": os.path.join(out_dir, name),
            })
    results.sort(key=lambda r: os.path.getmtime(r["path"]), reverse=True)
    return results
