"""Dynamic-transaction importer for cleaned intermediates (G5).

This is the replay step for the artifacts produced by
``auto_ingest.ingest_write``. Each cleaned intermediate file (one per media
key + stage) is loaded and written into Neo4j with **dynamic transactions**:
rows are streamed in small ``UNWIND $rows AS r`` batches (``batch`` param, e.g.
500–1000) so a 250k-row artifact never becomes a single giant transaction that
OOMs the ~21M-node graph.

Safety properties (per the live-graph constraint):
  * Every write is ``UNWIND $rows`` + ``MERGE`` keyed on ``key_field`` — bounded
    by ``batch``, no full-graph aggregation, no schema-wide scan.
  * ``MERGE`` makes the import idempotent/resumable: re-running against a key
    that is already partially present updates in place rather than duplicating.
  * The module never imports neo4j at top level; callers pass a ``driver``
    (anything exposing ``driver.session()`` returning a context manager).

Typical use
-----------
    from auto_ingest.ingest_import import import_intermediate, import_all
    import_intermediate(driver, "out/CLT_1423_F.metadata.json",
                         batch=1000, label="LocationSample", key_field="id")
    import_all("out", driver, key="dashcam/2026-07-14/CLT_1423_F")

A stage script produces the artifact with::

    from auto_ingest.ingest_write import write_intermediate
    write_intermediate(out_dir, key, "metadata", cleaned_rows)

and later (possibly on another box) a worker replays it as above. This is the
replayable, resumable import the DEEP_DIVE §3.2 wanted.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, Iterable, List, Optional

# ---------------------------------------------------------------------------
# Loading the artifact
# ---------------------------------------------------------------------------

def _load_rows(path: str) -> List[Dict[str, Any]]:
    """Read an intermediate file (JSON or CSV) into a list of row dicts."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"intermediate not found: {path}")
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        if not isinstance(rows, list):
            raise ValueError(f"intermediate JSON must be a list of row dicts: {path}")
        return [dict(r) for r in rows]
    if path.endswith(".csv"):
        with open(path, "r", encoding="utf-8", newline="") as f:
            return [dict(r) for r in csv.DictReader(f)]
    raise ValueError(f"unsupported intermediate extension (need .json/.csv): {path}")


def _split_row(row: Dict[str, Any], key_field: str) -> tuple[Any, Dict[str, Any]]:
    """Return ``(key_value, props)`` for a row.

    The key field is pulled out as the MERGE key; everything else becomes the
    ``SET n += props`` payload.
    """
    if key_field not in row:
        raise KeyError(f"row missing key_field {key_field!r}: {row!r}")
    key_val = row[key_field]
    props = {k: v for k, v in row.items() if k != key_field}
    return key_val, props


# ---------------------------------------------------------------------------
# Dynamic-tx import
# ---------------------------------------------------------------------------

# UNWIND + MERGE, keyed on the row's key field. MERGE guarantees idempotency:
# a replay over already-present rows updates in place, never duplicates. The
# SET uses `+=` so new columns are added and existing ones overwritten without
# wiping unrelated properties.
_MERGE_QUERY = """
UNWIND $rows AS r
MERGE (n:{label} {{{key_field}: r.key}})
SET n += r.props
"""

# Same as above but also attaches the node to a parent keyed by `parent_field`,
# e.g. link a :LocationSample to its :Transcription via t_id. Optional.
_MERGE_WITH_EDGE_QUERY = """
UNWIND $rows AS r
MERGE (n:{label} {{{key_field}: r.key}})
SET n += r.props
WITH n, r
MATCH (p:{parent_label} {{{parent_field}: r.parent}})
MERGE (p)-[:{edge_label}]->(n)
"""


def import_intermediate(driver, path: str, *, batch: int = 1000,
                        label: str = "Chunk", key_field: str = "key",
                        parent_label: Optional[str] = None,
                        parent_field: Optional[str] = None,
                        edge_label: Optional[str] = None) -> int:
    """Import one cleaned intermediate file into Neo4j in dynamic batches.

    Rows are read and streamed to ``driver`` in ``UNWIND $rows`` chunks of size
    ``batch``. Each row MERGEs a ``label`` node keyed by ``key_field`` and sets
    the remaining fields as properties. Re-runs are safe (MERGE).

    Returns the number of rows imported.

    If ``parent_label``/``parent_field``/``edge_label`` are all given, each node
    is also attached to a parent node (matched by ``r.parent``) via
    ``(parent)-[:edge_label]->(node)`` — useful for e.g. linking a
    :LocationSample to its :Transcription.
    """
    if batch < 1:
        raise ValueError("batch must be >= 1")
    rows = _load_rows(path)
    if not rows:
        return 0

    has_edge = bool(parent_label and parent_field and edge_label)
    if has_edge and any("parent" not in r for r in rows):
        raise KeyError("rows must carry a 'parent' field when linking to a parent node")

    # Pre-split each row into {"key", "props"} (and "parent" when linking).
    prepared = []
    for r in rows:
        key_val, props = _split_row(r, key_field)
        item: Dict[str, Any] = {"key": key_val, "props": props}
        if has_edge:
            item["parent"] = r["parent"]
        prepared.append(item)

    if has_edge:
        cy = _MERGE_WITH_EDGE_QUERY.format(
            label=label, key_field=key_field,
            parent_label=parent_label, parent_field=parent_field, edge_label=edge_label,
        )
    else:
        cy = _MERGE_QUERY.format(label=label, key_field=key_field)

    total = 0
    with driver.session() as sess:
        for i in range(0, len(prepared), batch):
            chunk = prepared[i:i + batch]
            sess.run(cy, rows=chunk)
            total += len(chunk)
    return total


# ---------------------------------------------------------------------------
# Multi-stage / multi-key orchestration
# ---------------------------------------------------------------------------

# Stage import order matters: a parent stage (e.g. Transcription header) must
# exist before a child stage (e.g. Segments, LocationSamples) links to it. The
# order below is the canonical ingest pipeline order. ``import_all`` sorts the
# discovered stages by this list when not given an explicit ``stages``.
STAGE_ORDER = [
    "transcription",   # Transcription header / :Transcription node
    "segments",        # :Segment + edges
    "utterances",      # :Utterance + edges
    "speakers",        # :Speaker nodes
    "entities",        # :Entity + MENTIONS
    "metadata",        # :LocationSample + HAS_LOCATION
    "summary",         # :Summary + Tasks
]

# Default label/key for each known stage. Override per call if needed.
STAGE_LABEL = {
    "transcription": "Transcription",
    "segments": "Segment",
    "utterances": "Utterance",
    "speakers": "Speaker",
    "entities": "Entity",
    "metadata": "LocationSample",
    "summary": "Summary",
}

# Stages that link to a parent Transcription node (need parent_field/edge).
_STAGES_WITH_PARENT = {"segments", "utterances", "entities", "metadata"}

# Real relationship type used to attach each child stage to its Transcription.
_STAGE_EDGE = {
    "segments": "HAS_SEGMENT",
    "utterances": "HAS_UTTERANCE",
    "entities": "MENTIONS",
    "metadata": "HAS_LOCATION",
}


def _sort_stages(stages: List[str]) -> List[str]:
    return sorted(stages, key=lambda s: STAGE_ORDER.index(s) if s in STAGE_ORDER else 999)


def import_all(out_dir: str, driver, key: Optional[str] = None,
               stages: Optional[Iterable[str]] = None, batch: int = 1000) -> Dict[str, int]:
    """Import every intermediate under ``out_dir`` for ``key`` (or all keys).

    Stages are imported in canonical ``STAGE_ORDER`` so parents exist before
    children link to them. Returns a ``{stage: rows_imported}`` summary.

    Pass ``stages`` to import only a subset (e.g. ``("metadata",)``).
    """
    found = list_intermediates_for_import(out_dir, key=key)
    if not found:
        return {}

    # Group by (key, stage) keeping the preferred format (json over csv).
    best: Dict[tuple[str, str], Dict[str, str]] = {}
    for item in found:
        k = (item["key"], item["stage"])
        if k not in best or (item["fmt"] == "json" and best[k]["fmt"] != "json"):
            best[k] = item

    wanted_stages = set(stages) if stages is not None else None
    selected = [v for (k, v) in best.items() if wanted_stages is None or k[1] in wanted_stages]

    summary: Dict[str, int] = {}
    for item in sorted(selected, key=lambda v: (v["key"], STAGE_ORDER.index(v["stage"]) if v["stage"] in STAGE_ORDER else 999)):
        stage = item["stage"]
        label = STAGE_LABEL.get(stage, stage.capitalize())
        kwargs: Dict[str, Any] = {"batch": batch, "label": label, "key_field": "id" if stage != "transcription" else "id"}
        if stage in _STAGES_WITH_PARENT:
            kwargs.update(parent_label="Transcription", parent_field="id",
                          edge_label=_STAGE_EDGE[stage])
        n = import_intermediate(driver, item["path"], **kwargs)
        summary[f"{item['key']}/{stage}"] = n
    return summary


# ---------------------------------------------------------------------------
# Discovery helper (mirrors ingest_write.list_intermediates)
# ---------------------------------------------------------------------------

def list_intermediates_for_import(out_dir: str, key: Optional[str] = None,
                                  stage: Optional[str] = None) -> List[Dict[str, str]]:
    """List intermediates for import (thin wrapper; lives here to keep the
    importer self-contained). Delegate to the writer's discovery."""
    from auto_ingest.ingest_write import list_intermediates
    return list_intermediates(out_dir, key=key, stage=stage)
