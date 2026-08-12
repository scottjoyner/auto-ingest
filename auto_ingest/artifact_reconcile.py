"""Reconcile filesystem commit journals with Neo4j artifact provenance."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from auto_ingest.commit_protocol import verify_artifact


@dataclass(frozen=True)
class ReconcileFinding:
    artifact_id: str
    journal_state: str
    file_ok: bool
    graph_present: bool
    classification: str


def iter_journals(root: str | Path) -> Iterable[dict]:
    base = Path(root) / ".commit-journal"
    if not base.exists():
        return []
    records = []
    for path in sorted(base.rglob("*.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


def graph_artifact_exists(driver, artifact_id: str) -> bool:
    with driver.session() as session:
        rec = session.run(
            "MATCH (a:IngestArtifact {artifact_id:$id}) RETURN count(a) AS n",
            id=artifact_id,
        ).single()
    return bool(rec and rec.get("n"))


def classify_record(driver, record: dict) -> ReconcileFinding:
    aid = record["artifact_id"]
    path = record.get("artifact_path")
    digest = record.get("artifact_sha256")
    file_ok = bool(path and digest and verify_artifact(path, digest))
    graph_present = graph_artifact_exists(driver, aid)
    state = record.get("state", "UNKNOWN")

    if state == "COMMITTED" and file_ok and graph_present:
        classification = "HEALTHY"
    elif file_ok and graph_present:
        classification = "JOURNAL_STALE"
    elif file_ok and not graph_present:
        classification = "ORPHAN_FILE"
    elif not file_ok and graph_present:
        classification = "GRAPH_DANGLING"
    else:
        classification = "INCOMPLETE"
    return ReconcileFinding(aid, state, file_ok, graph_present, classification)


def reconcile(root: str | Path, driver) -> list[ReconcileFinding]:
    return [classify_record(driver, record) for record in iter_journals(root)]
