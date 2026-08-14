"""Read-only persisted invariants for production ingest state.

These checks intentionally detect impossible states rather than application-level
business outcomes. They are safe to run in diagnostics, CI, or against production
because they never mutate graph data.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class Invariant:
    name: str
    query: str
    description: str


INVARIANTS = (
    Invariant(
        "active_job_requires_lease",
        """
        MATCH (j:IngestJob)
        WHERE j.lifecycle_state IN ['CLAIMED','RUNNING','VALIDATING']
          AND (coalesce(j.owner,'') = '' OR coalesce(j.fence_token,0) < 1
               OR coalesce(j.heartbeat_at,j.claimed_at,0) <= 0)
        RETURN j.key AS key, j.lifecycle_state AS state, j.owner AS owner,
               coalesce(j.fence_token,0) AS fence_token
        LIMIT $limit
        """,
        "Active lifecycle jobs require an owner, positive fence token, and lease timestamp.",
    ),
    Invariant(
        "done_job_requires_graph_stage",
        """
        MATCH (j:IngestJob)
        WHERE (j.status='done' OR j.lifecycle_state='DONE')
          AND NOT 'graph_written' IN coalesce(j.completed_stages,[])
          AND NOT 'graph-written' IN coalesce(j.completed_tasks,[])
        RETURN j.key AS key, j.status AS status, j.lifecycle_state AS lifecycle_state
        LIMIT $limit
        """,
        "Completed jobs must record the graph-write terminal stage/task.",
    ),
    Invariant(
        "artifact_requires_provenance_fields",
        """
        MATCH (a:IngestArtifact)
        WHERE coalesce(a.artifact_id,'') = '' OR coalesce(a.path,'') = ''
           OR coalesce(a.sha256,'') = '' OR coalesce(a.source_hash,'') = ''
           OR coalesce(a.stage,'') = ''
        RETURN a.artifact_id AS artifact_id, a.path AS path, a.stage AS stage
        LIMIT $limit
        """,
        "Artifacts require identity, durable path, digest, source identity, and stage.",
    ),
    Invariant(
        "artifact_requires_producer",
        """
        MATCH (a:IngestArtifact)
        WHERE NOT (:IngestJob)-[:PRODUCED]->(a)
        RETURN a.artifact_id AS artifact_id, a.path AS path
        LIMIT $limit
        """,
        "Every registered artifact must have an ingest-job provenance edge.",
    ),
)


def audit(driver, *, limit: int = 50) -> dict[str, Any]:
    if limit < 1:
        raise ValueError("limit must be positive")
    results: dict[str, Any] = {}
    with driver.session() as session:
        for invariant in INVARIANTS:
            rows = session.run(invariant.query, limit=limit).data()
            results[invariant.name] = {
                "ok": not rows,
                "description": invariant.description,
                "violations": [dict(row) for row in rows],
            }
    return {
        "ok": all(item["ok"] for item in results.values()),
        "invariants": results,
    }


def _driver():
    from neo4j import GraphDatabase
    from auto_ingest_config import get_neo4j_config

    cfg = get_neo4j_config()
    return GraphDatabase.driver(cfg["uri"], auth=(cfg["user"], cfg["password"]))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m auto_ingest.invariants")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    driver = _driver()
    try:
        report = audit(driver, limit=args.limit)
    finally:
        driver.close()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
    else:
        for name, item in report["invariants"].items():
            print(f"{'PASS' if item['ok'] else 'FAIL'} {name}: {item['description']}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
