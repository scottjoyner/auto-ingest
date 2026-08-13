"""Dependency-free Prometheus text metrics for the ingest control plane."""
from __future__ import annotations

import argparse
from typing import Sequence

LIFECYCLE_STATES = (
    "DISCOVERED",
    "READY",
    "CLAIMED",
    "RUNNING",
    "VALIDATING",
    "DONE",
    "RETRY",
    "FAILED",
    "QUARANTINED",
)


def collect_metrics(driver, *, stale_after_sec: int = 600) -> dict:
    if stale_after_sec < 1:
        raise ValueError("stale_after_sec must be positive")
    cutoff = int(__import__("time").time() * 1000) - stale_after_sec * 1000
    with driver.session() as session:
        state_rows = session.run(
            """
            MATCH (j:IngestJob)
            WITH coalesce(j.lifecycle_state,'UNKNOWN') AS state, count(*) AS count
            RETURN state, count
            """
        ).data()
        active = session.run(
            """
            MATCH (j:IngestJob)
            WHERE coalesce(j.owner,'') <> ''
            RETURN count(j) AS count
            """
        ).single()
        stale = session.run(
            """
            MATCH (j:IngestJob)
            WHERE j.lifecycle_state IN ['CLAIMED','RUNNING','VALIDATING']
              AND coalesce(j.heartbeat_at,j.claimed_at,0) < $cutoff
            RETURN count(j) AS count
            """,
            cutoff=cutoff,
        ).single()
        artifacts = session.run(
            "MATCH (a:IngestArtifact) RETURN count(a) AS count"
        ).single()
    by_state = {row["state"]: int(row["count"]) for row in state_rows}
    return {
        "jobs_by_state": by_state,
        "active_leases": int(active["count"] if active else 0),
        "stale_jobs": int(stale["count"] if stale else 0),
        "artifacts": int(artifacts["count"] if artifacts else 0),
    }


def render_prometheus(metrics: dict) -> str:
    lines = [
        "# HELP auto_ingest_jobs Number of persisted ingest jobs by lifecycle state.",
        "# TYPE auto_ingest_jobs gauge",
    ]
    states = dict(metrics.get("jobs_by_state", {}))
    for state in sorted(set(states) | set(LIFECYCLE_STATES)):
        lines.append(f'auto_ingest_jobs{{state="{state}"}} {int(states.get(state, 0))}')
    lines += [
        "# HELP auto_ingest_active_leases Number of jobs with a current lease owner.",
        "# TYPE auto_ingest_active_leases gauge",
        f"auto_ingest_active_leases {int(metrics.get('active_leases', 0))}",
        "# HELP auto_ingest_stale_jobs Claimed/running jobs with stale heartbeat/claim timestamps.",
        "# TYPE auto_ingest_stale_jobs gauge",
        f"auto_ingest_stale_jobs {int(metrics.get('stale_jobs', 0))}",
        "# HELP auto_ingest_artifacts Number of registered content-addressed ingest artifacts.",
        "# TYPE auto_ingest_artifacts gauge",
        f"auto_ingest_artifacts {int(metrics.get('artifacts', 0))}",
    ]
    return "\n".join(lines) + "\n"


def _driver():
    from neo4j import GraphDatabase

    from auto_ingest_config import get_neo4j_config

    cfg = get_neo4j_config()
    return GraphDatabase.driver(cfg["uri"], auth=(cfg["user"], cfg["password"]))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m auto_ingest.metrics")
    parser.add_argument("--stale-after-sec", type=int, default=600)
    args = parser.parse_args(argv)
    driver = _driver()
    try:
        print(render_prometheus(collect_metrics(driver, stale_after_sec=args.stale_after_sec)), end="")
        return 0
    finally:
        driver.close()


if __name__ == "__main__":
    raise SystemExit(main())
