"""Readiness probe for production ingest services.

A process is ready only when Neo4j is reachable and the runtime identity schema
is enforced. Queue pressure/quarantine counts are reported but do not by
themselves make the service unready.
"""
from __future__ import annotations

import argparse
import json
from typing import Sequence

from auto_ingest.metrics import collect_metrics
from auto_ingest.runtime_schema import audit_schema


def readiness(driver) -> dict:
    with driver.session() as session:
        rec = session.run("RETURN 1 AS ok").single()
    db_ok = bool(rec and rec.get("ok") == 1)
    schema = audit_schema(driver) if db_ok else {"ok": False, "missing_constraints": []}
    metrics = collect_metrics(driver) if db_ok else {}
    return {
        "ready": bool(db_ok and schema.get("ok")),
        "database": db_ok,
        "schema": schema,
        "metrics": metrics,
    }


def _driver():
    from neo4j import GraphDatabase

    from auto_ingest_config import get_neo4j_config

    cfg = get_neo4j_config()
    return GraphDatabase.driver(cfg["uri"], auth=(cfg["user"], cfg["password"]))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m auto_ingest.readiness")
    parser.add_argument("--json", action="store_true", help="Emit structured readiness payload.")
    args = parser.parse_args(argv)
    driver = _driver()
    try:
        report = readiness(driver)
        if args.json:
            print(json.dumps(report, indent=2, default=str))
        elif report["ready"]:
            print("ready")
        else:
            print("not ready")
        return 0 if report["ready"] else 1
    finally:
        driver.close()


if __name__ == "__main__":
    raise SystemExit(main())
