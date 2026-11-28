"""Read-only exact-operation operator CLI.

Requires a separately admitted local Bolt connection and a read-only database
principal. No discovery, default credentials, remote dispatch or replay.
Identity is read from stdin, never a URL command-line argument.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from auto_ingest.executor_evidence import _http_identity
from auto_ingest.executor_http_report import MAX_INPUT_BYTES, _unique
from auto_ingest.executor_operation_report import reconcile_operation


class _DatabaseDriver:
    def __init__(self, driver, database):
        self.driver = driver
        self.database = database

    def session(self, **kwargs):
        if kwargs != {"default_access_mode": "READ"}:
            raise RuntimeError("read-only session required")
        return self.driver.session(database=self.database, **kwargs)


def main(argv=None, *, driver_factory=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--bolt-port", type=int, required=True)
    parser.add_argument("--database", required=True)
    parser.add_argument("--confirm-read-only-admission", action="store_true", required=True,
                        help="Operator confirms independent admission and read-only principal")
    args = parser.parse_args(argv)
    try:
        if not 1 <= args.bolt_port <= 65535:
            raise ValueError("invalid port")
        if not args.database.strip() or len(args.database) > 63:
            raise ValueError("invalid database")
        data = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
        if len(data) > MAX_INPUT_BYTES:
            raise ValueError("input exceeds limit")
        identity = json.loads(data, object_pairs_hook=_unique)
        if not isinstance(identity, dict) or set(identity) != {"task_id", "run_id", "url"}:
            raise ValueError("invalid input fields")
        _http_identity(**identity)
        user = os.environ.get("EXECUTOR_REPORT_USER")
        password = os.environ.get("EXECUTOR_REPORT_PASSWORD")
        if not user or not password:
            raise ValueError("explicit read-only credentials required")
        if driver_factory is None:
            from neo4j import GraphDatabase
            driver_factory = GraphDatabase.driver
        with driver_factory(f"bolt://127.0.0.1:{args.bolt_port}", auth=(user, password),
                            connection_timeout=3, connection_acquisition_timeout=5,
                            max_transaction_retry_time=0) as driver:
            result = reconcile_operation(_DatabaseDriver(driver, args.database),
                                         args.artifact_dir, **identity)
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception:
        # Driver diagnostics may include sensitive parameters. Never print them.
        print("Operation report unavailable; no replay or activation authorization.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
