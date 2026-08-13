"""Config-backed entrypoint for dashcam YOLO embedding ingestion.

The historical module exposes Neo4j credentials as CLI options with unsafe
standalone defaults. This wrapper resolves canonical config, injects arguments
inside the Python process (not the OS command line), and invokes the existing
implementation without exposing the database password through process listings.
"""
from __future__ import annotations

import os
import sys
from typing import Sequence


def build_internal_argv(extra: Sequence[str] | None = None) -> list[str]:
    from auto_ingest_config import get_neo4j_config

    cfg = get_neo4j_config()
    uri = os.environ.get("NEO4J_URI") or cfg["uri"]
    user = os.environ.get("NEO4J_USER") or cfg["user"]
    password = os.environ.get("NEO4J_PASSWORD") or cfg["password"]
    args = [
        "yolo_embeddings",
        "--neo4j-uri",
        uri,
        "--neo4j-user",
        user,
        "--neo4j-pass",
        password,
        "--resume",
        "--grid",
        os.environ.get("YOLO_GRID", "16x9"),
        "--pyramid",
        "--heatmap",
        "--repair-missing-moov",
        "--win-mins",
        os.environ.get("YOLO_LOCATION_WINDOW_MINS", "10"),
    ]
    args.extend(extra or ())
    return args


def main(argv: Sequence[str] | None = None) -> int:
    from auto_ingest.dashcam import yolo_embeddings

    previous = sys.argv
    try:
        sys.argv = build_internal_argv(argv)
        result = yolo_embeddings.main()
        return int(result or 0)
    finally:
        sys.argv = previous


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
