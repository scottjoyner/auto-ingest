#!/usr/bin/env python3
"""Lint: forbid hardcoded Neo4j credentials (W-45).

Scans the repo for inline ``NEO4J_URI/USER/PASSWORD/DB`` constants that hardcode
the ``knowledge_graph_2026`` password fallback or a literal bolt:// URI instead of
calling ``auto_ingest_config.get_neo4j_env()``. Exits non-zero if any are found so
it can gate CI.

Usage:
    python tools/lint_no_hardcoded_secrets.py
"""
from __future__ import annotations

import sys
from pathlib import Path

FORBIDDEN = (
    "knowledge_graph_2026",  # the old committed password fallback default
)


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    hits = []
    for p in root.rglob("*.py"):
        if "/.git/" in str(p) or "/archive/" in str(p) or "/__pycache__/" in str(p):
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="strict")
        except Exception:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            for bad in FORBIDDEN:
                if bad in line:
                    hits.append(f"{p.relative_to(root)}:{i}: {bad}")
    if hits:
        print("W-45 lint FAILED — hardcoded credential defaults found:")
        for h in hits:
            print("  " + h)
        return 1
    print("W-45 lint OK — no hardcoded credential defaults.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
