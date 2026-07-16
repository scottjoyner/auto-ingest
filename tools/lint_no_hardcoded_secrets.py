#!/usr/bin/env python3
"""Lint: standardize the Neo4j password default (W-45 / W-53).

The historical password ``knowledge_graph_2026`` is still supported as a default,
but it must be sourced through the single canonical env var
``NEO4J_PASSWORD_DEFAULT`` rather than hardcoded inline everywhere. The accepted
resolution chain (in any script) is::

    NEO4J_PASSWORD  ->  NEO4J_PASSWORD_DEFAULT  ->  "knowledge_graph_2026"

This linter allows a line that contains the literal ONLY when that same line also
references ``NEO4J_PASSWORD_DEFAULT`` (i.e. it is the documented fallback), or when
the file is one of the canonical definition sites (auto_ingest_config.py, the
.env* / *.env.example templates). Every other bare occurrence fails so CI can gate
regressions back to the old inline hardcoding.

Usage:
    python tools/lint_no_hardcoded_secrets.py
"""
from __future__ import annotations

from pathlib import Path

LITERAL = "knowledge_graph_2026"
CANONICAL_ENV = "NEO4J_PASSWORD_DEFAULT"

# Files allowed to define the literal as the baked-in default / template value.
ALLOWLIST_NAMES = {
    "auto_ingest_config.py",       # the one baked-in default (_BAKED_IN_NEO4J_PASSWORD)
    "lint_no_hardcoded_secrets.py",  # this file documents the rule
}
ALLOWLIST_SUFFIXES = (
    ".env",
    ".env.example",
    ".env.bak",
)

SCAN_GLOBS = ("*.py", "*.sh")
SKIP_DIR_MARKERS = ("/.git/", "/archive/", "/__pycache__/", "/.venv/", "/node_modules/")


def _is_allowed_line(line: str) -> bool:
    """A literal occurrence is fine if it is the documented default fallback."""
    if CANONICAL_ENV in line:
        return True
    # Pure comments that merely mention it (docs/examples) are allowed.
    stripped = line.lstrip()
    if stripped.startswith("#"):
        return True
    return False


def _is_allowed_file(path: Path) -> bool:
    if path.name in ALLOWLIST_NAMES:
        return True
    # Test files/dirs legitimately reference the literal to exercise resolution.
    if path.name.startswith("test_") or "/tests/" in str(path).replace("\\", "/"):
        return True
    name = path.name
    return any(name.endswith(sfx) or name == sfx.lstrip(".") for sfx in ALLOWLIST_SUFFIXES) \
        or name.startswith(".env")


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    hits = []
    for pattern in SCAN_GLOBS:
        for p in root.rglob(pattern):
            sp = str(p)
            if any(m in sp for m in SKIP_DIR_MARKERS):
                continue
            if _is_allowed_file(p):
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="strict")
            except Exception:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if LITERAL in line and not _is_allowed_line(line):
                    hits.append(f"{p.relative_to(root)}:{i}: {line.strip()}")

    if hits:
        print("W-53 lint FAILED — hardcoded Neo4j password not routed through "
              f"{CANONICAL_ENV}:")
        for h in hits:
            print("  " + h)
        print(f"\nFix: resolve via  NEO4J_PASSWORD -> {CANONICAL_ENV} -> "
              f'"{LITERAL}"  (see auto_ingest_config.get_neo4j_password).')
        return 1
    print(f"W-53 lint OK — every '{LITERAL}' default is routed through {CANONICAL_ENV}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
