#!/usr/bin/env python3
"""Fail if a pinned optional extra in pyproject.toml drifts from requirements.txt.

Standard library only; no new dependencies. The [project.optional-dependencies]
table is read with a small line-based parser so this also runs on Python 3.10
(where ``tomllib`` is not yet available).

Only distributions that are pinned in requirements.txt are checked. Packages
that appear in the extras but have no pin in requirements.txt (currently
sentence-transformers, pydantic and ruff) are reported as skipped: there is no
authoritative version to compare them against.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS = ROOT / "requirements.txt"
PYPROJECT = ROOT / "pyproject.toml"

_NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")
_PIN_RE = re.compile(r"==\s*([^\s;,]+)")
_QUOTED_RE = re.compile(r'"([^"]+)"')
_SECTION_RE = re.compile(r"^\s*\[([^\]]+)\]\s*(?:#.*)?$")


def normalize(name: str) -> str:
    """PEP 503 name normalization: lowercase, collapse runs of [-_.] to '-'."""
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_requirements(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        match = _NAME_RE.match(line)
        pin = _PIN_RE.search(line)
        if match and pin:
            pins[normalize(match.group(1))] = pin.group(1)
    return pins


def parse_extras(path: Path) -> list[tuple[str, str]]:
    """Return (extra_name, requirement_string) pairs from the extras table."""
    lines = path.read_text(encoding="utf-8").splitlines()
    in_extras = False
    current_key: str | None = None
    collected: list[tuple[str, str]] = []
    for raw in lines:
        stripped = raw.strip()
        section = _SECTION_RE.match(stripped)
        if section:
            in_extras = section.group(1).strip() == "project.optional-dependencies"
            current_key = None
            continue
        if not in_extras or not stripped or stripped.startswith("#"):
            continue
        if "=" in stripped and "[" in stripped:
            current_key = stripped.split("=", 1)[0].strip()
        for requirement in _QUOTED_RE.findall(stripped):
            collected.append((current_key or "<unknown>", requirement))
    return collected


def check() -> int:
    pins = parse_requirements(REQUIREMENTS)
    extras = parse_extras(PYPROJECT)
    if not pins:
        print(f"FAIL: no pins parsed from {REQUIREMENTS}", file=sys.stderr)
        return 1
    if not extras:
        print(f"FAIL: no extras parsed from {PYPROJECT}", file=sys.stderr)
        return 1

    errors: list[str] = []
    skipped: list[str] = []
    checked = 0
    for extra, requirement in extras:
        match = _NAME_RE.match(requirement.strip())
        if not match:
            continue
        name = normalize(match.group(1))
        if name not in pins:
            skipped.append(f"{name} (no pin in requirements.txt)")
            continue
        checked += 1
        expected = pins[name]
        pin = _PIN_RE.search(requirement)
        if pin is None:
            errors.append(
                f"extra '{extra}': {requirement!r} is not an exact '==' pin; "
                f"requirements.txt pins {name}=={expected}"
            )
        elif pin.group(1) != expected:
            errors.append(
                f"extra '{extra}': {requirement!r} pins {pin.group(1)}, "
                f"requirements.txt pins {expected}"
            )

    print(
        f"dependency pin drift check: {checked} overlapping extra(s) checked, "
        f"{len(skipped)} skipped (no requirements.txt pin)"
    )
    if skipped:
        print("  skipped: " + ", ".join(skipped))
    if errors:
        for error in errors:
            print(f"  DRIFT: {error}", file=sys.stderr)
        print(f"FAIL: {len(errors)} drift(s) detected", file=sys.stderr)
        return 1
    print("PASS: all overlapping extras match requirements.txt pins")
    return 0


if __name__ == "__main__":
    sys.exit(check())
