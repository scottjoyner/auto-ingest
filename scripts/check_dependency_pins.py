#!/usr/bin/env python3
"""Validate optional-extra pins against the flat, exact requirements lock.

This CI utility uses stdlib tomllib (Python 3.11+, CI runs 3.12).
It does not change the application's Python 3.10 runtime support.
Unsupported or ambiguous lock syntax fails closed rather than being skipped.
"""
from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS = ROOT / "requirements.txt"
PYPROJECT = ROOT / "pyproject.toml"

_NAME = r"[A-Za-z0-9][A-Za-z0-9._-]*"
_NAME_RE = re.compile(r"^\s*(" + _NAME + r")(?=\s|\[|[<>=!~;@]|$)")
_VERSION = r"[A-Za-z0-9][A-Za-z0-9.+!_-]*"
_EXACT_RE = re.compile(r"\s*(" + _NAME + r")\s*==\s*(" + _VERSION + r")\s*")


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_requirements(path: Path) -> dict[str, str]:
    pins = {}
    for index, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        match = _EXACT_RE.fullmatch(line)
        if not match:
            raise ValueError(f"unsupported requirements lock syntax at line {index}")
        name, version = normalize(match.group(1)), match.group(2)
        if name in pins:
            raise ValueError(f"duplicate normalized requirements pin: {name}")
        pins[name] = version
    if not pins:
        raise ValueError("empty requirements lock")
    return pins


def parse_extras(path: Path) -> list[tuple[str, str]]:
    with path.open("rb") as stream:
        project = tomllib.load(stream)
    extras = project.get("project", {}).get("optional-dependencies")
    if not isinstance(extras, dict) or not extras:
        raise ValueError("missing optional-dependencies table")
    collected = []
    for extra, requirements in extras.items():
        if not isinstance(requirements, list) or any(
            not isinstance(item, str) or not item.strip() for item in requirements
        ):
            raise ValueError(f"invalid optional dependency array: {extra}")
        collected.extend((extra, req) for req in requirements)
    if not collected:
        raise ValueError("no optional dependencies")
    return collected


def check(requirements_path=REQUIREMENTS, pyproject_path=PYPROJECT) -> int:
    try:
        pins = parse_requirements(requirements_path)
        extras = parse_extras(pyproject_path)
    except (OSError, UnicodeError, ValueError, TypeError, AttributeError) as exc:
        # TOML parse errors are safe to report only by type: malformed input
        # could contain arbitrary text.
        print(f"FAIL: invalid dependency metadata ({type(exc).__name__})", file=sys.stderr)
        return 1
    errors = []
    skipped = []
    checked = 0
    for extra, requirement in extras:
        name_match = _NAME_RE.match(requirement)
        if not name_match:
            errors.append(f"extra '{extra}': invalid requirement name")
            continue
        name = normalize(name_match.group(1))
        if name not in pins:
            skipped.append(name)
            continue
        checked += 1
        pin = _EXACT_RE.fullmatch(requirement)
        if pin is None or pin.group(2) != pins[name]:
            errors.append(f"extra '{extra}': {name} must be pinned exactly to {pins[name]}")
    if not checked:
        errors.append("no overlapping optional dependency pins checked")
    print(f"dependency pin drift check: {checked} overlapping extra(s) checked, {len(skipped)} skipped")
    if skipped:
        print("  skipped (no requirements.txt pin): " + ", ".join(skipped))
    for error in errors:
        print("  DRIFT: " + error, file=sys.stderr)
    if errors:
        print(f"FAIL: {len(errors)} drift(s) detected", file=sys.stderr)
        return 1
    print("PASS: all overlapping extras match requirements.txt pins")
    return 0


if __name__ == "__main__":
    sys.exit(check())
