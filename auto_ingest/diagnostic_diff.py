"""Compare sanitized diagnostics captured on two machines."""
from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path
from typing import Any, Mapping, Sequence


class DiagnosticDiffError(ValueError):
    pass


def load_report(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if source.name.endswith(".tar.gz"):
        with tarfile.open(source, "r:gz") as archive:
            try:
                member = archive.extractfile("auto-ingest-diagnostics/diagnostics.json")
            except KeyError as exc:
                raise DiagnosticDiffError("archive does not contain diagnostics.json") from exc
            if member is None:
                raise DiagnosticDiffError("archive diagnostics.json is not readable")
            payload = json.loads(member.read().decode("utf-8"))
    else:
        payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DiagnosticDiffError("diagnostics report must be a JSON object")
    return payload


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {prefix: value}
    out: dict[str, Any] = {}
    for key in sorted(value):
        path = f"{prefix}.{key}" if prefix else str(key)
        child = value[key]
        if isinstance(child, Mapping):
            out.update(_flatten(child, path))
        elif isinstance(child, list):
            out[path] = child
        else:
            out[path] = child
    return out


def compare(left: Mapping[str, Any], right: Mapping[str, Any]) -> list[dict[str, Any]]:
    # Generated timestamps are expected to differ and do not help diagnose drift.
    lflat = _flatten({k: v for k, v in left.items() if k != "generated_at_epoch"})
    rflat = _flatten({k: v for k, v in right.items() if k != "generated_at_epoch"})
    changes: list[dict[str, Any]] = []
    for key in sorted(set(lflat) | set(rflat)):
        before = lflat.get(key, "<missing>")
        after = rflat.get(key, "<missing>")
        if before != after:
            changes.append({"field": key, "left": before, "right": after})
    return changes


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m auto_ingest.diagnostic_diff")
    parser.add_argument("left", help="Known-good diagnostics JSON or .tar.gz")
    parser.add_argument("right", help="Failing-host diagnostics JSON or .tar.gz")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    changes = compare(load_report(args.left), load_report(args.right))
    if args.json:
        print(json.dumps(changes, indent=2, sort_keys=True, default=str))
    else:
        if not changes:
            print("No diagnostic differences.")
        for row in changes:
            print(f"{row['field']}: {row['left']!r} -> {row['right']!r}")
    return 1 if changes else 0


if __name__ == "__main__":
    raise SystemExit(main())
