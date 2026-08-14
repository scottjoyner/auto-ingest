"""Sanitized diagnostics bundle for reproducing machine-specific ingest bugs.

The bundle intentionally records capability and configuration *presence* rather
than secret values. It is safe to attach to a bug report after normal review and
contains enough environment information to compare CI with a production host.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

SENSITIVE_ENV_KEYS = (
    "NEO4J_PASSWORD",
    "NEXTCLOUD_PASSWORD",
    "NEXTCLOUD_TOKEN",
    "CONTENT_OS_LLM_API_KEY",
    "OPENAI_API_KEY",
    "FLEET_AUTH_PASS",
)
CONFIG_ENV_KEYS = (
    "NEO4J_URI",
    "NEO4J_USER",
    "NEO4J_DB",
    "FILESERVER_ROOT",
    "HOT_STORAGE_ROOT",
    "COLD_STORAGE_ROOT",
    "AUDIO_ROOT",
    "DASHCAM_ROOT",
    "BODYCAM_ROOT",
    "TRANSCRIPT_ROOT",
    "DROP_ROOT",
    "STATE_ROOT",
)
TOOLS = ("python3", "ffmpeg", "ffprobe", "docker", "git", "nvidia-smi", "rocminfo")
PACKAGES = (
    "numpy",
    "pandas",
    "neo4j",
    "pydantic",
    "PyYAML",
    "opencv-python-headless",
    "moviepy",
    "torch",
    "torchaudio",
    "transformers",
    "sentence-transformers",
    "ultralytics",
)


def _safe_command(command: Sequence[str], timeout_sec: int = 5) -> dict[str, Any]:
    executable = shutil.which(command[0])
    if not executable:
        return {"available": False}
    try:
        result = subprocess.run(
            [executable, *command[1:]],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except Exception as exc:
        return {"available": True, "error": f"{type(exc).__name__}: {exc}"}
    text = (result.stdout or result.stderr or "").strip().splitlines()
    return {
        "available": True,
        "returncode": result.returncode,
        "summary": text[0][:500] if text else "",
    }


def environment_presence() -> dict[str, Any]:
    return {
        "configured": {key: bool(os.environ.get(key)) for key in CONFIG_ENV_KEYS},
        "sensitive_present": {key: bool(os.environ.get(key)) for key in SENSITIVE_ENV_KEYS},
    }


def package_versions(packages: Iterable[str] = PACKAGES) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in packages:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def tool_status() -> dict[str, Any]:
    commands = {
        "python3": ("python3", "--version"),
        "ffmpeg": ("ffmpeg", "-version"),
        "ffprobe": ("ffprobe", "-version"),
        "docker": ("docker", "--version"),
        "git": ("git", "--version"),
        "nvidia-smi": ("nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"),
        "rocminfo": ("rocminfo",),
    }
    return {name: _safe_command(command) for name, command in commands.items()}


def storage_status(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in paths:
        path = Path(raw)
        row: dict[str, Any] = {"path": str(path), "exists": path.exists()}
        if path.exists():
            try:
                usage = shutil.disk_usage(path)
                row.update(
                    free_gb=round(usage.free / (1024**3), 2),
                    total_gb=round(usage.total / (1024**3), 2),
                    writable=os.access(path, os.W_OK),
                )
            except OSError as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return rows


def _configured_storage_paths() -> list[str]:
    seen: set[str] = set()
    paths: list[str] = []
    for key in CONFIG_ENV_KEYS:
        if not key.endswith("_ROOT"):
            continue
        value = os.environ.get(key)
        if value and value not in seen:
            seen.add(value)
            paths.append(value)
    return paths


def neo4j_status() -> dict[str, Any]:
    uri = os.environ.get("NEO4J_URI")
    user = os.environ.get("NEO4J_USER")
    secret = os.environ.get("NEO4J_PASSWORD")
    if not (uri and user and secret):
        return {"configured": False}
    try:
        from neo4j import GraphDatabase
        from auto_ingest.runtime_schema import audit_schema

        driver = GraphDatabase.driver(uri, auth=(user, secret))
        try:
            driver.verify_connectivity()
            schema = audit_schema(driver)
        finally:
            driver.close()
        return {
            "configured": True,
            "reachable": True,
            "schema_ok": bool(schema.get("ok")),
            "missing_constraints": list(schema.get("missing_constraints", [])),
            "duplicate_contracts": sorted(
                name for name, rows in schema.get("duplicates", {}).items() if rows
            ),
        }
    except Exception as exc:
        return {
            "configured": True,
            "reachable": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def collect() -> dict[str, Any]:
    return {
        "generated_at_epoch": int(time.time()),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
            "timezone": time.tzname[0] if time.tzname else "",
        },
        "environment": environment_presence(),
        "packages": package_versions(),
        "tools": tool_status(),
        "storage": storage_status(_configured_storage_paths()),
        "neo4j": neo4j_status(),
    }


def write_bundle(output: str | Path) -> Path:
    target = Path(output)
    if not target.name.endswith(".tar.gz"):
        target = target.with_suffix(target.suffix + ".tar.gz" if target.suffix else ".tar.gz")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="auto-ingest-diagnostics-") as temp_dir:
        root = Path(temp_dir) / "auto-ingest-diagnostics"
        root.mkdir()
        (root / "diagnostics.json").write_text(
            json.dumps(collect(), indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        with tarfile.open(target, "w:gz") as archive:
            archive.add(root, arcname=root.name)
    return target


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m auto_ingest.diagnostics")
    parser.add_argument("--output", default="auto-ingest-diagnostics.tar.gz")
    parser.add_argument("--json", action="store_true", help="Print sanitized JSON instead of an archive")
    args = parser.parse_args(argv)
    if args.json:
        print(json.dumps(collect(), indent=2, sort_keys=True, default=str))
        return 0
    print(write_bundle(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
