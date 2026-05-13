from __future__ import annotations

import importlib.util
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

yaml = None
if importlib.util.find_spec("yaml") is not None:
    import yaml  # type: ignore


def atomic_write(path: Path, text: str, *, backup: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if backup and path.exists():
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def write_if_missing(path: Path, text: str, *, force: bool = False) -> bool:
    if path.exists() and not force:
        return False
    atomic_write(path, text, backup=path.exists())
    return True


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    txt = read_text(path)
    if yaml:
        data = yaml.safe_load(txt)
        return data or {}
    try:
        return json.loads(txt or "{}")
    except json.JSONDecodeError:
        data: dict[str, Any] = {}
        for line in txt.splitlines():
            if not line.strip() or line.lstrip().startswith("#") or ":" not in line:
                continue
            k, v = line.split(":", 1)
            v = v.strip().strip("'\"")
            if v.lower() in {"true", "false"}:
                parsed: Any = v.lower() == "true"
            else:
                try:
                    parsed = int(v) if v.isdigit() else float(v)
                except ValueError:
                    parsed = v
            data[k.strip()] = parsed
        return data


def dump_yaml(data: Any) -> str:
    clean = json.loads(json.dumps(data))
    if yaml:
        return yaml.safe_dump(clean, sort_keys=False, allow_unicode=True)
    return json.dumps(clean, indent=2) + "\n"


def append(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(text)
