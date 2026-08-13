"""Deterministic artifact identity and provenance contracts.

Artifacts are addressed by the inputs that semantically produce them rather
than by worker-local filenames. This makes retries, dedupe, reconciliation, and
cross-host execution converge on one identity.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ArtifactIdentity:
    source_hash: str
    stage: str
    stage_version: str
    config_hash: str
    model_hash: str = "none"

    @property
    def artifact_id(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(value: Any) -> str:
    """Hash JSON-compatible configuration/model metadata deterministically."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_identity(
    *,
    source_hash: str,
    stage: str,
    stage_version: str,
    config: Mapping[str, Any] | None = None,
    model: Mapping[str, Any] | None = None,
) -> ArtifactIdentity:
    if not source_hash or not stage or not stage_version:
        raise ValueError("source_hash, stage, and stage_version are required")
    return ArtifactIdentity(
        source_hash=source_hash,
        stage=stage,
        stage_version=stage_version,
        config_hash=stable_hash(dict(config or {})),
        model_hash=stable_hash(dict(model or {})) if model else "none",
    )


def artifact_relative_path(identity: ArtifactIdentity, suffix: str = "") -> Path:
    aid = identity.artifact_id
    clean_suffix = suffix if not suffix or suffix.startswith(".") else f".{suffix}"
    return Path(identity.stage) / aid[:2] / aid[2:4] / f"{aid}{clean_suffix}"
