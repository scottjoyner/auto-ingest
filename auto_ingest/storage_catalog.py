"""Knowledge-backed NAS collection catalog.

This module is intentionally a *projection preparation* layer, not a storage
scanner and not a Neo4j writer.

The knowledge repo owns the human-reviewable catalog. This code:

1. loads ``60-Mappings/nas-collections.yaml``;
2. validates the stable collection contract;
3. normalizes values into Neo4j-safe scalar/list properties;
4. emits deterministic rows suitable for a future bounded
   ``MERGE (:StorageCollection {collection_id: ...})`` projection.

It never walks /nas and never touches Neo4j. That keeps discovery metadata
separate from byte custody and makes the first integration step read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

COLLECTION_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")

DEFAULT_REQUIRED_FIELDS = (
    "domain",
    "description",
    "lane",
    "tier",
    "durability",
    "availability",
    "authority",
    "logical_path",
    "current_path",
    "future_path",
    "mutability",
    "write_priority",
    "promotion_gate",
    "knowledge_refs",
)

PROJECTED_FIELDS = (
    "domain",
    "description",
    "lane",
    "tier",
    "durability",
    "availability",
    "authority",
    "logical_path",
    "current_path",
    "current_beelink_path",
    "future_path",
    "mutability",
    "write_priority",
    "promotion_gate",
    "cache_eligible",
    "knowledge_refs",
)


class StorageCatalogError(ValueError):
    """Catalog input violates the stable collection contract."""


@dataclass(frozen=True)
class StorageCatalogProjection:
    """Validated, deterministic projection input."""

    source: Path
    version: int | str | None
    updated: str | None
    catalog_hash: str
    rows: tuple[dict[str, Any], ...]


def default_catalog_path() -> Path:
    """Return the configured or conventional knowledge-repo catalog path."""

    configured = os.environ.get("NAS_COLLECTION_CATALOG")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / "knowledge" / "60-Mappings" / "nas-collections.yaml"


def _semantic_hash(value: Any) -> str:
    """Hash parsed data, not YAML formatting/comments."""

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _stringify_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value)


def _normalize_knowledge_refs(value: Any, collection_id: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
        raise StorageCatalogError(
            f"{collection_id}: knowledge_refs must be a list of non-empty strings"
        )
    return list(value)


def _required_fields(document: Mapping[str, Any]) -> tuple[str, ...]:
    schema = document.get("schema")
    if isinstance(schema, Mapping):
        required = schema.get("required_fields")
        if isinstance(required, list) and all(isinstance(v, str) for v in required):
            return tuple(required)
    return DEFAULT_REQUIRED_FIELDS


def _validate_nas_path(value: Any, *, collection_id: str, field: str) -> str:
    if not isinstance(value, str) or not value.startswith("/nas/"):
        raise StorageCatalogError(
            f"{collection_id}: {field} must be an absolute canonical /nas/... path"
        )
    if "/../" in value or value.endswith("/.."):
        raise StorageCatalogError(f"{collection_id}: {field} may not escape /nas")
    return value.rstrip("/")


def load_projection(path: str | Path | None = None) -> StorageCatalogProjection:
    """Load, validate, and normalize a storage catalog without external I/O."""

    source = Path(path).expanduser() if path else default_catalog_path()
    if not source.is_file():
        raise StorageCatalogError(f"storage catalog not found: {source}")

    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping):
        raise StorageCatalogError("storage catalog root must be a mapping")

    collections = raw.get("collections")
    if not isinstance(collections, Mapping):
        raise StorageCatalogError("storage catalog must contain a collections mapping")

    required = _required_fields(raw)
    rows: list[dict[str, Any]] = []

    for collection_id in sorted(collections):
        if not isinstance(collection_id, str) or not COLLECTION_ID_RE.fullmatch(collection_id):
            raise StorageCatalogError(f"invalid collection id: {collection_id!r}")

        record = collections[collection_id]
        if not isinstance(record, Mapping):
            raise StorageCatalogError(f"{collection_id}: collection body must be a mapping")

        missing = [field for field in required if field not in record]
        if missing:
            raise StorageCatalogError(
                f"{collection_id}: missing required field(s): {', '.join(sorted(missing))}"
            )

        logical_path = _validate_nas_path(
            record.get("logical_path"),
            collection_id=collection_id,
            field="logical_path",
        )
        current_path = _validate_nas_path(
            record.get("current_path"),
            collection_id=collection_id,
            field="current_path",
        )

        beelink = record.get("current_beelink_path")
        if beelink is not None:
            beelink = _validate_nas_path(
                beelink,
                collection_id=collection_id,
                field="current_beelink_path",
            )

        refs = _normalize_knowledge_refs(record.get("knowledge_refs"), collection_id)

        row: dict[str, Any] = {
            "collection_id": collection_id,
            "logical_path": logical_path,
            "current_path": current_path,
            "knowledge_refs": refs,
        }

        for field in PROJECTED_FIELDS:
            if field in {"logical_path", "current_path", "knowledge_refs"}:
                continue
            value = record.get(field)
            if value is None:
                continue
            if field == "current_beelink_path":
                row[field] = beelink
            elif field == "cache_eligible":
                row[field] = bool(value)
            else:
                row[field] = _stringify_scalar(value)

        row["content_hash"] = _semantic_hash(
            {key: row[key] for key in sorted(row) if key != "content_hash"}
        )
        rows.append(row)

    catalog_semantics = {
        "version": raw.get("version"),
        "updated": raw.get("updated"),
        "collections": {row["collection_id"]: row["content_hash"] for row in rows},
    }
    catalog_hash = _semantic_hash(catalog_semantics)

    for row in rows:
        row["catalog_hash"] = catalog_hash
        if raw.get("version") is not None:
            row["catalog_version"] = _stringify_scalar(raw.get("version"))
        if raw.get("updated") is not None:
            row["catalog_updated"] = _stringify_scalar(raw.get("updated"))

    return StorageCatalogProjection(
        source=source,
        version=raw.get("version"),
        updated=_stringify_scalar(raw.get("updated")) if raw.get("updated") is not None else None,
        catalog_hash=catalog_hash,
        rows=tuple(rows),
    )


def preview(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Return projection rows for callers/MCP adapters. No graph access."""

    return [dict(row) for row in load_projection(path).rows]


def _summary(projection: StorageCatalogProjection) -> dict[str, Any]:
    return {
        "source": str(projection.source),
        "version": projection.version,
        "updated": projection.updated,
        "catalog_hash": projection.catalog_hash,
        "collection_count": len(projection.rows),
        "collections": [
            {
                "collection_id": row["collection_id"],
                "logical_path": row["logical_path"],
                "current_path": row["current_path"],
                "lane": row.get("lane"),
                "durability": row.get("durability"),
                "authority": row.get("authority"),
                "content_hash": row["content_hash"],
            }
            for row in projection.rows
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate/preview the NAS collection catalog.")
    parser.add_argument("--catalog", type=Path, default=None)
    parser.add_argument(
        "--rows",
        action="store_true",
        help="emit normalized StorageCollection projection rows instead of a compact summary",
    )
    args = parser.parse_args(argv)

    try:
        projection = load_projection(args.catalog)
    except StorageCatalogError as exc:
        parser.error(str(exc))

    payload: Any = list(projection.rows) if args.rows else _summary(projection)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
