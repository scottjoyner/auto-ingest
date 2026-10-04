"""Tests for the read-only NAS collection projection preparation layer."""

from __future__ import annotations

import pytest
import yaml

from auto_ingest import storage_catalog as catalog


def _record(**overrides):
    row = {
        "domain": "media",
        "description": "Test collection.",
        "lane": "L2",
        "tier": "B0",
        "durability": "D3",
        "availability": "A1",
        "authority": False,
        "logical_path": "/nas/archive/test",
        "current_path": "/nas/test",
        "future_path": "/nas/archive/test",
        "mutability": "append",
        "write_priority": "P2",
        "promotion_gate": "verified",
        "cache_eligible": True,
        "knowledge_refs": ["10-Infrastructure/example.md"],
    }
    row.update(overrides)
    return row


def _write(tmp_path, collections, *, version=1, updated="2026-10-04"):
    body = {
        "version": version,
        "updated": updated,
        "schema": {"required_fields": list(catalog.DEFAULT_REQUIRED_FIELDS)},
        "collections": collections,
    }
    path = tmp_path / "nas-collections.yaml"
    path.write_text(yaml.safe_dump(body, sort_keys=False))
    return path


def test_valid_projection_is_sorted_and_graph_safe(tmp_path):
    path = _write(
        tmp_path,
        {
            "zeta": _record(logical_path="/nas/archive/zeta", current_path="/nas/zeta"),
            "alpha": _record(
                logical_path="/nas/archive/alpha",
                current_path="/nas/alpha",
                current_beelink_path="/nas/fileserver/alpha",
            ),
        },
    )

    projection = catalog.load_projection(path)

    assert [row["collection_id"] for row in projection.rows] == ["alpha", "zeta"]
    first = projection.rows[0]
    assert first["authority"] == "false"
    assert first["cache_eligible"] is True
    assert first["knowledge_refs"] == ["10-Infrastructure/example.md"]
    assert first["current_beelink_path"] == "/nas/fileserver/alpha"
    assert len(first["content_hash"]) == 64
    assert len(projection.catalog_hash) == 64
    assert all(row["catalog_hash"] == projection.catalog_hash for row in projection.rows)


def test_semantic_hash_is_stable_across_yaml_mapping_order(tmp_path):
    first_path = _write(tmp_path, {"beta": _record(), "alpha": _record()})
    first = catalog.load_projection(first_path)

    second_path = tmp_path / "second.yaml"
    second_path.write_text(
        yaml.safe_dump(
            {
                "collections": {"alpha": _record(), "beta": _record()},
                "updated": "2026-10-04",
                "version": 1,
                "schema": {"required_fields": list(catalog.DEFAULT_REQUIRED_FIELDS)},
            },
            sort_keys=True,
        )
    )
    second = catalog.load_projection(second_path)

    assert first.catalog_hash == second.catalog_hash
    assert [row["content_hash"] for row in first.rows] == [
        row["content_hash"] for row in second.rows
    ]


def test_invalid_collection_id_fails_closed(tmp_path):
    path = _write(tmp_path, {"Bad Collection": _record()})
    with pytest.raises(catalog.StorageCatalogError, match="invalid collection id"):
        catalog.load_projection(path)


def test_missing_required_field_fails_closed(tmp_path):
    row = _record()
    row.pop("promotion_gate")
    path = _write(tmp_path, {"broken": row})
    with pytest.raises(catalog.StorageCatalogError, match="missing required field"):
        catalog.load_projection(path)


@pytest.mark.parametrize("field", ["logical_path", "current_path"])
def test_primary_paths_must_be_under_nas(tmp_path, field):
    path = _write(tmp_path, {"broken": _record(**{field: "/media/scott/NAS5/test"})})
    with pytest.raises(catalog.StorageCatalogError, match=r"/nas/\.\.\. path"):
        catalog.load_projection(path)


def test_path_escape_fails_closed(tmp_path):
    path = _write(
        tmp_path,
        {"broken": _record(logical_path="/nas/archive/../outside")},
    )
    with pytest.raises(catalog.StorageCatalogError, match="may not escape /nas"):
        catalog.load_projection(path)


def test_optional_beelink_path_is_validated(tmp_path):
    path = _write(
        tmp_path,
        {"broken": _record(current_beelink_path="/srv/nas-pool/test")},
    )
    with pytest.raises(catalog.StorageCatalogError, match="current_beelink_path"):
        catalog.load_projection(path)


def test_knowledge_refs_must_be_string_list(tmp_path):
    path = _write(tmp_path, {"broken": _record(knowledge_refs="one-file.md")})
    with pytest.raises(catalog.StorageCatalogError, match="knowledge_refs"):
        catalog.load_projection(path)


def test_env_catalog_path_is_supported(tmp_path, monkeypatch):
    path = _write(tmp_path, {"alpha": _record()})
    monkeypatch.setenv("NAS_COLLECTION_CATALOG", str(path))
    assert catalog.default_catalog_path() == path
    assert catalog.preview()[0]["collection_id"] == "alpha"


def test_cli_summary_and_rows_are_read_only(tmp_path, capsys):
    path = _write(tmp_path, {"alpha": _record()})

    assert catalog.main(["--catalog", str(path)]) == 0
    summary = capsys.readouterr().out
    assert '"collection_count": 1' in summary
    assert '"collection_id": "alpha"' in summary

    assert catalog.main(["--catalog", str(path), "--rows"]) == 0
    rows = capsys.readouterr().out
    assert '"logical_path": "/nas/archive/test"' in rows
    assert "MERGE" not in rows
