"""Round-7 hermetic regression: intermediate artifact key identity.

``ingest_write`` encodes ``/`` in a media key as ``__`` in the on-disk artifact
name, and ``list_intermediates`` decodes ``__`` back to ``/``.  That mapping is
not injective: a *literal* ``__`` in a key maps to the same filename as a
slash-bearing key.  Without a guard, staging one silently overwrites the
other's cleaned stage output (data loss), and discovery reports the wrong key.

These tests use only pytest ``tmp_path``; no Neo4j, live DB, network or media.
"""

from __future__ import annotations

import pytest

from auto_ingest import ingest_write

SLASH_KEY = "dashcam/2026-07-14/CLT_1423_F"
DUNDER_KEY = "dashcam__2026-07-14__CLT_1423_F"


def _write(d, key, marker):
    return ingest_write.write_intermediate(d, key, "metadata", [{"id": marker}])


def test_literal_dunder_key_is_rejected_fail_closed(tmp_path):
    """A key containing the reserved '__' separator must never be staged."""
    with pytest.raises(ValueError, match="reserved"):
        _write(str(tmp_path), DUNDER_KEY, "SHOULD_NOT_LAND")

    assert list(tmp_path.iterdir()) == []


def test_collision_cannot_overwrite_slash_key_artifact(tmp_path):
    """The pre-fix defect: writing the ambiguous key clobbered the slash key."""
    d = str(tmp_path)
    _write(d, SLASH_KEY, "FROM_SLASH_KEY")

    with pytest.raises(ValueError, match="reserved"):
        _write(d, DUNDER_KEY, "FROM_LITERAL_DUNDER_KEY")

    # The genuine artifact survives unchanged.
    got = ingest_write.read_intermediate(d, SLASH_KEY, "metadata")
    assert got == [{"id": "FROM_SLASH_KEY"}]
    assert [m["key"] for m in ingest_write.list_intermediates(d)] == [SLASH_KEY]


def test_read_rejects_literal_dunder_key(tmp_path):
    """Read must fail closed rather than return another key's rows."""
    d = str(tmp_path)
    _write(d, SLASH_KEY, "FROM_SLASH_KEY")

    with pytest.raises(ValueError, match="reserved"):
        ingest_write.read_intermediate(d, DUNDER_KEY, "metadata")


def test_list_intermediates_rejects_ambiguous_key_filter(tmp_path):
    """A dunder key filter can never match; reject it instead of silent []."""
    d = str(tmp_path)
    _write(d, SLASH_KEY, "FROM_SLASH_KEY")

    with pytest.raises(ValueError, match="reserved"):
        ingest_write.list_intermediates(d, key=DUNDER_KEY)


def test_slash_key_round_trip_still_works(tmp_path):
    """Guard must not break the legitimate slash-bearing key encoding."""
    d = str(tmp_path)
    path = _write(d, SLASH_KEY, "ROUND_TRIP")

    assert path.endswith("dashcam__2026-07-14__CLT_1423_F.metadata.json")
    assert ingest_write.read_intermediate(d, SLASH_KEY, "metadata") == [{"id": "ROUND_TRIP"}]
    found = ingest_write.list_intermediates(d, key=SLASH_KEY)
    assert len(found) == 1
    assert found[0]["key"] == SLASH_KEY
    assert found[0]["stage"] == "metadata"


def test_single_underscore_keys_remain_allowed(tmp_path):
    """Only the reserved '__' sequence is ambiguous; a single '_' is fine."""
    d = str(tmp_path)
    key = "dashcam_CLT_1423_F"
    _write(d, key, "SINGLE_UNDERSCORE")

    assert ingest_write.read_intermediate(d, key, "metadata") == [{"id": "SINGLE_UNDERSCORE"}]
    assert [m["key"] for m in ingest_write.list_intermediates(d)] == [key]
