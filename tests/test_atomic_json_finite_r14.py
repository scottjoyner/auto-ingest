"""Atomic JSON state writes must fail closed on non-finite numbers (r14).

``write_json_atomic`` promises durable, valid JSON for the small state files the
scripts keep beside themselves.  ``json.dump`` defaults to ``allow_nan=True``,
which emits the non-standard tokens ``NaN`` / ``Infinity`` / ``-Infinity``.
Those tokens are rejected by strict JSON readers (jq, JavaScript, Go, Rust
serde_json), so a state file written by this helper can be unreadable by any
consumer other than Python's permissive decoder.

These tests use only ``tmp_path`` under pytest's basetemp: no Neo4j, network,
model, credential or live state file is touched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from auto_ingest.util.atomic import write_json_atomic


def _strict_loads(raw: str) -> object:
    """Parse with a strict decoder that rejects NaN/Infinity constants."""
    def _reject(token: str) -> None:
        raise ValueError(f"non-standard JSON constant: {token}")

    return json.loads(raw, parse_constant=_reject)


def _stranded_temps(target: Path) -> list[Path]:
    return sorted(target.parent.glob(f"{target.name}.*.tmp"))


@pytest.mark.parametrize(
    "bad_value",
    [float("nan"), float("inf"), float("-inf")],
    ids=["nan", "pos_inf", "neg_inf"],
)
def test_non_finite_write_is_rejected_and_target_is_unchanged(
    tmp_path: Path, bad_value: float
) -> None:
    """The regression: a non-finite value used to overwrite valid state."""
    target = tmp_path / "state.json"
    write_json_atomic(target, {"ok": True, "n": 3})
    before = target.read_text(encoding="utf-8")

    with pytest.raises(ValueError):
        write_json_atomic(target, {"v": bad_value})

    assert target.read_text(encoding="utf-8") == before
    assert _strict_loads(target.read_text(encoding="utf-8")) == {"ok": True, "n": 3}
    assert _stranded_temps(target) == []


def test_nested_non_finite_value_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "state.json"
    write_json_atomic(target, {"ok": True})

    with pytest.raises(ValueError):
        write_json_atomic(target, {"a": {"b": [1.0, float("inf")]}})

    assert _strict_loads(target.read_text(encoding="utf-8")) == {"ok": True}
    assert _stranded_temps(target) == []


def test_finite_payload_still_written_as_valid_json(tmp_path: Path) -> None:
    """Control: ordinary state writes are unchanged."""
    target = tmp_path / "state.json"
    write_json_atomic(target, {"ok": True, "n": 3}, indent=2)
    assert _strict_loads(target.read_text(encoding="utf-8")) == {"ok": True, "n": 3}
    assert _stranded_temps(target) == []


def test_unserializable_value_is_still_stringified(tmp_path: Path) -> None:
    """Control: the documented ``default=str`` behaviour is preserved."""
    target = tmp_path / "state.json"
    write_json_atomic(target, {"p": Path("/srv/example")})
    assert _strict_loads(target.read_text(encoding="utf-8")) == {"p": "/srv/example"}
