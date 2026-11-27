"""A fleet batch that created no tasks must not report success (r13).

``run_batch`` used to return 0 unconditionally after the loop, even when every
``create_task`` call raised.  A caller (cron, operator script, CI) therefore saw
a successful run while nothing was dispatched.  These tests stub ``create_task``
and use an inert Namespace -- no HTTP, credentials, filesystem writes or fleet
contact.
"""

from __future__ import annotations

from argparse import Namespace

import pytest

from auto_ingest import fleet_batch


def _args(items: list[str]) -> Namespace:
    return Namespace(
        assistx_url="http://example.invalid",
        auth_user="test-user",
        auth_pass="test-pass",
        items=items,
        input=None,
        glob=None,
        capabilities="script",
        command=None,
        yolo_command=None,
        title_prefix="batch",
        batch_id=None,
        priority="background",
    )


def _stub(monkeypatch: pytest.MonkeyPatch, outcomes: list[bool]) -> list[str]:
    calls: list[str] = []

    def fake_create_task(*a, **kw):
        item = kw.get("payload", {}).get("item", "")
        calls.append(item)
        if not outcomes[len(calls) - 1]:
            raise RuntimeError("stubbed task-creation failure")
        return {"task_id": f"stub-{len(calls)}"}

    monkeypatch.setattr(fleet_batch, "create_task", fake_create_task)
    return calls


def test_all_task_creations_failed_is_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    """The regression: pre-fix this returned 0 with zero tasks created."""
    calls = _stub(monkeypatch, [False, False])
    assert fleet_batch.run_batch(_args(["a", "b"])) != 0
    assert calls == ["a", "b"]


def test_partial_failure_is_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    """A partially dispatched batch must not look fully successful."""
    _stub(monkeypatch, [True, False, True])
    assert fleet_batch.run_batch(_args(["a", "b", "c"])) != 0


def test_all_task_creations_succeeded_is_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Control: the happy path keeps its existing exit status."""
    _stub(monkeypatch, [True, True])
    assert fleet_batch.run_batch(_args(["a", "b"])) == 0


def test_no_items_keeps_existing_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Control: the pre-existing 'nothing to do' status is unchanged."""
    _stub(monkeypatch, [])
    assert fleet_batch.run_batch(_args([])) == 1
