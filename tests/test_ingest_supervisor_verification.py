"""The supervisor must not retire a day on a zero exit code alone.

`scripts/ingest_supervisor.py` is not scheduled (no cron entry, no compose
service, no systemd timer), so these tests drive its pure decision functions
rather than `main()`, which would rebuild docker images and talk to Neo4j.

The defect they lock down: the ledger previously set `status='ok'` whenever the
ingest command returned 0, and `pending_days` skips `status=='ok'` forever. A run
that exits 0 while writing nothing - missing model, credentials accepted but no
write, empty result set - was therefore retired permanently, with no alert. That
is silent data loss.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ingest_supervisor.py"


def load_supervisor():
    """Import the supervisor without triggering its module-level side effects."""
    spec = importlib.util.spec_from_file_location("ingest_supervisor_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sup():
    return load_supervisor()


@pytest.mark.parametrize(
    "rc, nodes, status, reason",
    [
        (0, 12, "ok", None),      # the only genuinely successful case
        (0, 1, "ok", None),       # exactly the minimum still counts
        (0, 0, "fail", "no_nodes_verified"),      # exited 0, ingested nothing
        (0, -1, "fail", "verify_failed"),         # exited 0, verification blew up
        (1, 12, "fail", "nonzero_rc"),
        (137, 0, "fail", "nonzero_rc"),           # OOM-killed
        (2, -1, "fail", "nonzero_rc"),            # rc takes precedence
    ],
)
def test_classify_gates_on_verified_nodes(sup, rc, nodes, status, reason):
    assert sup.classify(rc, nodes) == (status, reason)


def test_a_zero_exit_with_no_nodes_is_never_ok(sup):
    """The exact regression: rc==0 used to be sufficient for status='ok'."""
    for nodes in (0, -1):
        status, _ = sup.classify(0, nodes)
        assert status == "fail", f"nodes={nodes} must not be retired as ok"


def test_a_silent_no_op_day_stays_pending(sup):
    """End of the chain: the failure must actually be retried, not skipped."""
    days = {"2026/04/12": {"age_ok": True}}
    ledger = {"2026/04/12": {"attempts": 1, "last_rc": 0, "nodes": 0,
                             "status": "fail", "reason": "no_nodes_verified"}}
    pending = sup.pending_days(days, ledger)
    assert pending == {"2026_04": ["2026/04/12"]}


def test_a_verified_day_is_retired(sup):
    days = {"2026/04/12": {"age_ok": True}}
    ledger = {"2026/04/12": {"attempts": 1, "last_rc": 0, "nodes": 12,
                             "status": "ok"}}
    assert sup.pending_days(days, ledger) == {}


def test_recent_days_are_still_deferred(sup):
    """The age guard must survive the refactor."""
    days = {"2026/04/12": {"age_ok": False}}
    logged = []
    assert sup.pending_days(days, {}, log=logged.append) == {}
    assert logged == ["defer 2026/04/12 (modified recently)"]


def test_days_group_by_month(sup):
    days = {"2026/04/12": {"age_ok": True}, "2026/04/30": {"age_ok": True},
            "2026/05/01": {"age_ok": True}}
    assert sup.pending_days(days, {}) == {"2026_04": ["2026/04/12", "2026/04/30"],
                                          "2026_05": ["2026/05/01"]}


def test_minimum_verified_nodes_is_one(sup):
    assert sup.MIN_VERIFIED_NODES == 1


@pytest.mark.parametrize("attempts, alerted, expected", [
    (1, False, False),    # still early - the streak is not long enough yet
    (2, False, False),
    (3, False, True),     # crosses MAX_ATTEMPTS -> alert exactly once
    (4, False, True),
    (3, True, False),     # already alerted -> stay quiet
    (99, True, False),    # a permanently broken day must not re-alert for ever
])
def test_should_alert_fires_once_per_failure_streak(sup, attempts, alerted, expected):
    st = {"attempts": attempts, "status": "fail", "alerted": alerted}
    assert sup.should_alert(st) is expected


def test_a_recovered_day_can_alert_again_later(sup):
    """The 'once' flag must clear on success, or a second outage is silent."""
    st = {"attempts": 9, "status": "fail", "alerted": True}
    assert sup.should_alert(st) is False
    # main() clears both keys on the ok branch
    st["status"] = "ok"
    st.pop("reason", None)
    st.pop("alerted", None)
    st["attempts"] = 1
    st["status"] = "fail"
    assert sup.should_alert(st) is False
    st["attempts"] = sup.MAX_ATTEMPTS
    assert sup.should_alert(st) is True, "a new streak must be able to alert again"


def test_exhausted_days_are_still_retried(sup):
    """Alerts stop; retries must not. Data that lands later is data we want."""
    days = {"2026/04/12": {"age_ok": True}}
    ledger = {"2026/04/12": {"attempts": 50, "status": "fail", "alerted": True,
                             "reason": "nonzero_rc"}}
    assert sup.pending_days(days, ledger) == {"2026_04": ["2026/04/12"]}
