"""Read-only, exact-ID executor reconciliation; never authorizes replay.

A RUNNING task can outlive its worker, or its external operation may have
completed before a lost acknowledgement. Age is a review hint, not liveness
or proof of an orphan. No broad graph scan, writes, or automatic recovery.
Before live use, verify the Task(id) lookup plan and database admission.
"""
from __future__ import annotations

from datetime import datetime, timedelta

MAX_TASKS = 50

# A cap of three rows is enough to detect any ambiguous identity or link.
# A matching RUNNING historical run is retained to expose inconsistent owners.
_QUERY = """
MATCH (t:Task {id:$task_id})
OPTIONAL MATCH (t)-[:HAS_RUN]->(r:Run)
WHERE r.id = t.active_run_id OR r.status = 'RUNNING'
RETURN elementId(t) AS identity, t.id AS task_id, t.status AS status,
       t.active_run_id AS owner, toString(t.updated_at) AS updated_at,
       r.id AS run_id, r.status AS run_status
LIMIT 3
"""


def _classify(rows, task_id, now, stale_after):
    if not rows:
        return "TASK_NOT_FOUND"
    if any(row["task_id"] != task_id for row in rows):
        return "IDENTITY_MISMATCH"
    if len(rows) != 1:
        return "AMBIGUOUS_TASK_OR_RUN"
    row = rows[0]
    if not row["identity"]:
        return "IDENTITY_MISMATCH"
    status, owner = row["status"], row["owner"]
    run_id, run_status = row["run_id"], row["run_status"]
    if status != "RUNNING":
        if owner is not None or run_id is not None or run_status is not None:
            return "INCONSISTENT_NONRUNNING"
        if status not in {"READY", "REVIEW", "DONE", "FAILED"}:
            return "UNKNOWN_TASK_STATE"
        return "NOT_RUNNING"
    if not isinstance(owner, str) or not owner.strip():
        return "UNOWNED_RUNNING_REVIEW"
    if run_id != owner or run_status != "RUNNING":
        return "OWNER_RUN_MISMATCH_REVIEW"
    try:
        stamp = datetime.fromisoformat(row["updated_at"].replace("Z", "+00:00"))
        if stamp.tzinfo is None or stamp.utcoffset() is None or stamp > now:
            return "CLOCK_OR_TIMESTAMP_UNKNOWN"
    except (AttributeError, TypeError, ValueError, OverflowError):
        return "CLOCK_OR_TIMESTAMP_UNKNOWN"
    if now - stamp >= stale_after:
        return "STALE_RUNNING_REVIEW"
    return "RUNNING_OBSERVED"


def inspect_tasks(driver, task_ids, *, now: datetime, stale_after: timedelta):
    """Inspect up to 50 caller-supplied exact task IDs, with no graph writes.

    This is an observational report, not an atomic snapshot across IDs.
    Driver errors propagate; no partial report is returned as a success.
    A stale or inconsistent result requires independent worker and side-effect
    evidence before any separately authorized recovery. Never call an executor.
    """
    if not isinstance(task_ids, (list, tuple)) or not 1 <= len(task_ids) <= MAX_TASKS:
        raise ValueError("provide 1 to 50 exact task IDs")
    if any(not isinstance(i, str) or not i.strip() or len(i) > 256 for i in task_ids):
        raise ValueError("invalid task ID")
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("duplicate task IDs")
    if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    if not isinstance(stale_after, timedelta) or stale_after <= timedelta(0):
        raise ValueError("stale_after must be positive")
    report = []
    with driver.session(default_access_mode="READ") as session:
        for task_id in task_ids:
            result = session.run(_QUERY, task_id=task_id)
            rows = list(result)
            result.consume()
            if len(rows) > 3:
                raise RuntimeError("unexpected query cardinality")
            report.append({
                "task_id": task_id,
                "classification": _classify(rows, task_id, now, stale_after),
                "replay_allowed": False,
            })
    return report
