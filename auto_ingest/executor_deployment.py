"""Offline, fail-closed deployment inventory assessment.

The caller supplies an independently established inventory and observations.
An observation is not authentication or proof of a loaded process image.
Even a clean manifest cannot authorize activation, replay or deployment.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta

_MAX_ENTRIES = 100
_MAX_AGE = timedelta(minutes=15)
_SHA = re.compile(r"[0-9a-f]{40}")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,159}")


def _valid_id(value):
    return isinstance(value, str) and _ID.fullmatch(value) is not None


def _valid_sha(value):
    return isinstance(value, str) and _SHA.fullmatch(value) is not None


def _aware(value):
    return isinstance(value, datetime) and value.tzinfo is not None and value.utcoffset() is not None


def assess_deployments(expected, observations, *, candidate_blob, now):
    """Compare named launch paths against a pinned executor Git blob.

    expected: list of unique deployment IDs from an external inventory.
    observations: list of dictionaries with deployment_id, observed_at
    (aware datetime), source_blob (40 lowercase hex or None),
    process_state (QUIESCENT/ACTIVE/UNKNOWN), launch_state
    (DISABLED/ENABLED/UNKNOWN). Both states must be independently observed.
    Missing, duplicate, unexpected, stale and invalid evidence fails closed.
    No process, file or database access is performed.
    """
    if not _valid_sha(candidate_blob) or not _aware(now):
        raise ValueError("invalid candidate blob or observation time")
    if not isinstance(expected, (list, tuple)) or not 1 <= len(expected) <= _MAX_ENTRIES:
        raise ValueError("expected inventory must contain 1 to 100 entries")
    if any(not _valid_id(i) for i in expected) or len(set(expected)) != len(expected):
        raise ValueError("invalid or duplicate expected deployment ID")
    if not isinstance(observations, (list, tuple)) or len(observations) > _MAX_ENTRIES:
        raise ValueError("invalid observation collection")
    indexed = {}
    for row in observations:
        if not isinstance(row, dict) or not _valid_id(row.get("deployment_id")):
            raise ValueError("invalid observation identity")
        identity = row["deployment_id"]
        if identity in indexed:
            raise ValueError("duplicate observation")
        indexed[identity] = row
    findings = []
    for identity in sorted(set(expected) | set(indexed)):
        issues = []
        row = indexed.get(identity)
        if identity not in expected:
            issues.append("UNEXPECTED_DEPLOYMENT")
        if row is None:
            issues.append("MISSING_OBSERVATION")
        else:
            stamp = row.get("observed_at")
            if not _aware(stamp) or stamp > now or now - stamp > _MAX_AGE:
                issues.append("STALE_OR_INVALID_OBSERVATION")
            blob = row.get("source_blob")
            if not _valid_sha(blob):
                issues.append("SOURCE_UNVERIFIED")
            elif blob != candidate_blob:
                issues.append("SOURCE_VERSION_MISMATCH")
            process = row.get("process_state")
            if process != "QUIESCENT":
                issues.append("PROCESS_ACTIVE" if process == "ACTIVE" else "PROCESS_UNVERIFIED")
            launch = row.get("launch_state")
            if launch != "DISABLED":
                issues.append("LAUNCH_ENABLED" if launch == "ENABLED" else "LAUNCH_UNVERIFIED")
        findings.append({"deployment_id": identity, "blockers": issues})
    return {
        "inventory_status": ("BLOCKERS_OBSERVED" if any(r["blockers"] for r in findings)
                             else "DECLARED_INVENTORY_CONSISTENT"),
        "findings": findings,
        "inventory_completeness": "UNVERIFIED",
        "observation_authenticity": "UNVERIFIED",
        "activation_allowed": False,
        "replay_allowed": False,
    }
