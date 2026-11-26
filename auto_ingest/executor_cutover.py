"""Read-only cutover blocker probes; an empty observation is not approval.

Requires independent database admission. Never creates indexes or alters
tasks, runs, writers or deployment state. Legacy writer quiescence cannot
be proved by graph observations alone.
"""
from __future__ import annotations

MAX_SAMPLE = 50
PROBES = {
    "RUNNING_TASK": """MATCH (t:Task) USING INDEX t:Task(status)
WHERE t.status = 'RUNNING' RETURN elementId(t) AS identity LIMIT $limit""",
    "ACTIVE_OWNER": """MATCH (t:Task) USING INDEX t:Task(active_run_id)
WHERE t.active_run_id IS NOT NULL RETURN elementId(t) AS identity LIMIT $limit""",
    "RUNNING_RUN": """MATCH (r:Run) USING INDEX r:Run(status)
WHERE r.status = 'RUNNING' RETURN elementId(r) AS identity LIMIT $limit""",
}
_INDEX = {"NodeIndexSeek", "NodeUniqueIndexSeek", "NodeIndexScan"}
_SAFE = _INDEX | {"ProduceResults", "Limit", "Projection", "Filter", "CacheProperties"}


def _index_only(plan):
    if not isinstance(plan, dict):
        return False
    pending = [plan]
    found = False
    while pending:
        node = pending.pop()
        operator = node.get("operatorType", "").split("@")[0]
        if operator not in _SAFE:
            return False
        found |= operator in _INDEX
        pending.extend(node.get("children", []))
    return found


def inspect_cutover(driver, *, sample_limit=20):
    """Bounded observational report; never authorizes activation or replay.

    All EXPLAIN plans must use only allowed index-based operators before
    executing any data probe. Missing indexes, unsupported plans and errors
    fail closed. The timeout bounds the read transaction; this is not a
    consistent snapshot under concurrent writes or a writer inventory.
    """
    if type(sample_limit) is not int or not 1 <= sample_limit <= MAX_SAMPLE:
        raise ValueError("sample_limit must be 1 to 50")
    findings = {}
    with driver.session(default_access_mode="READ") as session:
        with session.begin_transaction(timeout=5) as tx:
            for name, query in PROBES.items():
                summary = tx.run("EXPLAIN " + query, limit=sample_limit+1).consume()
                if not _index_only(summary.plan):
                    raise RuntimeError(f"index-only plan not verified: {name}")
            for name, query in PROBES.items():
                result = tx.run(query, limit=sample_limit+1)
                rows = list(result)
                result.consume()
                if len(rows) > sample_limit+1:
                    raise RuntimeError("unexpected probe cardinality")
                ids = [row["identity"] for row in rows]
                if any(not isinstance(i, str) or not i for i in ids):
                    raise RuntimeError("invalid graph identity")
                if len(set(ids)) != len(ids):
                    raise RuntimeError("duplicate probe identity")
                findings[name] = {"observed": min(len(ids), sample_limit),
                                  "truncated": len(ids) > sample_limit}
            tx.commit()
    blocked = any(item["observed"] for item in findings.values())
    return {
        "graph_status": "BLOCKERS_OBSERVED" if blocked else "NO_BLOCKERS_OBSERVED",
        "findings": findings,
        "legacy_writer_quiescence": "UNVERIFIED",
        "activation_allowed": False,
        "replay_allowed": False,
    }
