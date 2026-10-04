"""auto_ingest.custody.report - deterministic rendering of status and plan.

Two modes, one data source. ``--json`` emits sorted-key JSON so two runs against
unchanged evidence are byte-identical (which is what makes the commands usable
as a CI/Hermes gate). The human mode is a fixed-width rendering of exactly the
same fields - no extra lookups, no clock, no ordering by dict iteration.
"""

from __future__ import annotations

import json
from typing import List

from .planner import ResumePlan
from .store import CampaignStatus


def status_json(status: CampaignStatus, *, indent: int | None = 2) -> str:
    """Deterministic JSON for ``custody status``."""
    return json.dumps(status.to_dict(), sort_keys=True, indent=indent, default=str)


def plan_json(plan: ResumePlan, *, indent: int | None = 2) -> str:
    """Deterministic JSON for ``custody plan``."""
    return json.dumps(plan.to_dict(), sort_keys=True, indent=indent, default=str)


def status_text(status: CampaignStatus) -> str:
    """Human-readable custody status. Same facts, fixed order."""
    data = status.to_dict()
    campaign = status.campaign
    lines: List[str] = []
    lines.append(f"campaign_id        {campaign.campaign_id}")
    lines.append(f"card_id            {campaign.card_id}")
    card = data["source"]
    lines.append(
        "source             "
        f"device={card['device'] or '?'} uuid={card['filesystem_uuid'] or '?'} "
        f"label={card['label'] or '?'} mount={card['mount_point'] or '?'}"
    )
    lines.append(f"source_read_only   {str(card['read_only']).lower()}")
    dest = data["destination"]
    lines.append(
        "destination        "
        f"logical={dest['canonical']} host_path={dest['host_path'] or 'UNRESOLVED'} "
        f"from={dest['resolved_from']} mounted={dest['mounted']}"
    )
    identity = dest.get("identity") or {}
    lines.append(
        "destination_id     "
        f"uuid={identity.get('filesystem_uuid') or 'UNPROVEN'} "
        f"device={identity.get('device') or '-'}"
    )
    inv = data["inventory"]
    lines.append(
        "inventory          "
        f"files={inv['discovered_files']} bytes={inv['discovered_bytes']} "
        f"complete={str(inv['complete']).lower()}"
    )
    hsh = data["hash"]
    lines.append(
        "hash               "
        f"verified={hsh['verified']} errors={hsh['errors']} "
        f"complete={str(hsh['complete']).lower()} algorithm={hsh['algorithm'] or '?'}"
    )
    cpy = data["copy"]
    lines.append(
        "copy               "
        f"planned={cpy['planned_files']} completed={cpy['completed_files']} "
        f"complete={str(cpy['complete']).lower()} "
        f"interrupted={str(cpy['interrupted']).lower()}"
    )
    lines.append(
        "destination        "
        f"verified_files={dest['verified_files']} verified_bytes={dest['verified_bytes']} "
        f"verification_complete={str(dest['verification_complete']).lower()}"
    )
    rec = data["reconciliation"]
    lines.append(
        "reconciliation     "
        f"source_only={rec['source_only']} destination_only={rec['destination_only']} "
        f"mismatched={rec['mismatched']}"
    )
    worker = data["worker"]
    lines.append(
        "worker             "
        f"identity={worker['identity'] or 'none'} status={worker['status']}"
    )
    errs = data["errors"]
    lines.append(
        "errors             "
        f"unresolved={errs['unresolved']} fatal={errs['fatal']}"
    )
    lines.append(f"state              {data['state']}")
    lines.append(f"next_phase         {data['next_phase']}")
    lines.append(f"next_safe_action   {data['next_safe_action']}")
    lines.append(f"source_release_allowed     {str(data['source_release_allowed']).lower()}")
    lines.append(f"source_mutation_allowed    {str(data['source_mutation_allowed']).lower()}")
    lines.append(f"source_deletion_allowed    {str(data['source_deletion_allowed']).lower()}")
    if data["reasons"]:
        lines.append("reasons            " + "; ".join(data["reasons"]))
    if data.get("coerced_fields"):
        # A diagnostic that only appears in --json is invisible to an operator
        # running the default human mode, which is exactly the fault they need
        # to see: their own count was not a number.
        lines.append(
            "  MALFORMED COUNT  " + ", ".join(data["coerced_fields"])
            + "  (not numbers; fixed to 0 - fix the source evidence)"
        )
    if data.get("ignored_declared_fields"):
        lines.append(
            "  IGNORED          declared state keys dropped: "
            + ", ".join(data["ignored_declared_fields"])
        )
    conflict = data.get("observed_card_conflict")
    if conflict:
        lines.append(
            f"  CARD MISMATCH   {conflict['kind']}: "
            + ",".join(conflict["conflicting_fields"])
        )
        lines.append(f"      {conflict['remedy']}")
    for blocker in data["blockers"]:
        lines.append(
            f"  BLOCKED          {blocker['code']}: {blocker['detail']} -> {blocker['remedy']}"
        )
    for warning in data["warnings"]:
        lines.append(f"  warning          {warning}")
    for disagreement in data["ledger_disagreements"]:
        lines.append(f"  ledger drift     {disagreement}")
    for name, summary in sorted(status.ledgers.items()):
        lines.append(
            f"  ledger {name:<18} present={str(summary.present).lower()} "
            f"records={summary.records} verified_files={summary.files} "
            f"coherent={str(summary.coherent).lower()}"
        )
    return "\n".join(lines) + "\n"


def plan_text(plan: ResumePlan) -> str:
    """Human-readable resume plan. Describes; never executes."""
    lines: List[str] = []
    lines.append(f"campaign_id              {plan.campaign_id}")
    lines.append(f"current_state            {plan.current_state.value}")
    lines.append(f"next_phase               {plan.next_phase}")
    lines.append(f"next_safe_action         {plan.next_safe_action}")
    lines.append(f"safe_to_resume           {str(plan.safe_to_resume).lower()}")
    lines.append(
        f"source_mutation_allowed  {str(plan.source_mutation_allowed).lower()}"
    )
    lines.append(
        f"source_deletion_allowed  {str(plan.source_deletion_allowed).lower()}"
    )
    lines.append(
        f"operator_authorization   {str(plan.requires_operator_authorization).lower()}"
    )
    lines.append(f"plan_fingerprint         {plan.plan_fingerprint}")
    if plan.reasons:
        lines.append("reasons                  " + "; ".join(plan.reasons))
    if not plan.actions:
        lines.append("actions                  (none - nothing outstanding)")
    for action in plan.actions:
        gated = f" after={','.join(action.gated_by)}" if action.gated_by else ""
        lines.append(
            f"  {action.action_id}  {action.operation} -> {action.target} "
            f"files={action.estimated_files} bytes={action.estimated_bytes} "
            f"excludes_verified={str(action.excludes_verified).lower()} "
            f"mutates_destination={str(action.mutates_destination).lower()}"
            f"{gated}"
        )
        if action.rationale:
            lines.append(f"      rationale: {action.rationale}")
    for blocker in plan.blockers:
        lines.append(f"  BLOCKED        {blocker.code}: {blocker.detail} -> {blocker.remedy}")
    for warning in plan.warnings:
        lines.append(f"  warning        {warning}")
    return "\n".join(lines) + "\n"


__all__ = ["plan_json", "plan_text", "status_json", "status_text"]
