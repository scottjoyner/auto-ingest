"""auto_ingest.custody.cli - read-only custody CLI.

    auto-ingest custody status  --bundle PATH [--json]
    auto-ingest custody plan    --bundle PATH [--json]
    auto-ingest custody verify  --bundle PATH [--json] [--execute]
    auto-ingest custody import  --bundle PATH --evidence FILE [--apply]
    auto-ingest custody new     --bundle PATH --card-id ID --uuid/--device/--label ... [--apply]

Every command is read-only by default. The only writes in this package are
``import --apply`` and ``new --apply``, both explicit. ``--execute`` is accepted
only by ``verify`` and refuses, because destination verification is performed by
an operator-authorized executor outside this package; the command here only
*describes* the verification set.

``status``, ``plan`` and ``verify`` never take a clock reading: their output is a
pure function of the bundle. ``new`` does take one (a campaign needs a creation
instant), which is why ``--created-at`` exists and why its output is a write.

Exit codes:
    0  command completed
    2  usage / unreadable bundle
    3  gate closed (``--require-release`` only), ``--execute`` refused, or a
       refusal (campaign exists, identity unprovable)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from .campaign import CardIdentity
from .policy import CustodyPolicy
from .report import plan_json, plan_text, status_json, status_text
from .store import (
    BundleError,
    CampaignCreationError,
    CampaignStatus,
    import_evidence,
    load_custody_config,
    load_policy,
    load_status,
    new_campaign,
)

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_GATE_CLOSED = 3

EXECUTE_REFUSED = (
    "refusing to execute: destination verification is performed by an "
    "operator-authorized executor outside auto_ingest.custody. This command "
    "describes the work only. Re-run without --execute."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="auto-ingest custody",
        description=(
            "SD-card ingest custody state machine. Read-only by default: "
            "status and plan never copy, delete or mutate anything."
        ),
    )
    sub = parser.add_subparsers(dest="custody_cmd", required=True)

    def common(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
        p.add_argument("--bundle", required=True,
                       help="campaign bundle directory (campaign.json + evidence.json)")
        p.add_argument("--json", action="store_true", help="emit JSON instead of text")
        p.add_argument("--require-release", action="store_true",
                       help=f"exit {EXIT_GATE_CLOSED} unless the release gate is open")
        p.add_argument("--observed-device", default=None,
                       help="device of the card now mounted (identity cross-check)")
        p.add_argument("--observed-uuid", default=None,
                       help="filesystem UUID of the card now mounted")
        p.add_argument("--observed-label", default=None, help="volume label of the card")
        p.add_argument("--policy-file", default=None,
                       help="JSON file overriding custody.policy")
        return p

    common(sub.add_parser("status", help="Show derived custody state (read-only)."))
    common(sub.add_parser("plan", help="Show the next safe operation (read-only)."))
    pv = common(sub.add_parser("verify",
                               help="Describe the destination verification set (read-only)."))
    pv.add_argument("--execute", action="store_true",
                    help="refused: verification execution is separately authorized")
    pi = common(sub.add_parser("import", help="Validate (default) or apply an evidence document."))
    pi.add_argument("--evidence", required=True, help="path to an evidence JSON document")
    pi.add_argument("--apply", action="store_true", help="write the evidence into the bundle")

    pn = sub.add_parser("new", help="Create a campaign record for observed card hardware.")
    pn.add_argument("--bundle", required=True, help="campaign bundle directory to create")
    pn.add_argument("--card-id", required=True, help="operator label, e.g. CARD-02")
    pn.add_argument("--device", default=None, help="block device, e.g. /dev/sdb1")
    pn.add_argument("--uuid", default=None,
                    help="filesystem UUID / volume id - the authoritative card identity")
    pn.add_argument("--serial", default=None, help="card serial, when the device exposes one")
    pn.add_argument("--label", default=None, help="volume label (recorded, never identity)")
    pn.add_argument("--mount", default=None, help="observed mount point")
    pn.add_argument("--read-only", action="store_true",
                    help="record the source as a read-only mount (custody expects this)")
    pn.add_argument("--created-at", default=None,
                    help="ISO-8601 creation instant (default: now, UTC)")
    pn.add_argument("--apply", action="store_true", help="write campaign.json")
    pn.add_argument("--json", action="store_true", help="emit JSON instead of text")
    pn.add_argument("--policy-file", default=None, help="JSON file overriding custody.policy")
    return parser


def _policy_from_file(path: Optional[str]) -> Optional[CustodyPolicy]:
    if not path:
        return None
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return CustodyPolicy.from_dict(raw.get("policy", raw))


def _observed(args) -> Optional[CardIdentity]:
    if not (args.observed_device or args.observed_uuid or args.observed_label):
        return None
    return CardIdentity(
        device=args.observed_device,
        filesystem_uuid=args.observed_uuid,
        label=args.observed_label,
    )


def _status(args) -> CampaignStatus:
    policy = _policy_from_file(getattr(args, "policy_file", None))
    if policy is None:
        policy = load_policy(load_custody_config())
    return load_status(args.bundle, policy, observed_card=_observed(args))


def cmd_status(args) -> int:
    status = _status(args)
    if args.json:
        print(status_json(status))
    else:
        sys.stdout.write(status_text(status))
    if args.require_release and not status.source_release_allowed:
        return EXIT_GATE_CLOSED
    return EXIT_OK


def cmd_plan(args) -> int:
    status = _status(args)
    plan = status.plan
    if args.json:
        print(plan_json(plan))
    else:
        sys.stdout.write(plan_text(plan))
    if args.require_release and not status.source_release_allowed:
        return EXIT_GATE_CLOSED
    return EXIT_OK


def cmd_verify(args) -> int:
    """Describe the verification work. Never performs it."""
    if args.execute:
        print(EXECUTE_REFUSED, file=sys.stderr)
        return EXIT_GATE_CLOSED
    status = _status(args)
    ev = status.evidence
    remaining = max(ev.copy.completed.files - ev.destination.verified_files, 0)
    payload = {
        "campaign_id": status.campaign.campaign_id,
        "state": status.derivation.state.value,
        "verification_started": ev.destination.verification_started,
        "verification_complete": ev.destination.verification_complete,
        "objects_to_verify": remaining,
        "objects_already_verified": ev.destination.verified_files,
        "unverified_objects_present_at_destination": ev.destination.unverified_present_files,
        "executed": False,
        "requires_operator_authorization": True,
        "source_mutation_allowed": False,
        "source_deletion_allowed": False,
    }
    if args.json:
        print(json.dumps(payload, sort_keys=True, indent=2, default=str))
    else:
        sys.stdout.write(
            "\n".join(
                [
                    f"campaign_id            {payload['campaign_id']}",
                    f"state                  {payload['state']}",
                    f"objects_to_verify      {payload['objects_to_verify']}",
                    f"already_verified       {payload['objects_already_verified']}",
                    f"present_unverified     "
                    f"{payload['unverified_objects_present_at_destination']}",
                    "executed               false",
                    "NOTE                   this command describes work only; "
                    "execution is separately authorized",
                ]
            )
            + "\n"
        )
    if args.require_release and not status.source_release_allowed:
        return EXIT_GATE_CLOSED
    return EXIT_OK


def cmd_import(args) -> int:
    policy = _policy_from_file(getattr(args, "policy_file", None))
    if policy is None:
        policy = load_policy(load_custody_config())
    raw = json.loads(Path(args.evidence).read_text(encoding="utf-8"))
    result = import_evidence(args.bundle, raw, policy, apply=bool(args.apply))
    if args.json:
        print(json.dumps(result, sort_keys=True, indent=2, default=str))
    else:
        mode = result["mode"]
        sys.stdout.write(
            f"mode                       {mode}\n"
            f"campaign_id                {result['campaign_id']}\n"
            f"derived_state              {result['derived_state']}\n"
            f"source_release_allowed     "
            f"{str(result['source_release_allowed']).lower()}\n"
            f"would_write                {result['would_write']}\n"
        )
        if result.get("declared_state_keys_ignored"):
            sys.stdout.write(
                "declared_state_keys_ignored  "
                + ",".join(result["declared_state_keys_ignored"])
                + "\n"
            )
        if result.get("error"):
            sys.stdout.write(f"error                      {result['error']}\n")
    return EXIT_OK if not result.get("error") else EXIT_USAGE


def cmd_new(args) -> int:
    """Create a campaign record for observed card hardware. Explicitly authorized."""
    observed = CardIdentity(
        device=args.device,
        filesystem_uuid=args.uuid,
        serial=args.serial,
        label=args.label,
    )
    created_at = args.created_at or datetime.now(timezone.utc).replace(
        microsecond=0
    ).isoformat().replace("+00:00", "Z")
    result = new_campaign(
        args.bundle,
        card_id=args.card_id,
        observed=observed,
        mount_point=args.mount,
        read_only=bool(args.read_only),
        created_at=created_at,
        config=load_custody_config(),
        apply=bool(args.apply),
    )
    if args.json:
        print(json.dumps(result, sort_keys=True, indent=2, default=str))
    else:
        sys.stdout.write(
            f"mode                       {result['mode']}\n"
            f"campaign_id                {result['campaign_id']}\n"
            f"card_id                    {result['card_id']}\n"
            f"card_key                   {result['card_key']}\n"
            f"source_read_only           {str(result['source_read_only']).lower()}\n"
            f"destination_resolved       "
            f"{str(result['destination_resolved']).lower()}\n"
            f"state                      {result['state']}\n"
            f"would_write                {result['would_write']}\n"
        )
    return EXIT_OK


_HANDLERS = {
    "status": cmd_status,
    "plan": cmd_plan,
    "verify": cmd_verify,
    "import": cmd_import,
    "new": cmd_new,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    handler = _HANDLERS[args.custody_cmd]
    try:
        return handler(args)
    except CampaignCreationError as exc:
        print(f"custody: refusing to create campaign: {exc}", file=sys.stderr)
        return EXIT_GATE_CLOSED
    except BundleError as exc:
        print(f"custody: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except FileNotFoundError as exc:
        print(f"custody: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except json.JSONDecodeError as exc:
        print(f"custody: invalid JSON: {exc}", file=sys.stderr)
        return EXIT_USAGE


__all__ = [
    "EXIT_GATE_CLOSED",
    "EXIT_OK",
    "EXIT_USAGE",
    "build_parser",
    "main",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
