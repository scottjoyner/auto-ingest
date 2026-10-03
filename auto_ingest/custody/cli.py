"""auto_ingest.custody.cli - read-only custody CLI.

    auto-ingest custody status  --bundle PATH [--json]
    auto-ingest custody plan    --bundle PATH [--json]
    auto-ingest custody verify  --bundle PATH [--json] [--execute]
    auto-ingest custody import  --bundle PATH --evidence FILE [--apply]
    auto-ingest custody new     --bundle PATH --card-id ID --uuid/--device/--label ... [--apply]
    auto-ingest custody reconcile --bundle PATH [--json]
    auto-ingest custody observe-mount --bundle PATH [--json] [--apply]
    auto-ingest custody capacity --bundle PATH [--json]
    auto-ingest custody preflight --bundle PATH [--json] [--job-dir PATH]

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
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

from .campaign import CardIdentity
from .capacity import (
    DEFAULT_HEADROOM_FRACTION,
    DEFAULT_HEADROOM_MIN_BYTES,
    capacity_report,
)
from .hashing import DEFAULT_ALGORITHM, hash_source, to_evidence
from .lock import competing_activity, is_locked, uncoordinated_writers
from .mounts import observe_campaign
from .policy import CustodyPolicy
from .report import plan_json, plan_text, status_json, status_text
from .store import (
    BundleError,
    CampaignCreationError,
    CampaignStatus,
    _write_json_atomic,
    import_evidence,
    load_campaign,
    load_custody_config,
    load_evidence,
    load_policy,
    load_status,
    new_campaign,
    reconcile_preview,
)
from .verify import to_evidence as verify_evidence
from .verify import verify_destination

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

    def common(p: argparse.ArgumentParser, *, gate: bool = True,
               observed: bool = True) -> argparse.ArgumentParser:
        p.add_argument("--bundle", required=True,
                       help="campaign bundle directory (campaign.json + evidence.json)")
        p.add_argument("--json", action="store_true", help="emit JSON instead of text")
        if gate:
            p.add_argument("--require-release", action="store_true",
                           help=f"exit {EXIT_GATE_CLOSED} unless the release gate is open")
        if observed:
            p.add_argument("--observed-device", default=None,
                           help="device of the card now mounted (identity cross-check)")
            p.add_argument("--observed-uuid", default=None,
                           help="filesystem UUID of the card now mounted")
            p.add_argument("--observed-label", default=None,
                           help="volume label of the card")
        p.add_argument("--policy-file", default=None,
                       help="JSON file overriding custody.policy")
        return p

    common(sub.add_parser("status", help="Show derived custody state (read-only)."))
    common(sub.add_parser("plan", help="Show the next safe operation (read-only)."))
    pv = common(sub.add_parser("verify",
                               help="Describe or run destination verification (read-only)."),
                observed=False)
    pv.add_argument("--execute", action="store_true",
                    help="refused: verification execution is separately authorized")
    pv.add_argument("--destination", default=None,
                    help="destination root to verify against (default: campaign's)")
    pv.add_argument("--recheck", action="store_true",
                    help="re-verify objects already proven at the destination")
    pv.add_argument("--algorithm", default=DEFAULT_ALGORITHM)
    pv.add_argument("--limit", type=int, default=None,
                    help="stop after N objects (a bounded probe)")
    pv.add_argument("--apply", action="store_true",
                    help="record the verification result as campaign evidence")
    # No --require-release on import: it is a mid-campaign write, so "is this
    # releasable yet?" has no coherent meaning there. Accepting a flag and
    # ignoring it is worse than not having it.
    pi = common(sub.add_parser("import", help="Validate (default) or apply an evidence document."),
                gate=False, observed=False)
    pi.add_argument("--evidence", required=True, help="path to an evidence JSON document")
    pi.add_argument("--apply", action="store_true", help="write the evidence into the bundle")

    pom = common(sub.add_parser(
        "observe-mount",
        help="Observe what the kernel says is mounted (read-only)."),
        observed=False)
    pom.add_argument("--apply", action="store_true",
                     help="record the observation into the campaign bundle")

    pcap = common(sub.add_parser(
        "capacity",
        help="Check the destination has room for the outstanding bytes (read-only)."),
        observed=False)
    pcap.add_argument("--headroom-fraction", type=float,
                      default=DEFAULT_HEADROOM_FRACTION,
                      help="extra space demanded as a fraction of the requirement")
    pcap.add_argument("--headroom-min-bytes", type=int,
                      default=DEFAULT_HEADROOM_MIN_BYTES,
                      help="floor for the headroom, so 'nearly full' still fails")

    pf = common(sub.add_parser(
        "preflight",
        help="Everything an executor would need to be safe, in one answer."),
        observed=False)
    pf.add_argument("--headroom-fraction", type=float,
                    default=DEFAULT_HEADROOM_FRACTION)
    pf.add_argument("--headroom-min-bytes", type=int, default=DEFAULT_HEADROOM_MIN_BYTES)
    pf.add_argument("--job-dir", default=None,
                    help="directory of .job files whose presence means work is queued")
    pf.add_argument("--drop-root", default=None,
                    help="ingest-worker DROP_ROOT, to check for claimed work")

    ph = common(sub.add_parser(
        "hash",
        help="Hash source objects into the campaign ledger (read-only on source)."),
        observed=False)
    ph.add_argument("--root", action="append", default=None,
                    help="directory to walk; repeatable")
    ph.add_argument("--limit", type=int, default=None,
                    help="stop after N new hashes (a bounded probe)")
    ph.add_argument("--algorithm", default=DEFAULT_ALGORITHM)
    ph.add_argument("--apply", action="store_true",
                    help="record the result as campaign evidence")

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

    prc = sub.add_parser("reconcile",
                         help="Diff the source and destination ledgers (read-only).")
    prc.add_argument("--bundle", required=True, help="campaign bundle directory")
    prc.add_argument("--json", action="store_true", help="emit JSON instead of text")
    prc.add_argument("--max-samples", type=int, default=None,
                     help="cap on per-category key samples (default: policy value)")
    prc.add_argument("--policy-file", default=None, help="JSON file overriding custody.policy")
    prc.add_argument("--require-release", action="store_true",
                     help=f"exit {EXIT_GATE_CLOSED} unless the release gate is open")
    prc.add_argument("--require-proposal", action="store_true",
                     help=f"exit {EXIT_GATE_CLOSED} unless a usable proposal is offered")
    return parser


def _policy_from_file(path: Optional[str]) -> Optional[CustodyPolicy]:
    if not path:
        return None
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return CustodyPolicy.from_dict(raw.get("policy", raw))


def _observed(args) -> Optional[CardIdentity]:
    # Subcommands that do not take --observed-* simply have nothing to compare.
    if not (getattr(args, "observed_device", None)
            or getattr(args, "observed_uuid", None)
            or getattr(args, "observed_label", None)):
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
    """Describe the verification work, or perform the read-only verification pass.

    Verification reads the destination and writes the campaign's own ledger, so it
    copies and deletes nothing: a verification that moved bytes would produce a
    result that looks authoritative but proves nothing. `--execute` remains
    refused because *copying* is separately authorized.
    """
    if args.execute:
        print(EXECUTE_REFUSED, file=sys.stderr)
        return EXIT_GATE_CLOSED
    policy = _policy_from_file(getattr(args, "policy_file", None))
    if policy is None:
        policy = load_policy(load_custody_config())
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
    destination = args.destination
    if destination is None:
        destination = status.campaign.destination.host_path
    result = None
    if destination:
        result = verify_destination(
            args.bundle, destination, algorithm=args.algorithm,
            limit=args.limit, recheck=args.recheck,
        )
        payload["verification"] = result.to_dict()
        payload["objects_to_verify"] = result.checked
        payload["objects_already_verified"] = result.skipped_existing + result.verified
        payload["executed"] = False   # verification ran; nothing was copied
        payload["destination_root"] = destination
    if args.apply and payload.get("verification"):
        applied = import_evidence(args.bundle, verify_evidence(result), policy, apply=True)
        payload["applied"] = bool(applied.get("applied"))
        payload["evidence_result"] = applied
        status = load_status(args.bundle, policy)
        payload["state_after"] = status.derivation.state.value
        payload["source_release_allowed_after"] = status.source_release_allowed
    if args.json:
        print(json.dumps(payload, sort_keys=True, indent=2, default=str))
    else:
        lines = [
            f"campaign_id            {payload['campaign_id']}",
            f"state                  {payload['state']}",
            f"objects_to_verify      {payload['objects_to_verify']}",
            f"objects_already_verified {payload['objects_already_verified']}",
            f"present_unverified     "
            f"{payload['unverified_objects_present_at_destination']}",
            "copied                 false",
            "NOTE                   verification reads only; copying is separately "
            "authorized",
        ]
        verification = payload.get("verification")
        if verification:
            lines = [
                f"campaign_id            {payload['campaign_id']}",
                f"destination_root       {payload['destination_root']}",
                f"verified               {verification['verified']}",
                f"verified_bytes         {verification['verified_bytes']}",
                f"missing                {verification['missing']}",
                f"mismatched             {verification['mismatched']}",
                f"failed                 {verification['failed']}",
                f"ledger                 {verification['ledger_path']}",
                "copied                 false   (verification reads only)",
            ]
            if payload.get("applied"):
                lines += [
                    f"state_after            {payload['state_after']}",
                    f"source_release_allowed {str(payload['source_release_allowed_after']).lower()}",
                ]
        sys.stdout.write("\n".join(lines) + "\n")
    if verification_failures(result):
        return EXIT_GATE_CLOSED
    if args.require_release and not status.source_release_allowed:
        return EXIT_GATE_CLOSED
    return EXIT_OK


def verification_failures(result) -> int:
    """Objects at the destination that are wrong, absent, or unreadable."""
    if result is None:
        return 0
    return result.missing + result.mismatched + result.failed


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


def cmd_observe_mount(args) -> int:
    """Report what the kernel says is mounted. Read-only; --apply records it.

    The point is that `campaign.source.read_only` is a *declared* field, so this
    is the first thing that can contradict it.
    """
    policy = _policy_from_file(getattr(args, "policy_file", None))
    if policy is None:
        policy = load_policy(load_custody_config())
    campaign = load_campaign(args.bundle)
    report = observe_campaign(campaign)
    source = report["source"]
    payload = {
        "campaign_id": report["campaign_id"],
        "source": source,
        "destination": report["destination"],
        "declared_read_only_trusted": False,
        "observations": True,
        "applied": False,
    }
    if args.apply:
        campaign_path = Path(args.bundle) / "campaign.json"
        _record_observed_read_only(campaign_path, source)
        payload["applied"] = True
    if args.json:
        print(json.dumps(payload, sort_keys=True, indent=2, default=str))
    else:
        lines = [
            f"campaign_id                {report['campaign_id']}",
            f"source_mount_point         {source['mount_point']}",
            f"source_present             {str(source['observation']['present']).lower()}",
            f"source_observed_read_only  {source['observed_read_only']}",
            f"source_declared_read_only  {str(source['declared_read_only']).lower()}",
            f"declaration_agrees         {source['read_only_agrees_with_declaration']}",
            f"source_filesystem          {source['observation']['filesystem_type']}",
            f"destination_present        "
            f"{str(report['destination']['observed_mounted']).lower()}",
            f"destination_filesystem     "
            f"{report['destination']['observation']['filesystem_type']}",
            f"applied                    {str(payload['applied']).lower()}",
        ]
        if source["observed_read_only"] is False:
            lines.append(
                "  WARNING           the source is mounted WRITABLE; do not copy from it"
            )
        if source["read_only_agrees_with_declaration"] is False:
            lines.append(
                "  CONFLICT          the bundle's declared read_only disagrees with the"
                " kernel"
            )
        sys.stdout.write("\n".join(lines) + "\n")
    if source["observed_read_only"] is not True:
        return EXIT_GATE_CLOSED
    return EXIT_OK


def _record_observed_read_only(campaign_path: Path, source: dict) -> None:
    """Stamp the observation next to the declaration, never over it.

    The declaration survives untouched, because the two are different claims: one
    is what the bundle asserts, the other is what the kernel reports. Only a
    human can resolve a disagreement between them.
    """
    raw = json.loads(campaign_path.read_text(encoding="utf-8"))
    src = raw.setdefault("source", {})
    src["observed_read_only"] = source["observed_read_only"]
    src["read_only_agrees_with_declaration"] = source[
        "read_only_agrees_with_declaration"
    ]
    _write_json_atomic(campaign_path, raw)


def cmd_capacity(args) -> int:
    """Ask whether the destination can hold what is outstanding. Read-only."""
    policy = _policy_from_file(getattr(args, "policy_file", None))
    if policy is None:
        policy = load_policy(load_custody_config())
    campaign = load_campaign(args.bundle)
    evidence = load_evidence(args.bundle, policy)
    report = capacity_report(
        campaign, evidence,
        headroom_fraction=args.headroom_fraction,
        headroom_min_bytes=args.headroom_min_bytes,
    )
    payload = dict(report.to_dict())
    payload["campaign_id"] = campaign.campaign_id
    if args.json:
        print(json.dumps(payload, sort_keys=True, indent=2, default=str))
    else:
        sys.stdout.write(
            f"campaign_id            {campaign.campaign_id}\n"
            f"destination            {report.path or 'UNRESOLVED'}\n"
            f"outstanding_bytes      {report.required_bytes}\n"
            f"headroom_bytes         {report.headroom_bytes}\n"
            f"total_needed_bytes     {report.total_needed}\n"
            f"available_bytes        {report.available_bytes}\n"
            f"sufficient             {report.sufficient}\n"
            + (f"reason                 {report.reason}\n" if report.reason else "")
        )
    if report.sufficient is not True:
        return EXIT_GATE_CLOSED
    return EXIT_OK


def _queued_jobs(job_dir: Optional[str]) -> Tuple[int, ...]:
    """Count queued ``.job`` files. A read of one directory, no enumeration of /nas."""
    if not job_dir:
        return ()
    try:
        entries = os.listdir(job_dir)
    except OSError:
        return ()
    return tuple(sorted(e for e in entries if e.endswith(".job")))


def cmd_preflight(args) -> int:
    """One answer to "could an executor run right now, and why not if not?".

    Aggregates the observers plus the release gate, because an operator should
    not have to correlate four commands by hand - and because the whole value of
    preflight is that it is *complete*: an executor may only proceed when this
    says so, so anything left unchecked here is a hole in the gate.

    Read-only, like everything else in this package.
    """
    policy = _policy_from_file(getattr(args, "policy_file", None))
    if policy is None:
        policy = load_policy(load_custody_config())
    evidence = load_evidence(args.bundle, policy)
    status = load_status(args.bundle, policy)
    campaign_obj = status.campaign

    mounts = observe_campaign(campaign_obj)
    source = mounts["source"]
    cap = capacity_report(campaign_obj, evidence,
                          headroom_fraction=args.headroom_fraction,
                          headroom_min_bytes=args.headroom_min_bytes)
    queued = _queued_jobs(args.job_dir)
    activity = competing_activity(job_dir=args.job_dir, drop_root=args.drop_root)

    checks: list[dict] = []

    def add(name: str, ok: Optional[bool], detail: str, remedy: str) -> None:
        checks.append({"name": name, "ok": ok, "detail": detail, "remedy": remedy})

    add("source_present", source["observation"]["present"],
        f"mount_point={source['mount_point']}",
        "insert the card, or correct campaign_obj.source.mount_point")
    add("source_read_only_observed", source["observed_read_only"] is True,
        f"observed={source['observed_read_only']} declared={source['declared_read_only']}",
        "remount the source read-only; do not copy from a writable card")
    if source["read_only_agrees_with_declaration"] is False:
        add("declaration_matches_observation", False,
            "campaign_obj.source.read_only disagrees with the kernel",
            "re-record the campaign with `custody new` for the card actually present")
    add("destination_resolved", campaign_obj.destination.resolved,
        f"host_path={campaign_obj.destination.host_path or 'UNRESOLVED'}",
        "set CUSTODY_DESTINATION_ROOT or custody.destination_root")
    add("destination_mounted", cap.checked and campaign_obj.destination.mounted is not False,
        f"statable={cap.checked} mounted={campaign_obj.destination.mounted}",
        "mount the canonical destination")
    add("capacity_sufficient", cap.sufficient,
        f"need={cap.total_needed} available={cap.available_bytes}",
        "free space at the destination, or point at a larger one")
    add("no_competing_jobs", not queued,
        f"queued={len(queued)}" + (f" {queued[:5]}" if queued else ""),
        "let the existing worker drain the queue, or stop it for the campaign")

    dest_key = campaign_obj.destination.logical.canonical
    lock_held = is_locked(dest_key)
    add("destination_not_locked_by_another_campaign", not lock_held,
        f"{dest_key} locked={lock_held}",
        "another campaign holds this destination; wait for it or pick another")

    # Competing writers are reported, not gated on: they are a standing property
    # of this host, and none of them takes the campaign lock yet. Silently passing
    # this would claim a safety the repo cannot currently provide.
    blockers_ = uncoordinated_writers(activity)
    add("no_uncoordinated_writers", not blockers_,
        "none" if not blockers_ else ",".join(blockers_),
        "stop sync-service / ingest-worker for the campaign, or teach them to "
        "consult the campaign lock as a separate change")
    add("release_gate_open", status.release.allowed,
        "blockers=" + ",".join(b.code for b in status.release.blockers),
        "resolve the release blockers; see `custody status`")

    failed = [c for c in checks if c["ok"] is False]
    unknown = [c for c in checks if c["ok"] is None]
    payload = {
        "campaign_id": campaign_obj.campaign_id,
        "state": status.derivation.state.value,
        # Known, standing, uncoordinated writers keep this false. They cannot be
        # resolved from inside this package: the fix belongs to those services.
        "safe_to_execute": not failed and not unknown and not blockers_,
        "checks": checks,
        "competing_activity": [a.to_dict() for a in activity],
        "failed": [c["name"] for c in failed],
        "unverified": [c["name"] for c in unknown],
        "source_release_allowed": status.source_release_allowed,
        "plan_fingerprint": status.plan.plan_fingerprint,
    }
    if args.json:
        print(json.dumps(payload, sort_keys=True, indent=2, default=str))
    else:
        lines = [
            f"campaign_id            {payload['campaign_id']}",
            f"state                  {payload['state']}",
            f"safe_to_execute        {str(payload['safe_to_execute']).lower()}",
            "checks:",
        ]
        for check in checks:
            mark = {True: "ok  ", False: "FAIL", None: "unknown"}[check["ok"]]
            lines.append(f"  [{mark}] {check['name']:<28} {check['detail']}")
            if check["ok"] is not True:
                lines.append(f"         -> {check['remedy']}")
        for item in payload["competing_activity"]:
            lines.append(f"  competing     {item['kind']}: {item['detail']}")
        sys.stdout.write("\n".join(lines) + "\n")
    return EXIT_OK if payload["safe_to_execute"] else EXIT_GATE_CLOSED


def _discover_keys(roots: Sequence[str], pattern: Optional[str]) -> Dict[str, Path]:
    """Map custody key -> source path, sorted.

    The key is the POSIX-relative path: it is stable across runs, unique per
    object, and it is what reconciliation joins on, so both ledgers must use it.
    """
    import fnmatch

    found: Dict[str, Path] = {}
    for root in roots:
        base = Path(root)
        if not base.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames.sort()
            for name in sorted(filenames):
                if pattern and not fnmatch.fnmatch(name, pattern):
                    continue
                full = Path(dirpath) / name
                found[full.relative_to(base).as_posix()] = full
    return found


def cmd_hash(args) -> int:
    """Produce hash.jsonl - the first custody ledger that is actually measured.

    Read-only on the source: every file is opened `rb` and read. The only file
    created is the ledger inside the campaign bundle.
    """
    policy = _policy_from_file(getattr(args, "policy_file", None))
    if policy is None:
        policy = load_policy(load_custody_config())
    if not args.root:
        campaign_obj = load_campaign(args.bundle)
        if not campaign_obj.source.mount_point:
            print("custody: no source mount recorded; pass --root", file=sys.stderr)
            return EXIT_USAGE
        args.root = [campaign_obj.source.mount_point]

    keys = _discover_keys(args.root, None)
    result = hash_source(args.bundle, keys, algorithm=args.algorithm,
                         limit=args.limit)
    payload = dict(result.to_dict())
    payload["discovered"] = len(keys)
    payload["applied"] = False
    if args.apply:
        fragment = to_evidence(result, algorithm=args.algorithm,
                                discovered=len(keys))
        write_status = import_evidence(args.bundle, fragment, policy, apply=True)
        payload["applied"] = bool(write_status.get("applied"))
        payload["evidence_result"] = write_status
    if args.json:
        print(json.dumps(payload, sort_keys=True, indent=2, default=str))
    else:
        sys.stdout.write(
            f"ledger                {result.ledger_path}\n"
            f"discovered            {len(keys)}\n"
            f"hashed                {result.hashed}\n"
            f"skipped_existing      {result.skipped_existing}\n"
            f"failed                {result.failed}\n"
            f"bytes_read            {result.bytes_read}\n"
            f"complete              {str(result.complete).lower()}\n"
            f"applied               {str(payload['applied']).lower()}\n"
            + ("".join(f"  error               {e}\n" for e in result.errors))
        )
    if result.failed:
        return EXIT_GATE_CLOSED
    return EXIT_OK


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


def cmd_reconcile(args) -> int:
    """Diff the two ledgers and show what the answer would mean. Never applies."""
    policy = _policy_from_file(getattr(args, "policy_file", None))
    if policy is None:
        policy = load_policy(load_custody_config())
    result, status = reconcile_preview(
        args.bundle, policy, max_samples=args.max_samples
    )
    payload = {
        "campaign_id": status.campaign_id,
        "reconciliation": result.to_dict(),
        "usable": result.usable,
        "proposal": result.proposal(),
        "applied": False,
        "current_state": status.derivation.state.value,
        "state_if_imported": status.derivation.state.value,
        "source_release_allowed_if_imported": status.source_release_allowed,
        "next_safe_action_if_imported": status.next_safe_action,
        "requires_operator_authorization": True,
    }
    if args.json:
        print(json.dumps(payload, sort_keys=True, indent=2, default=str))
    else:
        lines = [
            f"campaign_id                {payload['campaign_id']}",
            f"ledgers_readable           {str(result.usable).lower()}",
            f"source_objects             {result.source_objects}",
            f"destination_objects        {result.destination_objects}",
            f"verified                   {result.verified}",
            f"source_only                {result.source_only}",
            f"destination_only           {result.destination_only}",
            f"mismatched                 {result.mismatched}",
            f"unverifiable               {result.unverifiable}",
            f"state_now                  {status.derivation.state.value}",
            f"state_if_imported          {payload['state_if_imported']}",
            f"source_release_allowed_if_imported  "
            f"{str(status.source_release_allowed).lower()}",
            "applied                    false  (import the proposal with --apply)",
        ]
        if not result.usable:
            lines.insert(1, "NOTE: no proposal is offered - "
                            + "; ".join(result.incoherent))
        for name, samples in (
            ("source_only", result.source_only_samples),
            ("destination_only", result.destination_only_samples),
            ("mismatched", result.mismatched_samples),
            ("unverifiable", result.unverifiable_samples),
        ):
            if samples:
                lines.append(f"  {name:<21} {', '.join(samples)}")
        sys.stdout.write("\n".join(lines) + "\n")
    if args.require_proposal and not result.usable:
        print("custody: ledgers are not coherent; refusing to report a diff as a result",
              file=sys.stderr)
        return EXIT_GATE_CLOSED
    if args.require_release and not status.source_release_allowed:
        return EXIT_GATE_CLOSED
    return EXIT_OK


_HANDLERS = {
    "status": cmd_status,
    "plan": cmd_plan,
    "verify": cmd_verify,
    "import": cmd_import,
    "new": cmd_new,
    "reconcile": cmd_reconcile,
    "observe-mount": cmd_observe_mount,
    "capacity": cmd_capacity,
    "preflight": cmd_preflight,
    "hash": cmd_hash,
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
