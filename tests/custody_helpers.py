"""Shared builders for the SD-card custody tests.

Every helper here returns plain in-memory objects. Nothing in this module touches
a real card, a real mount, or a real network path - the tests are hermetic.
"""
from __future__ import annotations

from pathlib import Path

from auto_ingest.custody import (
    Campaign,
    CampaignEvidence,
    CardIdentity,
    CopyEvidence,
    Counts,
    CustodyPolicy,
    DestinationEvidence,
    DestinationRef,
    ErrorEvidence,
    HashEvidence,
    InventoryEvidence,
    LogicalDestination,
    ReconciliationEvidence,
    SourceRef,
    StorageIdentity,
    WorkerEvidence,
)

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "custody"
CARD_01_BUNDLE = FIXTURE_ROOT / "card-01"

CREATED_AT = "2026-04-12T09:15:00Z"


def card(uuid: str = "7A3E-2C19", device: str = "/dev/sdb1",
         label: str = "UNTITLED", serial: str = "0x8f2a41c7") -> CardIdentity:
    return CardIdentity(device=device, filesystem_uuid=uuid, label=label, serial=serial)


def destination(
    *,
    name: str = "primary",
    relative_path: str = "fileserver/dashcam",
    host_path: str = "/mnt/custody/destination",
    mounted: bool = True,
    identity: StorageIdentity | None = None,
) -> DestinationRef:
    return DestinationRef(
        logical=LogicalDestination(name=name, relative_path=relative_path),
        host_path=host_path,
        resolved_from="test",
        mounted=mounted,
        identity=identity if identity is not None else StorageIdentity(
            filesystem_uuid="DEST-9988", device="/dev/nvme9n9p1", label="custody",
            filesystem_type="ext4",
        ),
    )


def campaign(
    *,
    card_identity: CardIdentity | None = None,
    dest: DestinationRef | None = None,
    read_only: bool = True,
    card_id: str = "CARD-01",
    mount_point: str = "/media/scott/UNTITLED",
) -> Campaign:
    card_identity = card_identity or card()
    source = SourceRef(mount_point=mount_point, read_only=read_only, card=card_identity)
    return Campaign.create(
        card_id=card_id,
        source=source,
        destination=dest or destination(),
        created_at=CREATED_AT,
    )


def inventory(files: int = 0, nbytes: int = 0, *, complete: bool = False,
              verified: bool = False, started: bool | None = None) -> InventoryEvidence:
    if started is None:
        started = bool(complete or verified or files)
    return InventoryEvidence(discovered_files=files, discovered_bytes=nbytes,
                             complete=complete, verified=verified, started=started)


def hashing(verified_files: int = 0, *, verified_bytes: int = 0, failed: int = 0,
            complete: bool = False, started: bool | None = None,
            algorithm: str = "sha256",
            exemptions: tuple[str, ...] = ()) -> HashEvidence:
    if started is None:
        started = bool(complete or verified_files or failed)
    return HashEvidence(verified_files=verified_files, verified_bytes=verified_bytes,
                        failed=failed, complete=complete, started=started,
                        algorithm=algorithm, exemptions=exemptions)


def copying(*, planned_files: int = 0, planned_bytes: int = 0, completed_files: int = 0,
            completed_bytes: int = 0, started: bool = False, result_complete: bool = False,
            interrupted: bool = False, ledger_complete: bool = False) -> CopyEvidence:
    return CopyEvidence(
        planned=Counts(planned_files, planned_bytes),
        completed=Counts(completed_files, completed_bytes),
        started=started,
        result_complete=result_complete,
        interrupted=interrupted,
        ledger_complete=ledger_complete,
    )


def destination_evidence(*, verified_files: int = 0, verified_bytes: int = 0,
                         failures: int = 0, verification_started: bool = False,
                         verification_complete: bool = False,
                         unverified_present_files: int = 0,
                         observed_identity: StorageIdentity | None = None
                         ) -> DestinationEvidence:
    return DestinationEvidence(
        verified_files=verified_files,
        verified_bytes=verified_bytes,
        failures=failures,
        verification_started=verification_started,
        verification_complete=verification_complete,
        unverified_present_files=unverified_present_files,
        observed_identity=observed_identity,
    )


def reconciliation(*, source_only: int = 0, destination_only: int = 0,
                   mismatched: int = 0) -> ReconciliationEvidence:
    return ReconciliationEvidence(source_only=source_only,
                                  destination_only=destination_only,
                                  mismatched=mismatched)


def worker(*, identity: str | None = None, status: str = "unknown") -> WorkerEvidence:
    return WorkerEvidence(identity=identity, status=status)


def errors(*, unresolved: int = 0, fatal: int = 0,
           summaries: tuple[str, ...] = (), witness: str | None = None) -> ErrorEvidence:
    return ErrorEvidence(unresolved=unresolved, fatal=fatal, summaries=summaries,
                         witness=witness)


def evidence(
    *,
    inv: InventoryEvidence | None = None,
    hsh: HashEvidence | None = None,
    cpy: CopyEvidence | None = None,
    dst: DestinationEvidence | None = None,
    rec: ReconciliationEvidence | None = None,
    wkr: WorkerEvidence | None = None,
    err: ErrorEvidence | None = None,
    campaign_id: str = "test-campaign",
) -> CampaignEvidence:
    return CampaignEvidence(
        campaign_id=campaign_id,
        inventory=inv if inv is not None else inventory(),
        hashing=hsh if hsh is not None else hashing(),
        copy=cpy if cpy is not None else copying(),
        destination=dst if dst is not None else destination_evidence(),
        reconciliation=rec if rec is not None else reconciliation(),
        worker=wkr if wkr is not None else worker(),
        errors=err if err is not None else errors(),
    )


def strict_policy(**overrides) -> CustodyPolicy:
    base = {
        "required_scope": "all_inventory",
        "allow_hash_exemptions": True,
        "declared_hash_exemptions": (),
        "require_operator_witness": False,
        "strict_destination_scope": False,
        "require_destination_identity": True,
        "require_read_only_source": True,
        "require_mounted_destination": True,
    }
    base.update(overrides)
    return CustodyPolicy(**base)


def fully_copied_campaign(*, total_files: int = 100, total_bytes: int = 1_000_000,
                          verified_files: int | None = None,
                          source_only: int = 0, destination_only: int = 0,
                          mismatched: int = 0, unresolved: int = 0,
                          unverified_present_files: int = 0):
    """Hash complete, copy attested complete, verification complete and clean."""
    verified = total_files if verified_files is None else verified_files
    c = campaign()
    e = evidence(
        inv=inventory(total_files, total_bytes, complete=True, verified=True),
        hsh=hashing(total_files, verified_bytes=total_bytes, complete=True),
        cpy=copying(planned_files=total_files, planned_bytes=total_bytes,
                    completed_files=total_files, completed_bytes=total_bytes,
                    started=True, result_complete=True, ledger_complete=True),
        dst=destination_evidence(
            verified_files=verified,
            verified_bytes=total_bytes if verified == total_files else 0,
            verification_started=True,
            verification_complete=True,
            unverified_present_files=unverified_present_files,
            observed_identity=c.destination.identity,
        ),
        rec=reconciliation(source_only=source_only, destination_only=destination_only,
                           mismatched=mismatched),
        wkr=worker(status="stopped"),
        err=errors(unresolved=unresolved),
    )
    return c, e


def write_bundle(root, campaign_obj: Campaign, evidence_obj: CampaignEvidence) -> Path:
    """Materialise a campaign bundle on disk for CLI/store tests."""
    import json

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "campaign.json").write_text(
        json.dumps(campaign_obj.to_dict(), sort_keys=True, indent=2), encoding="utf-8"
    )
    (root / "evidence.json").write_text(
        json.dumps(evidence_obj.to_dict(), sort_keys=True, indent=2), encoding="utf-8"
    )
    return root
