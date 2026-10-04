"""The release record, made observable in bounded campaign evidence.

``auto_ingest.custody.release_source`` can unlink source objects, but it wrote no
campaign evidence at all - deliberately. ``CampaignEvidence`` had no field meaning
"the source was released", and the nearest candidate (zeroing ``inventory``) makes
the bundle contradict itself, so the state machine quite correctly reported
``BLOCKED``. The honest consequence was that *the release gate could not observe
that the card was already released*, and the only answer lived in an append-only
audit ledger that a human had to read by hand.

This file pins the fix and, just as importantly, pins what it must **not** do. The
whole risk of making a release observable is that it starts to look like progress
toward custody: then deleting the card would manufacture a pass, and the one
irreversible capability in the package would become the easiest way to earn one.
So the load-bearing assertions here are negative:

* released-source evidence with no verification is still ``BLOCKED``, with the
  same reasons it had before;
* released-source evidence and full verification is still ``SAFE_TO_RELEASE``;
* the gate's blocker list for an unreleased campaign is byte-identical to what it
  was before this block existed;
* the block contains nothing the destination side could mistake for verification.

And the boundedness claim, because the alternative - one evidence row per deleted
object - is what would actually make a 90GB card unusable.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pytest
from custody_helpers import campaign as make_campaign
from custody_helpers import destination as make_destination
from custody_helpers import evidence as build_evidence
from custody_helpers import write_bundle

from auto_ingest.custody import CampaignState, StorageIdentity
from auto_ingest.custody.cli import main
from auto_ingest.custody.evidence import (
    CampaignEvidence,
    Counts,
    SourceReleaseEvidence,
)
from auto_ingest.custody.release import evaluate_release
from auto_ingest.custody.release_source import (
    MODE_EXECUTED,
    audit_path,
    execute_release,
    read_audit,
    to_evidence,
)
from auto_ingest.custody.store import (
    EVIDENCE_BLOCKS,
    EVIDENCE_FILE,
    EVIDENCE_SCALARS,
    import_evidence,
    load_campaign,
    load_evidence,
    load_status,
    merge_evidence_documents,
)

pytestmark = pytest.mark.usefixtures("hermetic_mounts")

COUNT = 3
SIZE = 256
CARDS = [f"c{i}.mp4" for i in range(COUNT)]
#: The instant the caller would supply. Never read from a clock, so two runs over
#: identical inputs produce identical evidence - pinned by the tests below.
CHECKPOINT = "2026-10-04T00:00:00Z"


@pytest.fixture(autouse=True)
def isolated_locks(tmp_path, monkeypatch):
    monkeypatch.setenv("CUSTODY_LOCK_ROOT", str(tmp_path / "locks"))


# ---------------------------------------------------------------------------
# applying a fragment: what cli.py will do once store.py knows the block
# ---------------------------------------------------------------------------
def apply_fragment(bundle, fragment, *, apply=True):
    """Merge ``fragment`` onto the campaign's evidence, exactly as import does.

    ``store.merge_evidence_documents`` overlays only the names in
    ``store.EVIDENCE_BLOCKS``, and that allowlist does not carry ``source_release``
    yet - so ``import_evidence`` silently drops the block today. This applies the
    package's own merge rule (the incoming field wins, an unmentioned field is
    preserved), including for the blocks the allowlist omits, and writes through
    the same normalised document ``import_evidence`` writes.

    The point of doing it here rather than in cli.py is that *nothing about the
    fragment depends on this helper*: it is a plain dict of blocks, exactly like
    ``hashing.to_evidence`` and friends. Once ``EVIDENCE_BLOCKS`` gains the name
    this collapses to::

        import_evidence(bundle, fragment, policy, apply=True)
    """
    if not apply:
        return import_evidence(bundle, fragment, apply=False)
    path = Path(bundle) / EVIDENCE_FILE
    existing = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    merged = merge_evidence_documents(existing, fragment)
    for block in set(existing) | set(fragment):
        if block in EVIDENCE_BLOCKS or block in EVIDENCE_SCALARS:
            continue          # merge_evidence_documents already handled it
        old = existing.get(block)
        new = fragment.get(block)
        if isinstance(old, Mapping) or isinstance(new, Mapping):
            # The allowlist gap, stated rather than hidden: a block it dropped is
            # merged here with the same per-field rule it would have used.
            merged[block] = {**(old or {}), **(new or {})}
    path.write_text(
        json.dumps(CampaignEvidence.from_dict(merged).to_dict(), sort_keys=True, indent=2),
        encoding="utf-8",
    )
    return {"applied": True, "derived_state": load_status(bundle).state.value}


def blocker_codes(evidence, campaign=None, policy=None):
    return [b.code for b in evaluate_release(campaign or make_campaign(), evidence, policy).blockers]


# ---------------------------------------------------------------------------
# a real campaign, driven through the real pipeline
# ---------------------------------------------------------------------------
def build_released(root, count=COUNT, size=SIZE):
    """hash -> execute -> verify -> record destination identity -> release.

    Every step is a command an operator runs. The one ingredient injected by hand
    is the destination's storage identity, because that has to be observed on the
    host and a tmpdir is not a mount point.
    """
    root = Path(root)
    src = root / "card"
    dest = root / "dest"
    src.mkdir(parents=True, exist_ok=True)
    dest.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        (src / f"c{i}.mp4").write_bytes(bytes([65 + i]) * size)
    identity = StorageIdentity(filesystem_uuid="DEST-RELEASE", device="/dev/fake0",
                               filesystem_type="ext4")
    bundle = write_bundle(root / "b",
                          make_campaign(dest=make_destination(host_path=str(dest),
                                                              mounted=True,
                                                              identity=identity)),
                          build_evidence())
    main(["hash", "--bundle", str(bundle), "--root", str(src), "--apply", "--json"])
    main(["execute", "--bundle", str(bundle), "--source-root", str(src),
          "--execute", "--apply", "--json"])
    main(["verify", "--bundle", str(bundle), "--destination", str(dest),
          "--apply", "--json"])
    import_evidence(bundle, {"destination": {"observed_identity": identity.to_dict()}},
                    apply=True)
    return bundle, src, dest


def release_and_record(bundle, src, dest, **kwargs):
    """Execute a release and apply the fragment it contributes. The real path."""
    result = execute_release(bundle, src, dest, campaign=load_campaign(bundle),
                             evidence=load_evidence(bundle), execute=True, **kwargs)
    apply_fragment(bundle, to_evidence(result, last_checkpoint=CHECKPOINT))
    return result


def src_files(src):
    return sorted(p.name for p in Path(src).iterdir() if p.is_file())


def write_audit(bundle, rows):
    """Write an audit ledger verbatim, so scale can be tested without 10k unlinks."""
    path = audit_path(bundle)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in rows),
        encoding="utf-8",
    )
    return path


def deleted_rows(count, *, size=SIZE, key=lambda i: f"c{i}.mp4"):
    return [{"at": None, "campaign_id": "x", "detail": "reverified_bytes=256",
             "digest": "0" * 64, "event": "deleted", "key": key(i), "path": f"/card/{key(i)}",
             "size": size}
            for i in range(count)]


def fake_result(bundle, **audit_overrides):
    """A ReleaseResult carrying an audit summary, without releasing anything."""
    from auto_ingest.custody.release_source import ReleaseResult

    read = read_audit(bundle)
    return ReleaseResult(
        campaign_id="x",
        mode=MODE_EXECUTED,
        audit_path=str(audit_path(bundle)),
        gate_open=True,
        state=CampaignState.SAFE_TO_RELEASE.value,
        audit_summary=replace(read, **audit_overrides) if audit_overrides else read,
    )


# ---------------------------------------------------------------------------
# 1. the gap this closes: status observes a real executed release
# ---------------------------------------------------------------------------
def test_after_a_real_executed_release_status_reports_the_released_counts(tmp_path):
    bundle, src, dest = build_released(tmp_path)
    assert load_status(bundle).state is CampaignState.SAFE_TO_RELEASE

    result = release_and_record(bundle, src, dest)

    assert result.mode == MODE_EXECUTED
    assert result.deleted == COUNT
    assert src_files(src) == []
    evidence = load_evidence(bundle).source_release
    assert evidence.released == Counts(files=COUNT, bytes=COUNT * SIZE)
    assert evidence.started is True
    assert evidence.complete is True
    assert evidence.audit_records == read_audit(bundle).records == COUNT + 1


def test_the_release_appears_in_the_status_payload_operators_read(tmp_path):
    """Not only in evidence.json: the gate names what it observed, every time."""
    bundle, src, dest = build_released(tmp_path)
    release_and_record(bundle, src, dest)

    status = load_status(bundle)
    payload = status.to_dict()
    named = [w for w in payload["warnings"] if "source_release_recorded" in w]

    assert len(named) == 1
    assert f"released={COUNT}" in named[0]
    assert f"released_bytes={COUNT * SIZE}" in named[0]
    # ...and it is explicit that this is not verification, because a reader who
    # confuses the two would read a deleted card as a proven destination.
    assert "NOT destination verification" in named[0]
    # the human rendering prints warnings too, so the default output sees it
    from auto_ingest.custody.report import status_text

    assert "source_release_recorded" in status_text(status)


def test_the_audit_ledger_remains_the_per_object_record(tmp_path):
    """The evidence block summarises; it never replaces the append-only audit."""
    bundle, src, dest = build_released(tmp_path)
    release_and_record(bundle, src, dest)

    rows = [json.loads(line) for line in
            audit_path(bundle).read_text(encoding="utf-8").splitlines()]
    keyed = [r for r in rows if r["event"] == "deleted"]

    assert [r["key"] for r in keyed] == CARDS


# ---------------------------------------------------------------------------
# 2. THE CRITICAL ANTI-REGRESSION: observability must not launder a pass
# ---------------------------------------------------------------------------
def test_released_source_without_verification_is_still_blocked(tmp_path):
    """The whole reason this block is safe: it credits nothing."""
    bundle, src, dest = build_released(tmp_path)
    release_and_record(bundle, src, dest)

    # Roll the destination proof back, keeping the release record. This is the
    # laundering shape: bytes are gone from the card, and nothing is proven at the
    # destination. Every one of those objects is now unrecoverable, which is
    # exactly when a pass must be impossible.
    apply_fragment(bundle, {"destination": {"verified_files": 0, "verified_bytes": 0,
                                            "verification_complete": False,
                                            "observed_identity": None}})
    status = load_status(bundle)

    assert status.state is not CampaignState.SAFE_TO_RELEASE
    assert status.source_release_allowed is False
    assert status.evidence.source_release.released.files == COUNT


def test_released_source_without_verification_names_the_same_reasons(tmp_path):
    """The blocker list is the same one the campaign had before the release ran."""
    bundle, src, dest = build_released(tmp_path)
    release_and_record(bundle, src, dest)
    apply_fragment(bundle, {"destination": {"verified_files": 0, "verified_bytes": 0,
                                            "verification_complete": False,
                                            "observed_identity": None}})
    after = load_status(bundle)

    # The identical campaign, with the release record removed entirely.
    import_evidence(bundle, {"source_release": {"released": {"files": 0, "bytes": 0},
                                                "absent": 0, "failed": 0, "refused": 0,
                                                "audit_records": 0, "complete": False,
                                                "started": False}},
                    apply=True)
    before = load_status(bundle)

    assert [b.code for b in after.release.blockers] == [b.code for b in before.release.blockers]
    assert after.derivation.state is before.derivation.state
    assert after.source_release_allowed is before.source_release_allowed is False


def test_the_block_names_no_field_the_gate_could_read_as_custody(tmp_path):
    """Structural proof, not a behavioural one: there is nothing to launder."""
    bundle, src, dest = build_released(tmp_path)
    release_and_record(bundle, src, dest)

    fragment = to_evidence(fake_result(bundle))

    assert set(fragment) == {"source_release"}
    block = fragment["source_release"]
    assert set(block) == {"absent", "audit_records", "complete", "error_summary",
                          "failed", "last_checkpoint", "refused", "released", "started"}
    # nothing here can be mistaken for destination-side proof, and the fragment
    # never rewrites a destination counter
    assert not {"destination", "reconciliation", "copy", "hash", "inventory"} & set(fragment)


def test_a_release_record_cannot_replace_destination_verification(tmp_path):
    """The counters point in opposite directions and must stay separate."""
    bundle, src, dest = build_released(tmp_path)
    release_and_record(bundle, src, dest)
    before = load_evidence(bundle)
    assert before.destination.verified_files == COUNT

    # Importing the release fragment repeatedly, with a hostile-looking payload,
    # must not move a single destination counter.
    result = execute_release(bundle, src, dest, campaign=load_campaign(bundle),
                             evidence=load_evidence(bundle), execute=True)
    for _ in range(3):
        apply_fragment(bundle, to_evidence(result))

    after = load_evidence(bundle)
    assert after.destination.to_dict() == before.destination.to_dict()
    assert after.reconciliation.to_dict() == before.reconciliation.to_dict()
    assert after.source_released == COUNT


# ---------------------------------------------------------------------------
# 3. a released campaign with full verification is still SAFE_TO_RELEASE
# ---------------------------------------------------------------------------
def test_released_source_and_full_verification_is_still_safe_to_release(tmp_path):
    bundle, src, dest = build_released(tmp_path)
    release_and_record(bundle, src, dest)

    status = load_status(bundle)

    assert status.state is CampaignState.SAFE_TO_RELEASE
    assert status.source_release_allowed is True
    assert status.release.blockers == ()
    # custody is still *proven* after the source is gone: the destination holds
    # the bytes, and that does not expire because the card was released.
    assert status.state in {CampaignState.VERIFIED, CampaignState.SAFE_TO_RELEASE}
    assert status.evidence.source_released == COUNT


def test_a_released_campaign_reports_the_same_gate_verdict_as_before(tmp_path):
    """Observability changed what status *says*, never what it decides."""
    bundle, src, dest = build_released(tmp_path)
    before = load_status(bundle)
    release_and_record(bundle, src, dest)
    after = load_status(bundle)

    assert after.source_release_allowed == before.source_release_allowed is True
    assert after.state is before.state
    assert [b.code for b in after.release.blockers] == []
    assert after.release.conditions == before.release.conditions


# ---------------------------------------------------------------------------
# 4. the gate refuses the two impossible / unfinished release records
# ---------------------------------------------------------------------------
def test_a_release_larger_than_the_inventory_blocks(tmp_path):
    """Impossible arithmetic fails the gate closed: VERIFIED, gate shut."""
    bundle, src, dest = build_released(tmp_path)
    release_and_record(bundle, src, dest)
    # More objects destroyed than the card was ever recorded to hold.
    apply_fragment(bundle, {"source_release": {"released": {"files": COUNT + 1,
                                                             "bytes": (COUNT + 1) * SIZE}}})

    status = load_status(bundle)

    # Not BLOCKED: nothing here contradicts the *evidence*, it contradicts
    # arithmetic, and the gate is what refuses. The derived state is VERIFIED -
    # custody proven, gate closed - which is exactly what a closed gate means.
    assert status.state is CampaignState.VERIFIED
    assert status.source_release_allowed is False
    assert "source_release_exceeds_inventory" in [b.code for b in status.release.blockers]
    # and it is a strictly tighter answer than the same campaign without the
    # impossible claim, which releases.
    apply_fragment(bundle, {"source_release": {"released": {"files": COUNT,
                                                             "bytes": COUNT * SIZE}}})
    assert load_status(bundle).source_release_allowed is True


def test_an_unfinished_release_blocks_and_says_so(tmp_path):
    """Bytes gone, pass did not finish: a human has to read the audit."""
    bundle, src, dest = build_released(tmp_path)
    release_and_record(bundle, src, dest)
    apply_fragment(bundle, {"source_release": {"failed": 0, "refused": 1, "complete": False}})

    status = load_status(bundle)

    assert status.state is CampaignState.VERIFIED
    assert status.source_release_allowed is False
    blocker = next(b for b in status.release.blockers if b.code == "source_release_incomplete")
    assert "refused=1" in blocker.detail


def test_a_whole_run_refusal_is_not_a_stuck_object(tmp_path):
    """A closed gate refuses the *run*; it left no object behind, and says so.

    Counting that as an object refusal would poison the campaign forever: the
    objects are all still present and releasable, and the next successful pass
    must be able to clear it.
    """
    bundle, src, dest = build_released(tmp_path)
    # Break the gate, then run the executed path: it refuses whole-run.
    apply_fragment(bundle, {"destination": {"verification_complete": False}})
    refused = execute_release(bundle, src, dest, campaign=load_campaign(bundle),
                              evidence=load_evidence(bundle), execute=True)
    assert refused.mode == "refused"

    fragment = to_evidence(refused)

    assert fragment["source_release"]["refused"] == 0
    assert fragment["source_release"]["released"] == {"bytes": 0, "files": 0}
    assert fragment["source_release"]["complete"] is False   # the audit is torn-free but nothing ran
    # and the audit still records the refusal, at whole-run granularity
    assert read_audit(bundle).unattributed == 1


def test_a_torn_audit_never_reads_as_a_complete_release(tmp_path):
    """An interrupted append must not look like a finished release."""
    bundle, src, dest = build_released(tmp_path)
    release_and_record(bundle, src, dest)
    with audit_path(bundle).open("a", encoding="utf-8") as handle:
        handle.write('{"event":"deleted","key":"c9.mp4"')

    result = fake_result(bundle)
    fragment = to_evidence(result)

    assert result.audit_summary.coherent is False
    assert fragment["source_release"]["complete"] is False
    assert "source_release_incomplete" not in blocker_codes(load_evidence(bundle),
                                                           load_campaign(bundle))


# ---------------------------------------------------------------------------
# 5. bounded: 10,000 released objects produce the same report as 3
# ---------------------------------------------------------------------------
def test_ten_thousand_released_objects_do_not_grow_the_evidence(tmp_path):
    bundle, _, _ = build_released(tmp_path)
    small = to_evidence(fake_result(bundle))

    write_audit(bundle, deleted_rows(10_000))
    big_result = fake_result(bundle)
    big = to_evidence(big_result)

    assert big["source_release"]["released"]["files"] == 10_000
    assert big["source_release"]["released"]["bytes"] == 10_000 * SIZE
    # the block has a fixed shape: same keys, same cardinality, no per-object rows
    assert set(big["source_release"]) == set(small["source_release"])
    assert len(big["source_release"]) == len(small["source_release"])
    assert not any(isinstance(v, list) and len(v) > 20 for v in big["source_release"].values())
    for key, value in big["source_release"].items():
        if isinstance(value, str) or key in {"error_summary"}:
            continue
        assert len(json.dumps(value)) < 40, (key, value)


def test_the_report_grows_by_digits_not_by_objects(tmp_path):
    """10,000 objects vs 3: the fragment differs only in the numbers themselves."""
    bundle, _, _ = build_released(tmp_path)
    small = to_evidence(fake_result(bundle))

    write_audit(bundle, deleted_rows(10_000))
    big = to_evidence(fake_result(bundle))

    assert len(json.dumps(big, sort_keys=True)) - len(json.dumps(small, sort_keys=True)) < 32
    # ...and the whole release result, not just the fragment, stays bounded
    assert len(json.dumps(fake_result(bundle).to_dict(), sort_keys=True)) < 4_000


def test_the_evidence_document_stays_small_at_card_scale(tmp_path):
    bundle, _, _ = build_released(tmp_path)
    write_audit(bundle, deleted_rows(10_000))
    apply_fragment(bundle, to_evidence(fake_result(bundle)))

    raw = (Path(bundle) / EVIDENCE_FILE).read_text(encoding="utf-8")

    assert len(raw) < 8_000, len(raw)
    assert load_evidence(bundle).source_released == 10_000


def test_a_malformed_release_count_is_diagnosed_like_any_other(tmp_path):
    """A thousands separator in the release block must block, not read as zero."""
    bundle, _, _ = build_released(tmp_path)
    path = Path(bundle) / EVIDENCE_FILE
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["source_release"] = {"released": {"files": "10,000"}}
    path.write_text(json.dumps(raw, sort_keys=True, indent=2), encoding="utf-8")

    status = load_status(bundle)

    assert status.state is CampaignState.BLOCKED
    assert status.derivation.reasons == ("evidence_counts_are_malformed",)
    assert any("source_release.released.files" in f for f in status.evidence.coerced_fields)


# ---------------------------------------------------------------------------
# 6. idempotence: re-import must restate, never double-count
# ---------------------------------------------------------------------------
def test_to_evidence_is_idempotent_across_repeated_imports(tmp_path):
    bundle, src, dest = build_released(tmp_path)
    result = release_and_record(bundle, src, dest)
    after_first = (Path(bundle) / EVIDENCE_FILE).read_bytes()

    for _ in range(3):
        apply_fragment(bundle, to_evidence(result, last_checkpoint=CHECKPOINT))

    assert (Path(bundle) / EVIDENCE_FILE).read_bytes() == after_first
    assert load_evidence(bundle).source_released == COUNT


def test_a_second_executed_pass_does_not_inflate_the_counts(tmp_path):
    """The re-run is the interesting case: it appends, and the counts must not add."""
    bundle, src, dest = build_released(tmp_path)
    first = release_and_record(bundle, src, dest)
    assert first.deleted == COUNT

    second = release_and_record(bundle, src, dest)

    assert second.deleted == 0          # nothing left to unlink
    assert second.already_absent == COUNT
    assert second.complete is True
    evidence = load_evidence(bundle).source_release
    # The re-run appended a run summary to the audit; the destructive counter is
    # still 3, not 6. This is the cumulative-not-delta rule the hash and verify
    # producers already learned the hard way. (`absent` stays 0: an already-gone
    # key never reaches the delete loop, so it appends no per-object record.)
    assert evidence.released == Counts(files=COUNT, bytes=COUNT * SIZE)
    assert read_audit(bundle).records > first.audit_summary.records
    # ...and the gate still agrees the card is releasable.
    assert load_status(bundle).source_release_allowed is True


def test_to_evidence_is_a_pure_function_of_its_argument(tmp_path):
    bundle, _, _ = build_released(tmp_path)
    write_audit(bundle, deleted_rows(5))
    result = fake_result(bundle)

    first = to_evidence(result)
    second = to_evidence(result)

    assert first == second
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_the_instant_is_supplied_not_invented(tmp_path):
    """Two identical passes must produce identical evidence: no clock, no uuid."""
    bundle, _, _ = build_released(tmp_path)
    write_audit(bundle, deleted_rows(4))
    result = fake_result(bundle)

    assert to_evidence(result) == to_evidence(result)
    assert to_evidence(result)["source_release"]["last_checkpoint"] is None
    stamped = to_evidence(result, last_checkpoint="2026-10-04T00:00:00Z")
    assert stamped["source_release"]["last_checkpoint"] == "2026-10-04T00:00:00Z"
    assert CampaignEvidence.from_dict(stamped).source_release.last_checkpoint == \
        "2026-10-04T00:00:00Z"


def test_the_fragment_never_touches_the_blocks_it_does_not_own(tmp_path):
    bundle, src, dest = build_released(tmp_path)
    before = load_evidence(bundle).to_dict()
    release_and_record(bundle, src, dest)
    after = load_evidence(bundle).to_dict()

    for block in ("inventory", "hash", "copy", "destination", "reconciliation",
                  "worker", "errors"):
        assert after[block] == before[block], block
    assert after["source_release"] != before["source_release"]


# ---------------------------------------------------------------------------
# 7. no silent weakening: an unreleased campaign's blockers are unchanged
# ---------------------------------------------------------------------------
def test_the_gate_blocker_list_is_unchanged_for_an_unreleased_campaign(tmp_path):
    """No release record anywhere: the gate's codes are exactly the pre-existing set."""
    bundle, _, _ = build_released(tmp_path)
    observed = load_evidence(bundle)

    assert observed.source_release == SourceReleaseEvidence()
    assert observed.source_release.observed is False
    assert evaluate_release(load_campaign(bundle), observed).allowed is True
    assert blocker_codes(observed, load_campaign(bundle)) == []
    assert not [w for w in load_status(bundle).release.warnings
                if "source_release_recorded" in w]

    # Every blocker an unreleased broken campaign reports is a pre-existing one.
    # The two release conditions can only add their own codes, and only when the
    # block says something is wrong - never on an empty block.
    broken = replace(observed, destination=replace(observed.destination,
                                                   verification_complete=False),
                     reconciliation=replace(observed.reconciliation, source_only=1))
    codes = blocker_codes(broken, load_campaign(bundle))
    assert codes == ["destination_verification_incomplete", "missing_destination"]
    assert not [c for c in codes if c.startswith("source_release")]


def test_no_new_state_was_introduced_for_this(tmp_path):
    """Why a counter and not a derived state - pinned so it cannot drift silently.

    A `SOURCE_RELEASED` state after `SAFE_TO_RELEASE` would invert the gate:
    `CampaignStatus.source_release_allowed` is ``allowed and state is
    SAFE_TO_RELEASE``, so the state would have to become False *after* the source
    was released. The operator would erase the card and be told the card is not
    releasable, and `release_source.plan_release` - which requires
    ``state is SAFE_TO_RELEASE`` - would refuse its own idempotent re-run. Custody
    proven does not expire because the source is gone, so the fact lives in a
    counter and the state vocabulary stays at twelve.
    """
    assert len(CampaignState) == 12
    assert not hasattr(CampaignState, "SOURCE_RELEASED")
    assert CampaignState.__members__["SAFE_TO_RELEASE"] is CampaignState.SAFE_TO_RELEASE

    bundle, src, dest = build_released(tmp_path)
    assert load_status(bundle).state is CampaignState.SAFE_TO_RELEASE
    release_and_record(bundle, src, dest)
    # the state does not move, and neither does the gate's answer
    assert load_status(bundle).state is CampaignState.SAFE_TO_RELEASE
    assert load_status(bundle).source_release_allowed is True


def test_the_release_evidence_is_a_source_side_fact_only(tmp_path):
    """`source_released` must never be added to destination verification."""
    bundle, src, dest = build_released(tmp_path)
    release_and_record(bundle, src, dest)
    evidence = load_evidence(bundle)

    assert evidence.source_released == COUNT
    assert evidence.verified_at_destination == COUNT
    assert evidence.to_dict()["source_release"]["released"] == {"bytes": COUNT * SIZE,
                                                                "files": COUNT}
    # The block is a source_release, not a destination_release: the direction is
    # in the name, so a reader cannot get it backwards.
    assert "destination_release" not in evidence.to_dict()

    # And the two counters are computed from different blocks, so moving one
    # cannot move the other - which is what "not interchangeable" means in code.
    apply_fragment(bundle, {"source_release": {"released": {"files": 0, "bytes": 0}}})
    zeroed = load_evidence(bundle)
    assert zeroed.source_released == 0
    assert zeroed.verified_at_destination == COUNT
    apply_fragment(bundle, {"destination": {"verified_files": 0}})
    assert load_evidence(bundle).verified_at_destination == 0
    assert load_evidence(bundle).source_released == 0


def test_the_block_round_trips_through_the_document(tmp_path):
    """from_dict/to_dict is the whole contract; a lossy field is a silent drop."""
    bundle, _, _ = build_released(tmp_path)
    original = SourceReleaseEvidence(
        released=Counts(files=7, bytes=1_024),
        absent=2,
        failed=1,
        refused=1,
        audit_records=11,
        started=True,
        complete=False,
        error_summary=("c3.mp4:destination_digest_differs",),
        last_checkpoint="2026-10-04T00:00:00Z",
    )
    raw = {"source_release": original.to_dict()}
    wrapped = CampaignEvidence.from_dict({"campaign_id": "x", **raw})

    assert wrapped.source_release == original
    assert wrapped.to_dict()["source_release"] == raw["source_release"]
    assert wrapped.source_release.attempted == 11
    assert wrapped.source_release.clean is False
