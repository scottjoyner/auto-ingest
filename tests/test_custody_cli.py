"""`auto-ingest custody ...` CLI: read-only by default, human and JSON modes."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import pytest
from custody_helpers import (
    CARD_01_BUNDLE,
    campaign,
    copying,
    destination_evidence,
    evidence,
    fully_copied_campaign,
    hashing,
    inventory,
    reconciliation,
    strict_policy,
    worker,
    write_bundle,
)

from auto_ingest.custody.cli import (
    EXIT_GATE_CLOSED,
    EXIT_OK,
    EXIT_USAGE,
    build_parser,
    main,
)
from auto_ingest.custody.store import STATUS_SCHEMA, load_status

# The source end of a campaign is a fact this module controls, not a fact
# about whether the developer's card happens to be plugged in.
pytestmark = pytest.mark.usefixtures("hermetic_mounts")
def run(argv, capsys):
    code = main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------
def test_parser_exposes_the_documented_subcommands():
    parser = build_parser()
    sub = [a for a in parser._subparsers._group_actions if hasattr(a, "choices")][0]
    assert set(sub.choices) == {
        "status", "plan", "verify", "import", "new", "reconcile",
        "observe-mount", "capacity", "preflight", "hash", "execute",
        "release-source", "stage",
    }


def test_status_defaults_are_read_only():
    args = build_parser().parse_args(["status", "--bundle", str(CARD_01_BUNDLE)])
    assert args.json is False
    assert args.require_release is False
    assert not hasattr(args, "apply")


def test_release_source_is_off_without_the_flag():
    """The one irreversible command must default to describing, not doing."""
    args = build_parser().parse_args(["release-source", "--bundle", "x"])
    assert args.execute is False


def test_import_does_not_apply_without_the_flag():
    args = build_parser().parse_args(["import", "--bundle", "x", "--evidence", "y"])
    assert args.apply is False


def test_new_does_not_apply_without_the_flag():
    args = build_parser().parse_args(["new", "--bundle", "x", "--card-id", "CARD-02",
                                      "--uuid", "ABCD-1234"])
    assert args.apply is False


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------
def test_status_human_output(capsys):
    code, out, _ = run(["status", "--bundle", str(CARD_01_BUNDLE)], capsys)
    assert code == EXIT_OK
    assert "state              RECONCILE_REQUIRED" in out
    assert "next_safe_action   reconcile_destination" in out
    assert "source_release_allowed     false" in out
    assert "source_mutation_allowed    false" in out
    assert "source_deletion_allowed    false" in out


def test_status_json_output_matches_the_documented_contract(capsys):
    code, out, _ = run(["status", "--bundle", str(CARD_01_BUNDLE), "--json"], capsys)
    assert code == EXIT_OK
    data = json.loads(out)
    assert data["schema"] == STATUS_SCHEMA
    assert data["campaign_id"] == "sdcard-9f1c4a2b7d3e5a60-20260412"
    assert data["state"] == "RECONCILE_REQUIRED"
    assert data["source_read_only"] is True
    assert data["hash"]["verified"] == 67644
    assert data["copy"]["complete"] is False
    assert data["destination"]["verified_files"] == 0
    assert data["destination"]["verified_bytes"] == 0
    assert data["source_release_allowed"] is False
    assert data["next_safe_action"] == "reconcile_destination"


def test_status_json_is_reproducible(capsys):
    _, first, _ = run(["status", "--bundle", str(CARD_01_BUNDLE), "--json"], capsys)
    _, second, _ = run(["status", "--bundle", str(CARD_01_BUNDLE), "--json"], capsys)
    assert first == second


def test_status_require_release_exits_nonzero_when_the_gate_is_closed(capsys):
    code, _, _ = run(
        ["status", "--bundle", str(CARD_01_BUNDLE), "--require-release"], capsys
    )
    assert code == EXIT_GATE_CLOSED


def test_status_require_release_passes_for_a_fully_custodied_campaign(tmp_path, capsys):
    camp, ev = fully_copied_campaign()
    bundle = write_bundle(tmp_path / "ok", camp, ev)
    code, out, _ = run(["status", "--bundle", str(bundle), "--require-release"], capsys)
    assert code == EXIT_OK
    assert "SAFE_TO_RELEASE" in out


def test_status_reports_missing_bundle_cleanly(tmp_path, capsys):
    code, _, err = run(["status", "--bundle", str(tmp_path / "nope")], capsys)
    assert code == EXIT_USAGE
    assert "campaign" in err.lower()


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------
def test_plan_json_is_a_description_not_an_execution(capsys):
    code, out, _ = run(["plan", "--bundle", str(CARD_01_BUNDLE), "--json"], capsys)
    assert code == EXIT_OK
    data = json.loads(out)
    assert data["current_state"] == "RECONCILE_REQUIRED"
    assert data["source_mutation_allowed"] is False
    assert data["source_deletion_allowed"] is False
    assert data["requires_operator_authorization"] is True
    assert data["actions"][0]["operation"] == "verify_existing_destination"
    assert data["actions"][0]["mutates_destination"] is False


def test_plan_human_output_orders_verification_before_copying(capsys):
    _, out, _ = run(["plan", "--bundle", str(CARD_01_BUNDLE)], capsys)
    verify_at = out.index("verify_existing_destination")
    copy_at = out.index("copy_objects")
    assert verify_at < copy_at


def test_plan_is_idempotent_through_the_cli(capsys):
    _, first, _ = run(["plan", "--bundle", str(CARD_01_BUNDLE), "--json"], capsys)
    run(["status", "--bundle", str(CARD_01_BUNDLE)], capsys)
    _, second, _ = run(["plan", "--bundle", str(CARD_01_BUNDLE), "--json"], capsys)
    assert first == second


def test_reconcile_campaign_with_verified_subset_skips_verified(capsys, tmp_path):
    camp = campaign()
    ev = evidence(
        inv=inventory(1000, 10_000, complete=True, verified=True),
        hsh=hashing(1000, verified_bytes=10_000, complete=True),
        cpy=copying(planned_files=1000, planned_bytes=10_000, started=True,
                    result_complete=False, interrupted=True),
        dst=destination_evidence(verified_files=800, verified_bytes=8000,
                                 verification_started=True,
                                 unverified_present_files=50),
        rec=reconciliation(source_only=200),
        wkr=worker(status="stopped"),
    )
    bundle = write_bundle(tmp_path / "subset", camp, ev)
    _, out, _ = run(["plan", "--bundle", str(bundle), "--json"], capsys)
    actions = json.loads(out)["actions"]
    copy_action = next(a for a in actions if a["operation"] == "copy_objects")
    verify_existing = next(a for a in actions
                           if a["operation"] == "verify_existing_destination")
    # 1000 inventoried, 800 verified, 50 present-but-unattested:
    # 200 outstanding, of which 50 need verifying rather than copying.
    assert verify_existing["estimated_files"] == 50
    assert copy_action["estimated_files"] == 150
    assert copy_action["excludes_verified"] is True


# ---------------------------------------------------------------------------
# verify (describe only)
# ---------------------------------------------------------------------------
def test_verify_describes_and_never_executes(capsys):
    code, out, _ = run(["verify", "--bundle", str(CARD_01_BUNDLE), "--json"], capsys)
    assert code == EXIT_OK
    data = json.loads(out)
    assert data["executed"] is False
    assert data["requires_operator_authorization"] is True
    assert data["source_mutation_allowed"] is False
    assert data["source_deletion_allowed"] is False
    assert data["objects_to_verify"] == 0


def test_verify_execute_flag_is_refused(capsys):
    code, _, err = run(["verify", "--bundle", str(CARD_01_BUNDLE), "--execute"], capsys)
    assert code == EXIT_GATE_CLOSED
    assert "refusing to execute" in err


# ---------------------------------------------------------------------------
# import
# ---------------------------------------------------------------------------
def test_import_validates_without_writing(tmp_path, capsys):
    bundle = write_bundle(tmp_path / "b", campaign(), evidence())
    doc = tmp_path / "new-evidence.json"
    doc.write_text(json.dumps({"hash": {"verified_files": 5}}), encoding="utf-8")
    code, out, _ = run(
        ["import", "--bundle", str(bundle), "--evidence", str(doc), "--json"], capsys
    )
    assert code == EXIT_OK
    data = json.loads(out)
    assert data["mode"] == "validate_only"
    assert data["applied"] is False
    assert (bundle / "evidence.json").read_text(encoding="utf-8") == \
        json.dumps(evidence().to_dict(), sort_keys=True, indent=2)


def test_import_apply_writes_the_evidence(tmp_path, capsys):
    bundle = write_bundle(tmp_path / "b", campaign(), evidence())
    doc = tmp_path / "new-evidence.json"
    payload = {
        "campaign_id": campaign().campaign_id,
        "inventory": {"discovered_files": 7, "discovered_bytes": 70, "complete": True},
        "hash": {"verified_files": 7, "verified_bytes": 70, "complete": True},
    }
    doc.write_text(json.dumps(payload), encoding="utf-8")
    code, out, _ = run(
        ["import", "--bundle", str(bundle), "--evidence", str(doc), "--apply", "--json"],
        capsys,
    )
    assert code == EXIT_OK
    assert json.loads(out)["applied"] is True
    written = json.loads((bundle / "evidence.json").read_text(encoding="utf-8"))
    assert written["hash"]["verified_files"] == 7


def test_import_rejects_a_mismatched_campaign_id(tmp_path, capsys):
    bundle = write_bundle(tmp_path / "b", campaign(), evidence())
    doc = tmp_path / "e.json"
    doc.write_text(json.dumps({"campaign_id": "sdcard-somebody-else"}), encoding="utf-8")
    code, out, _ = run(
        ["import", "--bundle", str(bundle), "--evidence", str(doc), "--apply", "--json"],
        capsys,
    )
    assert code == EXIT_USAGE
    assert "does not match" in json.loads(out)["error"]


def test_import_reports_ignored_declared_state(tmp_path, capsys):
    bundle = write_bundle(tmp_path / "b", campaign(), evidence())
    doc = tmp_path / "e.json"
    doc.write_text(json.dumps({"state": "SAFE_TO_RELEASE"}), encoding="utf-8")
    _, out, _ = run(
        ["import", "--bundle", str(bundle), "--evidence", str(doc), "--json"], capsys
    )
    data = json.loads(out)
    assert data["declared_state_keys_ignored"] == ["state"]
    assert data["source_release_allowed"] is False


# ---------------------------------------------------------------------------
# identity cross-check from the CLI
# ---------------------------------------------------------------------------
def test_status_flags_a_different_card_at_the_same_mount(capsys):
    code, out, _ = run(
        ["status", "--bundle", str(CARD_01_BUNDLE), "--json",
         "--observed-uuid", "FFFF-9999", "--observed-device", "/dev/sdc1",
         "--observed-label", "UNTITLED"],
        capsys,
    )
    assert code == EXIT_OK
    data = json.loads(out)
    assert data["observed_card_matches_campaign"] is False


def test_card_mismatch_says_what_to_do(capsys):
    """A bare `false` is not actionable; the operator needs the remedy."""
    _, out, _ = run(
        ["status", "--bundle", str(CARD_01_BUNDLE),
         "--observed-uuid", "FFFF-9999", "--observed-device", "/dev/sdc1"],
        capsys,
    )
    assert "CARD MISMATCH   different_card: filesystem_uuid" in out
    assert "custody new" in out
    assert "do not resume this campaign" in out


def test_card_mismatch_is_absent_when_hardware_matches(capsys):
    _, out, _ = run(
        ["status", "--bundle", str(CARD_01_BUNDLE), "--json",
         "--observed-uuid", "7A3E-2C19", "--observed-device", "/dev/sdc9"],
        capsys,
    )
    assert json.loads(out)["observed_card_conflict"] is None


def test_unprovable_identity_is_distinguished_from_a_different_card(capsys):
    """Recorded a UUID, observed none: not "same", and not "provably different"."""
    _, out, _ = run(
        ["status", "--bundle", str(CARD_01_BUNDLE), "--json",
         "--observed-device", "/dev/sdb1"],
        capsys,
    )
    conflict = json.loads(out)["observed_card_conflict"]
    assert conflict["kind"] == "identity_unprovable"
    assert conflict["different_hardware"] is False
    assert "filesystem_uuid" in conflict["unprovable_fields"]


def test_no_observed_hardware_means_no_conflict_claim(capsys):
    _, out, _ = run(["status", "--bundle", str(CARD_01_BUNDLE), "--json"], capsys)
    data = json.loads(out)
    assert data["observed_card_matches_campaign"] is None
    assert data["observed_card_conflict"] is None


# ---------------------------------------------------------------------------
# the default mode is human: a diagnostic only in --json is invisible
# ---------------------------------------------------------------------------
def test_malformed_counts_are_visible_without_json(tmp_path):
    """`status` defaults to human output, so that is where a fault must show."""
    from auto_ingest.custody import CampaignEvidence
    from auto_ingest.custody.report import status_text

    bundle = write_bundle(tmp_path / "b", campaign(), evidence())
    bad = CampaignEvidence.from_dict({
        "inventory": {"discovered_files": "6,764", "discovered_bytes": 1,
                      "complete": True},
        "hash": {"verified_files": 67644, "verified_bytes": 1, "complete": True},
    }, strict_policy())
    (bundle / "evidence.json").write_text(json.dumps(bad.to_dict()), encoding="utf-8")
    text = status_text(load_status(bundle, strict_policy()))
    assert "MALFORMED COUNT" in text
    assert "6,764" in text
    assert "evidence_counts_are_malformed" in text


def test_the_malformed_count_diagnostic_survives_a_round_trip(tmp_path):
    """`import --apply` writes normalised evidence and reads it back.

    Without rehydration the record of the malformation would be dropped by the
    very operation an operator runs to inspect it, and they would be told only
    that evidence "contradicts itself".
    """
    from auto_ingest.custody import CampaignEvidence
    from auto_ingest.custody.store import load_status

    bundle = write_bundle(tmp_path / "b", campaign(), evidence())
    doc = tmp_path / "e.json"
    doc.write_text(json.dumps({
        "inventory": {"discovered_files": "6,764", "discovered_bytes": 1,
                      "complete": True},
        "hash": {"verified_files": 67644, "verified_bytes": 1, "complete": True},
    }), encoding="utf-8")

    code = main(["import", "--bundle", str(bundle), "--evidence", str(doc),
                 "--apply", "--json"])
    assert code == EXIT_OK

    stored = json.loads((bundle / "evidence.json").read_text(encoding="utf-8"))
    assert stored["coerced_fields"] == ["inventory.discovered_files='6,764'"]

    reread = CampaignEvidence.from_dict(stored, strict_policy())
    assert reread.coerced_fields == ("inventory.discovered_files='6,764'",)
    status = load_status(bundle, strict_policy())
    assert status.derivation.reasons == ("evidence_counts_are_malformed",)


def test_ignored_declared_state_keys_are_visible_without_json(tmp_path):
    from auto_ingest.custody.report import status_text

    bundle = write_bundle(tmp_path / "b", campaign(), evidence())
    raw = json.loads((bundle / "evidence.json").read_text(encoding="utf-8"))
    raw["state"] = "SAFE_TO_RELEASE"
    raw["ignored_declared_fields"] = ["state"]
    (bundle / "evidence.json").write_text(json.dumps(raw), encoding="utf-8")
    text = status_text(load_status(bundle, strict_policy()))
    assert "IGNORED" in text
    assert "state" in text


def test_status_accepts_the_same_card(capsys):
    _, out, _ = run(
        ["status", "--bundle", str(CARD_01_BUNDLE), "--json",
         "--observed-uuid", "7A3E-2C19", "--observed-device", "/dev/sdc9"],
        capsys,
    )
    assert json.loads(out)["observed_card_matches_campaign"] is True


@pytest.mark.parametrize("cmd", ["status", "plan", "verify", "new"])
def test_no_subcommand_writes_by_default(cmd, capsys, tmp_path):
    """Even `new`, pointed at an existing bundle, writes nothing."""
    before = sorted(p.name for p in CARD_01_BUNDLE.iterdir())
    argv = [cmd, "--bundle", str(CARD_01_BUNDLE)]
    if cmd == "new":
        argv += ["--card-id", "CARD-XX", "--uuid", "ABCD-1234"]
    run(argv, capsys)
    assert sorted(p.name for p in CARD_01_BUNDLE.iterdir()) == before


def test_status_writes_nothing_even_with_a_missing_policy_override(capsys):
    before = sorted(p.name for p in CARD_01_BUNDLE.iterdir())
    code, _, _ = run(["status", "--bundle", str(CARD_01_BUNDLE),
                      "--observed-uuid", "7A3E-2C19"], capsys)
    assert code == EXIT_OK
    assert sorted(p.name for p in CARD_01_BUNDLE.iterdir()) == before


# ---------------------------------------------------------------------------
# stage: classify -> hash the in-scope set -> report
# ---------------------------------------------------------------------------
# The bug this section exists for: `staging` produces a key -> DESTINATION map,
# and `hash_source` wants key -> SOURCE path. Passing the map hashed nothing at
# all and still exited 0 - a green run over zero objects. Every test here checks
# the ledger has the objects in it, not merely that the command returned.

CARD_TREE = {
    # in scope, and stageable
    "DCIM/Movie/2026_0829_123850_F.MP4": b"F" * 512,
    "DCIM/Movie/2026_0829_123850_R.MP4": b"R" * 512,
    "DCIM/Movie/2026_0829_124157_F.MP4": b"F2" * 512,
    "yolo/2026_0829_123850_F_YOLOv8n.csv": b"k,v\n" * 4,
    "yolo/2026_0829_124157_F_YOLOv8n.csv": b"k,v\n" * 4,
    # a per-recording sidecar: must keep its suffix to stay discoverable
    "transcripts/2026_0829_123850_medium_transcription.txt": b"words\n",
    # out of scope entirely
    "overland/locations_0001.json": b"{}" * 32,
    "heatmap/2026_0829_123850_F.png": b"\x89PNG" * 8,
    ".Trashes/._junk": b"junk",
}


def _card(tmp_path):
    root = tmp_path / "card"
    for rel, data in CARD_TREE.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    return root


def _bundle(tmp_path):
    """A real bundle: hash_source reads campaign metadata, so staging cannot run
    against a bare directory."""
    return str(write_bundle(tmp_path / "b", campaign(), evidence()))


def _policy_file(tmp_path):
    f = tmp_path / "policy.json"
    f.write_text(json.dumps({"declared_source_exclusions": [".Trashes"]}))
    return f


def test_stage_hashes_exactly_the_objects_it_stages(tmp_path, capsys):
    """The load-bearing assertion: real ledger records exist for real bytes."""
    root = _card(tmp_path)
    code, out, _ = run(["stage", "--bundle", _bundle(tmp_path), "--root", str(root),
                        "--policy-file", str(_policy_file(tmp_path)),
                        "--apply", "--json"], capsys)
    assert code == EXIT_OK, out
    payload = json.loads(out)
    assert payload["hashed"] == payload["staged"], payload
    assert payload["hashed"] > 0

    rows = [json.loads(line) for line
            in (tmp_path / "b" / "ledgers" / "hash.jsonl").read_text().splitlines()]
    keys = {r["key"] for r in rows}
    assert "DCIM/Movie/2026_0829_123850_F.MP4" in keys
    assert "yolo/2026_0829_123850_F_YOLOv8n.csv" in keys
    assert sum(r["size"] for r in rows) > 0


def test_stage_never_puts_out_of_scope_objects_in_the_ledger(tmp_path, capsys):
    """Hashing the whole walk would hand plan_copy a Python package to copy."""
    root = _card(tmp_path)
    code, out, _ = run(["stage", "--bundle", _bundle(tmp_path), "--root", str(root),
                        "--policy-file", str(_policy_file(tmp_path)), "--json"], capsys)
    assert code == EXIT_OK, out
    rows = [json.loads(line) for line
            in (tmp_path / "b" / "ledgers" / "hash.jsonl").read_text().splitlines()]
    keys = {r["key"] for r in rows}
    assert not any(k.startswith("overland/") for k in keys)
    assert not any(k.startswith("heatmap/") for k in keys)
    assert not any(k.startswith(".Trashes/") for k in keys)
    payload = json.loads(out)
    assert payload["excluded_by_policy"] == 1
    assert payload["unstaged"] == 2  # the overland json and the heatmap png


def test_stage_does_not_modify_the_source(tmp_path, capsys):
    root = _card(tmp_path)
    before = {rel: (root / rel).read_bytes() for rel in CARD_TREE}
    before_paths = sorted(p.relative_to(root).as_posix() for p in root.rglob("*"))
    run(["stage", "--bundle", _bundle(tmp_path), "--root", str(root),
         "--policy-file", str(_policy_file(tmp_path)), "--apply", "--json"], capsys)
    assert sorted(p.relative_to(root).as_posix() for p in root.rglob("*")) == before_paths
    for rel, data in before.items():
        assert (root / rel).read_bytes() == data


def test_stage_records_what_it_declined_to_stage_and_why(tmp_path, capsys):
    """Silence here would read as "nothing was skipped"."""
    root = _card(tmp_path)
    _, out, _ = run(["stage", "--bundle", _bundle(tmp_path), "--root", str(root),
                     "--policy-file", str(_policy_file(tmp_path)), "--json"], capsys)
    reasons = json.loads(out)["unstaged_by_reason"]
    # Scope is reported ahead of the key: a Python file is not pipeline content,
    # whatever its name, and saying "no timestamp" would be a false problem.
    assert reasons == {"out_of_scope:other": 2}


def test_stage_is_the_layout_contract_not_a_copy(tmp_path, capsys):
    """stage lays out and hashes. Copying is `execute`, deliberately."""
    root = _card(tmp_path)
    dest = tmp_path / "dest"
    dest.mkdir()
    _, out, _ = run(["stage", "--bundle", _bundle(tmp_path), "--root", str(root),
                     "--policy-file", str(_policy_file(tmp_path)), "--json"], capsys)
    assert list(dest.iterdir()) == [], "stage must not write to the destination"
    assert not (tmp_path / "b" / "ledgers" / "copy.jsonl").exists()
    assert json.loads(out)["staged"] > 0


def test_stage_media_only_drops_derived_sidecars(tmp_path, capsys):
    root = _card(tmp_path)
    _, out, _ = run(["stage", "--bundle", _bundle(tmp_path), "--root", str(root),
                     "--policy-file", str(_policy_file(tmp_path)),
                     "--media-only", "--json"], capsys)
    payload = json.loads(out)
    assert payload["by_role"] == {"video": 3}, payload["by_role"]
    assert "detection_csv" not in payload["by_role"]


def test_stage_refuses_to_continue_when_two_objects_claim_one_path(
        tmp_path, capsys, monkeypatch):
    """A collision means the layout is wrong. Picking a winner loses a file."""
    from auto_ingest.custody import staging

    root = _card(tmp_path)
    # Two different source keys that stage onto the same destination path.
    monkeypatch.setattr(
        staging, "staged_filename",
        lambda stem, suffix, role, **kw: ("2026/08/29/same.mp4" if role == "video"
                                          else f"{stem}{suffix}"))
    code, _, err = run(["stage", "--bundle", _bundle(tmp_path), "--root", str(root),
                        "--policy-file", str(_policy_file(tmp_path)),
                        "--json"], capsys)
    assert code == EXIT_GATE_CLOSED
    assert "more than one source object" in err


def test_stage_reports_an_empty_source_without_pretending_to_work(
        tmp_path, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    code, out, _ = run(["stage", "--bundle", _bundle(tmp_path), "--root", str(empty),
                        "--policy-file", str(_policy_file(tmp_path)),
                        "--json"], capsys)
    assert code == EXIT_OK
    payload = json.loads(out)
    assert payload["walked"] == 0 and payload["staged"] == 0


# ---------------------------------------------------------------------------
# stage decides the layout; execute obeys the record
# ---------------------------------------------------------------------------
# The two commands must not each hold their own idea of where a file goes. If
# they disagreed, the copy ledger would name one path while the bytes sat at
# another, and reconciliation - which joins on the source key - could not tell.

def _stage(root, bundle, tmp_path):
    from auto_ingest.custody.cli import main as _main

    o, e = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(o), contextlib.redirect_stderr(e):
        code = _main(["stage", "--bundle", bundle, "--root", str(root),
                      "--policy-file", str(_policy_file(tmp_path)), "--apply",
                      "--json"])
    assert code == EXIT_OK, e.getvalue()
    return json.loads(o.getvalue())


def test_stage_records_the_layout_in_the_bundle(tmp_path):
    from auto_ingest.custody.ledger import read_staged_ledger

    root = _card(tmp_path)
    bundle = _bundle(tmp_path)
    payload = _stage(root, bundle, tmp_path)

    recorded = read_staged_ledger(bundle)
    assert recorded is not None, "execute would otherwise fall back to card layout"
    assert len(recorded) == payload["staged"]
    media = "DCIM/Movie/2026_0829_123850_F.MP4"
    assert recorded[media] == "2026/08/29/2026_0829_123850_F.MP4"
    # Nothing source-relative survives in the record.
    assert not any(v.startswith("DCIM/") or v.startswith("yolo/")
                   for v in recorded.values())


def test_a_bundle_with_no_recorded_layout_reads_as_absent_not_empty(tmp_path):
    """Absent means "copy source keys verbatim". Empty would mean "copy
    nothing", and treating them alike silently drops a campaign."""
    from auto_ingest.custody.ledger import read_staged_ledger

    assert read_staged_ledger(tmp_path / "nothing-here") is None


def test_the_recorded_layout_is_replaced_whole_not_appended(tmp_path):
    """A second, smaller stage must not leave the first stage's paths behind."""
    from auto_ingest.custody.ledger import read_staged_ledger

    root = _card(tmp_path)
    bundle = _bundle(tmp_path)
    _stage(root, bundle, tmp_path)
    first = read_staged_ledger(bundle)

    (root / "heatmap" / "2026_0829_123850_F.png").unlink()
    _stage(root, bundle, tmp_path)
    second = read_staged_ledger(bundle)

    assert second == first, "nothing changed, so the record must be identical"
    lines = (Path(bundle) / "ledgers" / "staged.jsonl").read_text().splitlines()
    assert len(lines) == len(second), "no superseded rows left behind"


def test_execute_reports_the_layout_it_will_use_before_it_runs(tmp_path):
    root = _card(tmp_path)
    bundle = _bundle(tmp_path)
    payload = _stage(root, bundle, tmp_path)

    from auto_ingest.custody.cli import main as _main

    o, e = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(o), contextlib.redirect_stderr(e):
        _main(["execute", "--bundle", bundle, "--json"])
    report = json.loads(o.getvalue())
    assert report["layout"] == "staged", "a recorded layout must be picked up"
    assert report["staged_objects"] == payload["staged"]


def test_flat_ignores_a_recorded_layout_on_purpose(tmp_path):
    root = _card(tmp_path)
    bundle = _bundle(tmp_path)
    _stage(root, bundle, tmp_path)

    from auto_ingest.custody.cli import main as _main

    o, e = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(o), contextlib.redirect_stderr(e):
        _main(["execute", "--bundle", bundle, "--flat", "--json"])
    assert json.loads(o.getvalue())["layout"] == "source_relative"


def test_the_plan_judges_presence_where_the_recorded_layout_says(tmp_path):
    """plan_copy must consult the recorded path, or an already-staged object is
    re-copied because its source-relative path is absent."""
    from auto_ingest.custody.executor import plan_copy
    from auto_ingest.custody.ledger import read_staged_ledger

    root = _card(tmp_path)
    bundle = _bundle(tmp_path)
    _stage(root, bundle, tmp_path)
    dest = tmp_path / "dest"
    dest.mkdir()

    mapping = read_staged_ledger(bundle)
    assert plan_copy(bundle, dest, mapping).absent, "nothing staged at the destination yet"

    landed = dest / mapping["DCIM/Movie/2026_0829_123850_F.MP4"]
    landed.parent.mkdir(parents=True, exist_ok=True)
    landed.write_bytes((root / "DCIM/Movie/2026_0829_123850_F.MP4").read_bytes())

    after = plan_copy(bundle, dest, mapping)
    assert "DCIM/Movie/2026_0829_123850_F.MP4" not in after.absent
