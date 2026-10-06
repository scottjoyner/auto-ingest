"""A destination that is a subdirectory of a network mount.

The shape that blocked release here: a CIFS share mounted at /nas, with the
archive at /nas/fileserver/headcam. `observe_mount` matches mount points, so the
destination read as "not mounted" with no identity, and the gate refused - even
though the bytes were verified on the storage and the copy had succeeded.

Two rules this module exists to keep honest:

  * "is the card there" must NEVER be answered by "is some filesystem here".
    For a source, a containing mount is not evidence of anything.
  * "mounted" and "the path exists" are different facts. Releasing a source on
    the strength of a mounted share whose directory is absent would delete the
    only copy of a recording nothing can currently write to.
"""
from __future__ import annotations

from dataclasses import replace

from auto_ingest.custody.campaign import Campaign
from auto_ingest.custody.mounts import (
    MountObservation,
    containing_mount,
    observations_to_evidence,
    observe_campaign,
    observe_mount,
)

CIFS = MountObservation(
    mount_point="/nas", present=True,
    device="//192.168.1.202/fileserver", filesystem_type="cifs",
    options=("rw", "relatime"),
)
VFAT = MountObservation(
    mount_point="/media/scott/UNTITLED", present=True,
    device="/dev/sdb1", filesystem_type="vfat", options=("ro", "relatime"),
)
TABLE = (CIFS, VFAT)


def _campaign(source_mount: str, dest_host_path: str) -> Campaign:
    from auto_ingest.custody.campaign import (
        Campaign,
        CardIdentity,
        SourceRef,
    )
    from auto_ingest.custody.destination import DestinationRef, LogicalDestination

    return Campaign.create(
        card_id="CARD-01",
        source=SourceRef(
            card=CardIdentity(device="/dev/sdb1", filesystem_uuid="4620-180F"),
            mount_point=source_mount, read_only=True),
        destination=DestinationRef(
            logical=LogicalDestination(name="primary",
                                       relative_path="fileserver/headcam"),
            host_path=dest_host_path, resolved_from="env:CUSTODY_DESTINATION_ROOT"),
    )


# ---------------------------------------------------------------------------
# containing_mount
# ---------------------------------------------------------------------------

def test_the_deepest_enclosing_mount_wins():
    inner = MountObservation(mount_point="/nas/fileserver", present=True,
                             device="//other/share", filesystem_type="cifs")
    assert containing_mount("/nas/fileserver/headcam",
                            (CIFS, inner)) is inner


def test_a_sibling_prefix_is_not_a_parent():
    """`/nasx` is not inside `/nas`. A string startswith check gets this wrong and
    would attach an unrelated filesystem's identity to a destination."""
    assert containing_mount("/nasx/headcam", TABLE) is None


def test_the_path_itself_is_not_its_own_container():
    assert containing_mount("/nas", TABLE) is None


def test_nothing_contains_a_bare_relative_path():
    assert containing_mount("headcam", TABLE) is None


# ---------------------------------------------------------------------------
# observe_mount: opt-in, and the difference it makes
# ---------------------------------------------------------------------------

def test_default_behaviour_is_unchanged_and_still_says_not_mounted():
    assert observe_mount("/nas/fileserver/headcam",
                         observations=TABLE).present is False


def test_the_observation_names_the_backing_mount_and_the_share(tmp_path):
    real = tmp_path / "nas" / "fileserver" / "headcam"
    real.mkdir(parents=True)
    m = observe_mount(str(real), observations=TABLE, allow_containing=True)
    # The observation is about a path under /nas, so build a table whose mount is
    # an ancestor of the tmp path.
    ancestor = replace(CIFS, mount_point=str(tmp_path / "nas"))
    m = observe_mount(str(real), observations=(ancestor,), allow_containing=True)
    assert m.present is True
    assert m.backing_mount_point == str(tmp_path / "nas")
    assert m.is_mount_point is False
    assert m.device == "//192.168.1.202/fileserver"
    assert m.filesystem_type == "cifs"
    assert m.path_exists is True
    assert m.usable is True


def test_a_missing_subdirectory_is_mounted_but_not_usable(tmp_path):
    ancestor = replace(CIFS, mount_point=str(tmp_path))
    absent = tmp_path / "fileserver" / "headcam"
    m = observe_mount(str(absent), observations=(ancestor,), allow_containing=True)
    assert m.present is True, "the filesystem really is mounted"
    assert m.path_exists is False
    assert m.usable is False, "and that is the fact the release gate needs"


def test_the_source_is_never_given_the_containing_fallback(tmp_path):
    """The one that matters. An empty mount point left behind by an unplugged
    card must not be reported as present because it happens to sit inside
    something else."""
    inside = tmp_path / "mnt" / "CARD-01"
    inside.mkdir(parents=True)
    ancestor = replace(CIFS, mount_point=str(tmp_path / "mnt"))
    m = observe_mount(str(inside), observations=(ancestor,))
    assert m.present is False
    assert m.device is None


def test_observe_campaign_gives_only_the_destination_the_fallback(tmp_path):
    """The asymmetry, stated as the campaign observer actually behaves."""
    from auto_ingest.custody.mounts import observe_storage_identity

    src = tmp_path / "mnt" / "CARD-01"
    src.mkdir(parents=True)
    dest = tmp_path / "mnt" / "fileserver" / "headcam"
    dest.mkdir(parents=True)
    ancestor = replace(CIFS, mount_point=str(tmp_path / "mnt"))
    table = (ancestor,)

    campaign = _campaign(str(src), str(dest))
    report = observe_campaign(campaign, mounts_path="/nonexistent")
    # An unreadable mount table yields an empty one; both ends then read absent.
    assert report["source"]["observation"]["present"] is False
    assert report["destination"]["observation"]["present"] is False

    # With a real table the destination resolves and the source still does not.
    assert observe_mount(str(src), observations=table).present is False
    d = observe_mount(str(dest), observations=table, allow_containing=True)
    assert d.present is True
    identity = observe_storage_identity(d)
    assert identity is not None
    assert identity.device == "//192.168.1.202/fileserver"


def test_the_destination_fragment_reports_usability(tmp_path):
    dest = tmp_path / "mnt" / "fileserver" / "headcam"
    dest.mkdir(parents=True)
    ancestor = replace(CIFS, mount_point=str(tmp_path / "mnt"))
    d = observe_mount(str(dest), observations=(ancestor,), allow_containing=True)
    assert d.usable is True


# ---------------------------------------------------------------------------
# Declared identity for a share
# ---------------------------------------------------------------------------

def test_a_share_identity_is_the_device_string_not_a_uuid():
    from auto_ingest.custody.destination import StorageIdentity, match_identity

    observed = StorageIdentity(device="//192.168.1.202/fileserver",
                               filesystem_type="cifs")
    declared = StorageIdentity(device="//192.168.1.202/fileserver")
    m = match_identity(declared, observed)
    assert m.matched is True
    assert "device_match" in m.reason


def test_a_uuid_declared_for_a_share_is_still_refused():
    """The rule that made this a real decision: a UUID nothing can observe is a
    claim, not a check."""
    from auto_ingest.custody.destination import StorageIdentity, match_identity

    declared = StorageIdentity(filesystem_uuid="deadbeef",
                               device="//192.168.1.202/fileserver")
    observed = StorageIdentity(device="//192.168.1.202/fileserver",
                               filesystem_type="cifs")
    m = match_identity(declared, observed)
    assert m.matched is False
    assert "unverifiable" in m.reason


def test_a_different_share_does_not_match():
    from auto_ingest.custody.destination import StorageIdentity, match_identity

    m = match_identity(StorageIdentity(device="//192.168.1.202/fileserver"),
                       StorageIdentity(device="//192.168.1.99/other"))
    assert m.matched is False


def test_observations_to_evidence_carries_usable_alongside_identity(tmp_path):
    dest = tmp_path / "mnt" / "fileserver" / "headcam"
    dest.mkdir(parents=True)
    ancestor = replace(CIFS, mount_point=str(tmp_path / "mnt"))
    d = observe_mount(str(dest), observations=(ancestor,), allow_containing=True)
    fragment = observations_to_evidence({
        "destination": {"identity": {"device": d.device}, "observed_usable": d.usable},
        "source": {"identity": None},
    })
    assert fragment["destination"]["observed_usable"] is True


# ---------------------------------------------------------------------------
# Declaring the destination
# ---------------------------------------------------------------------------

def _bundle_with_destination(tmp_path, *, identity=None, mounted=True,
                             host_path="default"):
    """A campaign on disk, so the declaration command has something to write to."""
    import json


    bundle = tmp_path / "b"
    bundle.mkdir()
    (bundle / "campaign.json").write_text(json.dumps({
        "campaign_id": "c1", "card_id": "CARD-01",
        "source": {"mount_point": "/media/scott/UNTITLED", "read_only": True,
                   "card": {"device": "/dev/sdb1",
                            "filesystem_uuid": "4620-180F"}},
        "destination": {
            "logical": {"name": "primary", "relative_path": "fileserver/headcam",
                        "canonical": "primary:fileserver/headcam"},
            "host_path": (str(tmp_path / "dest") if host_path == "default"
                          else host_path),
            "resolved_from": "test",
            "mounted": mounted,
            "identity": identity.to_dict() if identity else None,
        },
    }), encoding="utf-8")
    return bundle


def test_declaring_records_the_identity_in_the_campaign(tmp_path):
    import json

    from auto_ingest.custody.destination import StorageIdentity
    from auto_ingest.custody.store import declare_destination_identity

    bundle = _bundle_with_destination(tmp_path)
    result = declare_destination_identity(
        bundle, StorageIdentity(device="//host/share", filesystem_type="cifs"))
    assert result["declared"] is True
    written = json.loads((bundle / "campaign.json").read_text())
    assert written["destination"]["identity"]["device"] == "//host/share"


def test_a_declaration_is_not_silently_re_pointed(tmp_path):
    """Otherwise a wrong mount could satisfy the gate by being declared after the
    fact.

    The existing value is kept and the attempt reported, rather than the whole call
    being refused: a caller supplying both a host path and an identity should not
    lose the path because the identity was already on file.
    """
    import json

    from auto_ingest.custody.destination import StorageIdentity
    from auto_ingest.custody.store import declare_destination_identity

    bundle = _bundle_with_destination(
        tmp_path, identity=StorageIdentity(device="//host/share"))
    again = declare_destination_identity(bundle, StorageIdentity(device="//other/thing"))
    assert again["kept"]["identity"]["device"] == "//host/share"
    assert again["applied_identity"] is None
    written = json.loads((bundle / "campaign.json").read_text())
    assert written["destination"]["identity"]["device"] == "//host/share", (
        "the original declaration must survive a refused re-point")


def test_replacing_a_declaration_is_possible_but_explicit(tmp_path):
    import json

    from auto_ingest.custody.destination import StorageIdentity
    from auto_ingest.custody.store import declare_destination_identity

    bundle = _bundle_with_destination(
        tmp_path, identity=StorageIdentity(device="//host/share"))
    result = declare_destination_identity(
        bundle, StorageIdentity(device="//other/thing"), replace=True)
    assert result["declared"] is True and result["applied_identity"] is not None
    written = json.loads((bundle / "campaign.json").read_text())
    assert written["destination"]["identity"]["device"] == "//other/thing"


# ---------------------------------------------------------------------------
# The gate's asymmetry
# ---------------------------------------------------------------------------

def test_a_backed_but_absent_destination_blocks_release(tmp_path):
    """The new case. /nas mounted, /nas/fileserver/headcam not created: releasing
    the source would delete the only copy of a recording nothing can write to."""
    from auto_ingest.custody.destination import StorageIdentity
    from auto_ingest.custody.evidence import DestinationEvidence
    from auto_ingest.custody.policy import CustodyPolicy
    from auto_ingest.custody.release import evaluate_release

    campaign = _campaign("/media/scott/UNTITLED", str(tmp_path / "dest"))
    campaign = replace(campaign,
                       destination=replace(campaign.destination,
                                           identity=StorageIdentity(device="//h/s")))
    from auto_ingest.custody.evidence import CampaignEvidence

    evidence = CampaignEvidence(destination=DestinationEvidence(
        verified_files=1, verification_started=True, verification_complete=True,
        observed_identity=StorageIdentity(device="//h/s"),
        observed_usable=False, observed_backing_mount_point="/nas"))
    blockers = evaluate_release(campaign, evidence, CustodyPolicy()).blockers
    assert "destination_path_absent" in [b.code for b in blockers]
    assert "destination_not_mounted" not in [b.code for b in blockers], (
        "a mounted share with a missing subdirectory is a different problem")


def test_a_plain_directory_destination_is_not_treated_as_unmounted(tmp_path):
    """An ordinary directory is backed by no mount, and that is not a fact about
    any mount. Judged against the observation it refuses every campaign whose
    destination is a plain directory - which is most of them."""
    from auto_ingest.custody.destination import StorageIdentity
    from auto_ingest.custody.evidence import DestinationEvidence
    from auto_ingest.custody.policy import CustodyPolicy
    from auto_ingest.custody.release import evaluate_release

    campaign = _campaign("/media/scott/UNTITLED", str(tmp_path / "dest"))
    campaign = replace(campaign,
                       destination=replace(campaign.destination, mounted=True,
                                           identity=StorageIdentity(device="//h/s")))
    from auto_ingest.custody.evidence import CampaignEvidence

    evidence = CampaignEvidence(destination=DestinationEvidence(
        verified_files=1, verification_started=True, verification_complete=True,
        observed_identity=StorageIdentity(device="//h/s"),
        observed_usable=False, observed_backing_mount_point=None))
    blockers = evaluate_release(campaign, evidence, CustodyPolicy()).blockers
    assert "destination_not_mounted" not in [b.code for b in blockers]
    assert "destination_path_absent" not in [b.code for b in blockers]


def test_an_unset_destination_path_can_be_completed(tmp_path):
    """A campaign created with no destination env var resolves host_path to null,
    and the gate then refuses for want of a *path* rather than for want of an
    identity. Nothing was ever written anywhere, so filling it in is not
    re-pointing."""
    import json

    from auto_ingest.custody.destination import StorageIdentity
    from auto_ingest.custody.store import declare_destination_identity

    bundle = _bundle_with_destination(tmp_path, identity=None, host_path=None)
    assert json.loads((bundle / "campaign.json").read_text())[
        "destination"]["host_path"] is None
    result = declare_destination_identity(
        bundle, StorageIdentity(device="//h/s"), host_path="/nas/fileserver/dashcam")
    assert result["applied_host_path"] == "/nas/fileserver/dashcam"
    assert result["applied_identity"]["device"] == "//h/s"


def test_a_set_destination_path_is_not_re_pointed_without_replace(tmp_path):
    """This is the dangerous one: bytes already written to one archive while the
    campaign claims another."""
    from auto_ingest.custody.store import declare_destination_identity

    bundle = _bundle_with_destination(tmp_path)
    result = declare_destination_identity(bundle, host_path="/somewhere/else")
    assert result["applied_host_path"] is None
    assert result["kept"]["host_path"].endswith("/dest")


def test_declaring_nothing_is_a_usage_error_not_a_silent_no_op(tmp_path, capsys):
    import contextlib
    import io

    from auto_ingest.custody.cli import EXIT_USAGE, main

    bundle = _bundle_with_destination(tmp_path)
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(["declare-destination", "--bundle", str(bundle), "--apply"])
    assert code == EXIT_USAGE
    assert "nothing to record" in err.getvalue()
