"""Non-mutating hotplug inventory avoids reusing unverified old cards."""
import hashlib
import json

import pytest

from auto_ingest.custody import next_card_discovery as d


@pytest.fixture()
def prior(tmp_path):
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps({
        "host": "x1-370", "source_files_deleted": 0,
        "nas_full_destination_verification_complete": False,
        "cards": [{
            "uuid": "69BE-B963",
            "uuid_reverified_after_unmount": True,
            "still_kernel_mounted": False,
            "safe_for_physical_removal_with_data_retained": True,
            "safe_to_erase_or_reuse": False,
        }],
    }))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def device(name, uuid, mounted=False):
    return {
        "name": name, "type": "disk", "tran": "usb", "rm": True,
        "children": [{
            "name": name + "1", "type": "part", "fstype": "vfat",
            "uuid": uuid,
            "mountpoints": ["/media/scott/" + uuid] if mounted else [None],
        }],
    }


def test_existing_preserved_card_not_reingested_and_new_detected(prior):
    path, digest = prior
    old = d.verified_old_cards(path, digest)
    data = {"blockdevices": [
        device("sda", "69BE-B963"),
        device("sdb", "AABB-1122", mounted=True),
        {"name": "nvme0n1", "type": "disk", "rm": False},
        {"name": "sdc", "type": "disk", "rm": True, "tran": "usb"},
    ]}
    output = d.inspect_lsblk(data, prior_uuids=old)
    assert output["prior_card_count_still_inserted"] == 1
    assert output["new_card_count"] == 1
    new = output["new_media_candidates"][0]
    assert new["uuid"] == "AABB-1122"
    assert new["current_partition"] == "/dev/sdb1"
    assert new["classification"] == "new_mounted_candidate_mount_witness_required_before_inventory"
    assert not new["safe_to_begin_custody_inventory"]
    assert new["requires_independent_mount_uuid_witness"]
    assert new["not_allowed_to_erase_or_release"] is True
    old_card = output["previous_original_cards_still_detected"][0]
    assert old_card["classification"] == "original_preserve_do_not_reingest_as_new"
    assert old_card["safe_to_begin_custody_inventory"] is False
    assert output["copies_performed"] == 0 and output["deletions_performed"] == 0


def test_empty_reader_and_unmounted_new_card_are_separate(prior):
    output = d.inspect_lsblk({
        "blockdevices": [
            {"name": "sdc", "type": "disk", "rm": True, "tran": "usb"},
            device("sdd", "DDEE-1234"),
        ],
    }, prior_uuids={"69BE-B963"})
    assert output["new_card_count"] == 1
    assert not output["new_media_candidates"][0]["currently_mounted"]
    assert "manual_read_only_mount_needed" in output["new_media_candidates"][0]["classification"]
    assert output["unqualified_removable_devices"][0]["reason"] == "reader_empty_or_no_partition"


def test_cloned_uuid_is_not_safe_to_assign_campaign():
    with pytest.raises(ValueError, match="duplicate removable filesystem UUID"):
        d.inspect_lsblk({"blockdevices": [
            device("sda", "AABB-1122"), device("sdd", "AABB-1122"),
        ]}, prior_uuids=set())


def test_corrupt_old_handoff_and_changed_sha_are_denied(prior):
    path, digest = prior
    with pytest.raises(ValueError, match="prior-handoff SHA"):
        d.verified_old_cards(path, "0" * 64)
    data = json.loads(path.read_text())
    data["cards"][0]["safe_to_erase_or_reuse"] = True
    path.write_text(json.dumps(data))
    changed_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="unsafe or duplicate historical"):
        d.verified_old_cards(path, changed_sha)


def test_report_only_creates_exclusive_os_disk_receipt(tmp_path):
    dst = tmp_path / "inventory.json"
    data = d.inspect_lsblk({"blockdevices": []}, prior_uuids=set())
    d.write_receipt(dst, data)
    assert json.loads(dst.read_text())["source_release_allowed"] is False
    with pytest.raises(FileExistsError):
        d.write_receipt(dst, data)
    with pytest.raises(ValueError, match="off NAS"):
        d.write_receipt(__import__("pathlib").Path("/nas/do-not-write.json"), data)


def test_dangerous_device_identifier_denied():
    with pytest.raises(ValueError, match="untrusted USB device"):
        d.inspect_lsblk({"blockdevices": [
            {"name": "../../sda", "type": "disk", "tran": "usb", "rm": True},
        ]}, prior_uuids=set())
