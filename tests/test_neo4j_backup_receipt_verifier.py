"""Offline synthetic receipt/byte tests. No live Neo4j or production NAS I/O."""

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import neo4j_backup_receipt_verifier as r

DATA = b"synthetic neo4j backup test bytes\n" * 200
SHA = hashlib.sha256(DATA).hexdigest()


def sample(role="primary"):
    return {
        "schema": r.SCHEMA,
        "role": role,
        "runId": "synthetic-run-001",
        "startUtc": "2026-10-27T06:27:01Z",
        "finishUtc": "2026-10-27T07:01:02Z" if role == "primary" else "2026-10-27T07:52:02Z",
        "exitCode": 0,
        "archiveBytes": len(DATA),
        "archiveSha256": SHA,
    }


class ReceiptContractTests(unittest.TestCase):
    def test_unsigned_exit_claim_never_becomes_success_certificate(self):
        a = r.validate(sample())
        self.assertEqual(a["state"], "unsigned-success-claim-not-certified")
        self.assertTrue(a["claimedExitZero"])
        for flag in (
            "contentDigestIndependentlyVerified",
            "primarySourceBytesIndependentlyVerified",
            "mirrorSourceByteIdentityVerified",
            "writerSignatureVerified",
            "databaseConsistencyVerified",
            "restoreVerified",
            "deleteAuthorized",
            "backupChangeAuthorized",
        ):
            self.assertFalse(a[flag])
        self.assertNotIn("/restricted/path", json.dumps(a))

    def test_unknown_fields_missing_fields_and_bool_exit_fail_closed(self):
        for forged in (
            {"exitCode": True},
            {"exitCode": "0"},
            {"archiveBytes": True},
            {"archiveBytes": -1},
            {"archiveBytes": 0},
            {"archiveBytes": r.MAX_READ_MAX + 1},
            {"archiveSha256": "FAKE"},
            {"archiveSha256": SHA.upper()},
            {"role": "unknown"},
            {"schema": "v2"},
            {"runId": "../../unknown"},
            {"startUtc": "bad"},
            {"finishUtc": "2026-10-27T06:01:02Z"},
        ):
            with self.subTest(forged=forged):
                self.assertEqual(r.validate({**sample(), **forged})["state"], "unverified")
        broken = sample()
        broken.pop("archiveSha256")
        self.assertEqual(r.validate(broken)["state"], "unverified")
        broken = sample()
        broken["token"] = "sensitive"
        self.assertEqual(r.validate(broken)["state"], "unverified")

    def test_nonzero_exit_is_not_success(self):
        self.assertEqual(
            r.validate({**sample(), "exitCode": 1})["state"], "nonzero-exit-claimed-not-certified"
        )

    def test_matching_primary_mirror_claims_still_not_byte_proof(self):
        combined = r.compare_claims(sample(), sample("mirror"))
        self.assertEqual(combined["state"], "matching-unsigned-claims-not-certified")
        self.assertFalse(combined["mirrorSourceByteIdentityVerified"])
        self.assertFalse(combined["restoreVerified"])

    def test_claim_mismatches_rejected(self):
        for change in (
            {"runId": "backup-another-day"},
            {"archiveSha256": "0" * 64},
            {"archiveBytes": len(DATA) + 1},
            {"exitCode": 1},
            {"role": "primary"},
            {"finishUtc": "2026-10-27T06:00:00Z"},
        ):
            with self.subTest(change=change):
                self.assertEqual(
                    r.compare_claims(sample(), {**sample("mirror"), **change})["state"],
                    "unverified",
                )

    def test_byte_read_only_explicit_match_and_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.backup"
            path.write_bytes(DATA)
            a = r.check_selected_archive(sample(), path)
            self.assertEqual(a["state"], "local-hash-matched-writer-and-restore-unverified")
            self.assertEqual(a["archiveReads"], len(DATA))
            self.assertTrue(a["contentDigestIndependentlyVerified"])
            self.assertFalse(a["writerSignatureVerified"])
            self.assertFalse(a["restoreVerified"])
            self.assertFalse(a["mirrorSourceByteIdentityVerified"])
            wrong = r.check_selected_archive({**sample(), "archiveSha256": "0" * 64}, path)
            self.assertEqual(wrong["state"], "local-archive-hash-mismatch")
            self.assertFalse(wrong["contentDigestIndependentlyVerified"])

    def test_no_archive_read_when_oversized_or_wrong_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.backup"
            path.write_bytes(DATA)
            out = r.check_selected_archive(sample(), path, len(DATA) - 1)
            self.assertEqual(out["state"], "unverified")
            self.assertEqual(out["archiveReads"], 0)
            out = r.check_selected_archive({**sample(), "archiveBytes": len(DATA) + 1}, path)
            self.assertEqual(out["archiveReads"], 0)

    def test_symlink_and_nonbackup_extension_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            good = Path(tmp) / "data.backup"
            good.write_bytes(DATA)
            link = Path(tmp) / "link.backup"
            link.symlink_to(good)
            self.assertEqual(r.check_selected_archive(sample(), link)["state"], "unverified")
            other = Path(tmp) / "data.txt"
            other.write_bytes(DATA)
            self.assertEqual(r.check_selected_archive(sample(), other)["state"], "unverified")

    def test_malicious_io_limits_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "small.backup"
            p.write_bytes(DATA)
            for cap in (True, 0, -1, 1.5, "1GiB", r.MAX_READ_MAX + 1):
                with self.subTest(cap=cap):
                    self.assertEqual(
                        r.check_selected_archive(sample(), p, cap)["state"], "unverified"
                    )

    def test_small_manifest_loader_rejects_duplicate_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "receipt.json"
            p.write_text('{"role":"primary","role":"mirror"}')
            self.assertIsNone(r.load(p))

    def test_manifest_loader_rejects_symlink_and_oversize(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "receipt.json"
            p.write_text(json.dumps(sample()))
            self.assertEqual(r.load(p), sample())
            link = Path(tmp) / "link.json"
            link.symlink_to(p)
            self.assertIsNone(r.load(link))
            p.write_bytes(b"0" * (r.MAX_RECEIPT + 1))
            self.assertIsNone(r.load(p))

    def test_content_mutation_detects_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "backup.backup"
            p.write_bytes(DATA)
            altered = DATA[:-1] + b"x"
            p.write_bytes(altered)
            self.assertEqual(
                r.check_selected_archive(sample(), p)["state"], "local-archive-hash-mismatch"
            )

    def test_no_file_reads_when_untrusted_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "not-present.backup"
            x = r.check_selected_archive(None, p)
            self.assertEqual(x["archiveReads"], 0)
            self.assertEqual(x["state"], "unverified")

    def test_matching_two_independently_read_synthetic_files_proves_bytes_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "primary.backup"
            mirror = Path(tmp) / "mirror.backup"
            source.write_bytes(DATA)
            mirror.write_bytes(DATA)
            out = r.compare_independently_read_files(sample(), source, sample("mirror"), mirror)
            self.assertEqual(out["state"], "readback-byte-identical-writer-and-restore-unverified")
            self.assertTrue(out["mirrorSourceByteIdentityVerified"])
            self.assertTrue(out["primarySourceBytesIndependentlyVerified"])
            self.assertEqual(out["archiveReads"], 2 * len(DATA))
            self.assertFalse(out["writerSignatureVerified"])
            self.assertFalse(out["databaseConsistencyVerified"])
            self.assertFalse(out["restoreVerified"])
            self.assertFalse(out["backupChangeAuthorized"])

    def test_changed_copy_cannot_pass_independent_byte_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "primary.backup"
            mirror = Path(tmp) / "mirror.backup"
            source.write_bytes(DATA)
            mirror.write_bytes(DATA[:-1] + b"X")
            out = r.compare_independently_read_files(sample(), source, sample("mirror"), mirror)
            self.assertEqual(out["state"], "unverified")
            self.assertFalse(out["mirrorSourceByteIdentityVerified"])

    def test_identical_self_claims_cannot_skip_absent_primary_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "mirror.backup"
            destination.write_bytes(DATA)
            out = r.compare_independently_read_files(
                sample(), Path(tmp) / "absent.backup", sample("mirror"), destination
            )
            self.assertEqual(out["state"], "unverified")
            self.assertEqual(out["archiveReads"], 0)

    def test_pair_hashing_under_explicit_byte_cap_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "primary.backup"
            source.write_bytes(DATA)
            mirror = Path(tmp) / "mirror.backup"
            mirror.write_bytes(DATA)
            out = r.compare_independently_read_files(
                sample(), source, sample("mirror"), mirror, len(DATA) - 1
            )
            self.assertEqual(out["state"], "unverified")
            self.assertFalse(out["primarySourceBytesIndependentlyVerified"])

    def test_alias_to_same_inode_never_counts_as_independent_mirror(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "primary.backup"
            first.write_bytes(DATA)
            second = Path(tmp) / "mirror.backup"
            os.link(first, second)
            out = r.compare_independently_read_files(sample(), first, sample("mirror"), second)
            self.assertEqual(out["state"], "unverified")
            self.assertEqual(out["reason"], "source-and-mirror-not-distinct-regular-files")
            self.assertFalse(out["mirrorSourceByteIdentityVerified"])
            out = r.compare_independently_read_files(sample(), first, sample("mirror"), first)
            self.assertEqual(out["state"], "unverified")

    def test_replaced_mirror_path_during_pair_read_never_claims_byte_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "primary.backup"
            mirror = Path(tmp) / "mirror.backup"
            source.write_bytes(DATA)
            mirror.write_bytes(DATA)
            actual = r.check_selected_archive

            def swap_mirror_during_source_read(claim, path, max_bytes=r.DEFAULT_READ_MAX, **kwargs):
                if Path(path) == source:
                    # Simulate directory entry substitution after the pair
                    # captured distinct inodes, but before the second read.
                    mirror.unlink()
                    mirror.write_bytes(DATA)
                return actual(claim, path, max_bytes, **kwargs)

            with mock.patch.object(
                r, "check_selected_archive", side_effect=swap_mirror_during_source_read
            ):
                observed = r.compare_independently_read_files(
                    sample(), source, sample("mirror"), mirror
                )
            self.assertEqual(observed["state"], "unverified")
            self.assertEqual(observed["reason"], "mirror-byte-readback-not-verified")
            self.assertFalse(observed["mirrorSourceByteIdentityVerified"])
            self.assertEqual(observed["archiveReads"], 0)

    def test_replaced_primary_path_between_pair_stat_and_read_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "primary.backup"
            mirror = Path(tmp) / "mirror.backup"
            source.write_bytes(DATA)
            mirror.write_bytes(DATA)
            actual = r.check_selected_archive

            def substitute_primary_on_read(claim, path, max_bytes=r.DEFAULT_READ_MAX, **kwargs):
                if Path(path) == source:
                    source.unlink()
                    source.write_bytes(DATA)
                return actual(claim, path, max_bytes, **kwargs)

            with mock.patch.object(
                r, "check_selected_archive", side_effect=substitute_primary_on_read
            ):
                observed = r.compare_independently_read_files(
                    sample(), source, sample("mirror"), mirror
                )
            self.assertEqual(observed["state"], "unverified")
            self.assertEqual(observed["reason"], "primary-byte-readback-not-verified")
            self.assertFalse(observed["primarySourceBytesIndependentlyVerified"])

    def test_replaced_primary_after_successful_read_does_not_pass_final_pair_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "primary.backup"
            mirror = Path(tmp) / "mirror.backup"
            source.write_bytes(DATA)
            mirror.write_bytes(DATA)
            actual = r.check_selected_archive

            def replace_primary_when_mirror_read(
                claim, path, max_bytes=r.DEFAULT_READ_MAX, **kwargs
            ):
                if Path(path) == mirror:
                    source.unlink()
                    source.write_bytes(DATA)
                return actual(claim, path, max_bytes, **kwargs)

            with mock.patch.object(
                r, "check_selected_archive", side_effect=replace_primary_when_mirror_read
            ):
                observed = r.compare_independently_read_files(
                    sample(), source, sample("mirror"), mirror
                )
            self.assertEqual(observed["state"], "unverified")
            self.assertEqual(observed["reason"], "source-or-mirror-path-changed-after-readback")
            self.assertFalse(observed["mirrorSourceByteIdentityVerified"])


if __name__ == "__main__":
    unittest.main()
