"""Synthetic-only Ed25519 backup role-claim tests; no production key or backup."""

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import neo4j_signed_completion_receipt as signed
from neo4j_backup_receipt_verifier import validate as unsigned_validate

CRYPTO_AVAILABLE = importlib.util.find_spec("cryptography") is not None
if CRYPTO_AVAILABLE:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


def claim(role="primary"):
    return {
        "schema": "neo4j-backup-completion-evidence-v1",
        "role": role,
        "runId": "synthetic-run-001",
        "startUtc": "2026-10-27T06:27:01Z" if role == "primary" else "2026-10-27T07:27:01Z",
        "finishUtc": "2026-10-27T07:01:02Z" if role == "primary" else "2026-10-27T07:52:02Z",
        "exitCode": 0,
        "archiveBytes": 4096,
        "archiveSha256": "a" * 64,
    }


@unittest.skipUnless(CRYPTO_AVAILABLE, "Optional cryptography package not installed")
class SignedCompletionClaimTests(unittest.TestCase):
    def setUp(self):
        self.primary = Ed25519PrivateKey.generate()
        self.mirror = Ed25519PrivateKey.generate()

        def raw(private):
            return private.public_key().public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )

        self.keys = {
            "primary": {"keyId": "approved-primary", "publicKeyRaw": raw(self.primary)},
            "mirror": {"keyId": "approved-mirror", "publicKeyRaw": raw(self.mirror)},
        }
        self.a = signed.create_signed_claim(claim(), "approved-primary", self.primary)
        self.b = signed.create_signed_claim(claim("mirror"), "approved-mirror", self.mirror)

    def test_valid_signature_authenticates_only_a_claim_not_job_or_archive(self):
        for role, env in (("primary", self.a), ("mirror", self.b)):
            with self.subTest(role=role):
                report = signed.verify_signed_claim(env, self.keys)
                self.assertEqual(
                    report["state"], "approved-key-signed-claim-content-and-restore-unverified"
                )
                self.assertTrue(report["approvedProducerKeyVerified"])
                self.assertTrue(report["claimedExitZero"])
                for field in (
                    "archiveBytesIndependentlyVerified",
                    "sourceMirrorByteIdentityVerified",
                    "databaseConsistencyVerified",
                    "databaseRestoreVerified",
                    "backupJobExecutionVerified",
                    "deleteAuthorized",
                    "hostChangeAuthorized",
                ):
                    self.assertFalse(report[field])
                for field in ("archiveBytesRead", "nasWrites", "remoteWrites", "modelRequestsSent"):
                    self.assertEqual(report[field], 0)
        self.assertEqual(
            unsigned_validate(self.a["claim"])["state"],
            "unsigned-success-claim-not-certified",
        )

    def test_two_independent_approved_roles_only_prove_matching_signed_claims(self):
        out = signed.verify_signed_pair(self.a, self.b, self.keys)
        self.assertEqual(out["state"], "two-approved-keys-signed-matching-claims-not-byte-proof")
        self.assertFalse(out["sourceMirrorByteIdentityVerified"])
        self.assertFalse(out["backupJobExecutionVerified"])
        self.assertFalse(out["databaseRestoreVerified"])
        self.assertEqual(out["archiveBytesRead"], 0)

    def test_swap_envelope_roles_or_pins_fail_closed(self):
        self.assertFalse(
            signed.verify_signed_pair(self.b, self.a, self.keys)["approvedProducerKeyVerified"]
        )
        forged = {
            "primary": self.keys["mirror"],
            "mirror": self.keys["primary"],
        }
        self.assertFalse(
            signed.verify_signed_pair(self.a, self.b, forged)["approvedProducerKeyVerified"]
        )
        changed = {
            "primary": {**self.keys["primary"], "keyId": "different-primary"},
            "mirror": self.keys["mirror"],
        }
        self.assertFalse(signed.verify_signed_claim(self.a, changed)["approvedProducerKeyVerified"])

    def test_different_key_ids_with_same_approved_public_key_not_two_signers(self):
        other = {
            "primary": self.keys["primary"],
            "mirror": {
                "keyId": "approved-mirror",
                "publicKeyRaw": self.keys["primary"]["publicKeyRaw"],
            },
        }
        mirror_signed_with_same_key = signed.create_signed_claim(
            claim("mirror"), "approved-mirror", self.primary
        )
        r = signed.verify_signed_pair(self.a, mirror_signed_with_same_key, other)
        self.assertFalse(r["approvedProducerKeyVerified"])
        self.assertEqual(r["state"], "unverified")

    def test_single_signed_claim_is_invalid_after_any_field_tampering(self):
        attacks = (
            {"archiveBytes": 4097},
            {"archiveSha256": "b" * 64},
            {"exitCode": 1},
            {"runId": "another-run-2026"},
            {"role": "mirror"},
            {"startUtc": "2026-10-27T06:28:01Z"},
        )
        for change in attacks:
            with self.subTest(change=change):
                tampered = {**self.a, "claim": {**self.a["claim"], **change}}
                self.assertFalse(
                    signed.verify_signed_claim(tampered, self.keys)["approvedProducerKeyVerified"]
                )

    def test_signature_and_key_id_tampering_fail_closed(self):
        forged = [
            {**self.a, "keyId": "approved-mirror"},
            {**self.a, "signature": "A" * 86},
            {**self.a, "signature": "../fake-signature"},
            {**self.a, "extra": "ignored?"},
            {**self.a, "schema": "v2"},
            {**self.a, "claim": {**self.a["claim"], "archiveBytes": True}},
        ]
        for record in forged:
            with self.subTest(forged=list(record.keys())):
                self.assertFalse(
                    signed.verify_signed_claim(record, self.keys)["approvedProducerKeyVerified"]
                )

    def test_nonzero_exit_or_incomplete_claim_cannot_be_signed_as_success(self):
        for changed in (
            {"exitCode": 1},
            {"exitCode": True},
            {"archiveBytes": 0},
            {"archiveSha256": "unsigned"},
            {"runId": "../bad"},
        ):
            with self.subTest(change=changed):
                with self.assertRaises(ValueError):
                    signed.create_signed_claim(
                        {**claim(), **changed}, "approved-primary", self.primary
                    )

    def test_claim_field_order_is_canonical_for_signature(self):
        reordered = {k: self.a["claim"][k] for k in reversed(tuple(self.a["claim"]))}
        envelope = {**self.a, "claim": reordered}
        self.assertTrue(
            signed.verify_signed_claim(envelope, self.keys)["approvedProducerKeyVerified"]
        )

    def test_cold_mirror_timeline_and_hash_must_match_despite_signatures(self):
        for change in (
            {"startUtc": "2026-10-27T06:00:01Z"},
            {"archiveSha256": "f" * 64},
            {"archiveBytes": 4097},
            {"runId": "another-run-2026"},
        ):
            with self.subTest(change=change):
                envelope = signed.create_signed_claim(
                    {**claim("mirror"), **change}, "approved-mirror", self.mirror
                )
                self.assertFalse(
                    signed.verify_signed_pair(self.a, envelope, self.keys)[
                        "sourceMirrorByteIdentityVerified"
                    ]
                )
                self.assertFalse(
                    signed.verify_signed_pair(self.a, envelope, self.keys)[
                        "approvedProducerKeyVerified"
                    ]
                )

    def test_private_owner_directory_atomic_no_overwrite_receipt_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            private = Path(tmp) / "private-claims"
            private.mkdir(mode=0o700)
            file = signed.publish_signed_envelope(private, self.a)
            self.assertEqual(file.stat().st_mode & 0o777, 0o600)
            self.assertEqual(signed.load_signed_envelope(file), self.a)
            self.assertEqual(
                signed.verify_signed_claim(signed.load_signed_envelope(file), self.keys)[
                    "approvedProducerKeyVerified"
                ],
                True,
            )
            before = file.read_bytes()
            with self.assertRaises(FileExistsError):
                signed.publish_signed_envelope(private, self.a)
            self.assertEqual(file.read_bytes(), before)
            self.assertEqual(sorted(x.name for x in private.iterdir()), [file.name])

    def test_private_publisher_rejects_symlink_in_any_ancestor_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real = root / "real"
            real.mkdir(mode=0o700)
            nested = real / "nested"
            nested.mkdir(mode=0o700)
            private = nested / "private"
            private.mkdir(mode=0o700)
            alias = root / "alias"
            alias.symlink_to(real, target_is_directory=True)
            with self.assertRaises(ValueError):
                signed.publish_signed_envelope(alias / "nested" / "private", self.a)
            self.assertEqual(list(private.iterdir()), [])
            # No symlinked ancestor: same target directory is still valid.
            actual = signed.publish_signed_envelope(private, self.a)
            self.assertEqual(signed.load_signed_envelope(actual), self.a)

    def test_private_publisher_rejects_parent_traversal_and_relative_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            private = root / "private"
            private.mkdir(mode=0o700)
            paths = (root / "private" / ".." / "private", Path("private"))
            for path in paths:
                with self.subTest(kind="relative" if not path.is_absolute() else "traversal"):
                    with self.assertRaises(ValueError):
                        signed.publish_signed_envelope(path, self.a)
            self.assertEqual(list(private.iterdir()), [])

    def test_private_directory_path_rejects_intermediate_non_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            file = root / "file"
            file.write_bytes(b"synthetic-only")
            with self.assertRaises(ValueError):
                signed.publish_signed_envelope(file / "nested", self.a)
            self.assertEqual(file.read_bytes(), b"synthetic-only")

    def test_partial_write_failure_never_publishes_final_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            private = Path(tmp) / "private"
            private.mkdir(mode=0o700)
            with mock.patch.object(
                signed.os, "write", side_effect=OSError("synthetic write fault")
            ):
                with self.assertRaises(OSError):
                    signed.publish_signed_envelope(private, self.a)
            self.assertEqual(list(private.iterdir()), [])

    def test_directory_fsync_failure_rolls_back_visible_final_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            private = Path(tmp) / "private"
            private.mkdir(mode=0o700)
            actual = signed.os.fsync
            calls = [0]

            def fail_once(fd):
                calls[0] += 1
                if calls[0] == 2:
                    raise OSError("synthetic final-directory-durability-failure")
                return actual(fd)

            with mock.patch.object(signed.os, "fsync", side_effect=fail_once):
                with self.assertRaises(OSError):
                    signed.publish_signed_envelope(private, self.a)
            self.assertGreaterEqual(calls[0], 3)
            self.assertEqual(list(private.iterdir()), [])
            # A retry after a successfully rolled-back publication is safe.
            receipt = signed.publish_signed_envelope(private, self.a)
            self.assertEqual(signed.load_signed_envelope(receipt), self.a)

    def test_persistent_directory_fsync_failure_is_explicitly_ambiguous(self):
        with tempfile.TemporaryDirectory() as tmp:
            private = Path(tmp) / "private"
            private.mkdir(mode=0o700)
            actual = signed.os.fsync
            calls = [0]

            def fail_directory(fd):
                calls[0] += 1
                if calls[0] >= 2:
                    raise OSError("synthetic persistent-storage-durability-failure")
                return actual(fd)

            with mock.patch.object(signed.os, "fsync", side_effect=fail_directory):
                with self.assertRaisesRegex(
                    OSError,
                    "receipt publication uncertain; reconcile private directory",
                ):
                    signed.publish_signed_envelope(private, self.a)
            # In this synthetic environment unlink succeeded, but the
            # failed sync makes crash durability UNKNOWN. No success signal.
            self.assertEqual(list(private.iterdir()), [])

    def test_temporary_unlink_failure_after_hardlink_rolls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            private = Path(tmp) / "private"
            private.mkdir(mode=0o700)
            actual = signed.os.unlink
            calls = [0]

            def fail_first_unlink(path, *args, **kwargs):
                calls[0] += 1
                if calls[0] == 1:
                    raise OSError("synthetic temporary-link-removal-failure")
                return actual(path, *args, **kwargs)

            with mock.patch.object(signed.os, "unlink", side_effect=fail_first_unlink):
                with self.assertRaises(OSError):
                    signed.publish_signed_envelope(private, self.a)
            self.assertEqual(list(private.iterdir()), [])

    def test_directory_must_be_existing_owner_only_and_not_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "private"
            target.mkdir(mode=0o700)
            target.chmod(0o755)
            with self.assertRaises(ValueError):
                signed.publish_signed_envelope(target, self.a)
            target.chmod(0o700)
            alias = Path(tmp) / "alias"
            alias.symlink_to(target, target_is_directory=True)
            with self.assertRaises(ValueError):
                signed.publish_signed_envelope(alias, self.a)
            with self.assertRaises(OSError):
                signed.publish_signed_envelope(Path(tmp) / "absent", self.a)
            self.assertEqual(list(target.iterdir()), [])

    def test_loader_rejects_symlinks_duplicate_keys_and_oversized_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good = root / "good.signed.json"
            good.write_text(json.dumps(self.a))
            self.assertEqual(signed.load_signed_envelope(good), self.a)
            alias = root / "link.signed.json"
            alias.symlink_to(good)
            self.assertIsNone(signed.load_signed_envelope(alias))
            duplicate = root / "duplicate.signed.json"
            duplicate.write_text('{"schema":"v1","schema":"v2"}')
            self.assertIsNone(signed.load_signed_envelope(duplicate))
            big = root / "big.signed.json"
            big.write_bytes(b"x" * (signed.MAX_ATTESTATION + 1))
            self.assertIsNone(signed.load_signed_envelope(big))

    def _synthetic_byte_pair(self, directory, *, same_inode=False):
        """The only backup bytes touched by the new integration tests."""
        original = b"synthetic-primary-backup-content\\n" * 200
        digest = hashlib.sha256(original).hexdigest()
        primary_path = Path(directory) / "source.backup"
        mirror_path = Path(directory) / "mirror.backup"
        primary_path.write_bytes(original)
        if same_inode:
            mirror_path.hardlink_to(primary_path)
        else:
            mirror_path.write_bytes(original)
        a = signed.create_signed_claim(
            {**claim(), "archiveBytes": len(original), "archiveSha256": digest},
            "approved-primary",
            self.primary,
        )
        b = signed.create_signed_claim(
            {**claim("mirror"), "archiveBytes": len(original), "archiveSha256": digest},
            "approved-mirror",
            self.mirror,
        )
        options = {
            "expected_run_id": "synthetic-run-001",
            "not_before_utc": "2026-10-27T06:00:00Z",
            "not_after_utc": "2026-10-27T08:30:00Z",
            "primary_path": primary_path,
            "mirror_path": mirror_path,
            "max_bytes": len(original),
        }
        return a, b, options, len(original)

    def test_two_signed_claims_require_explicit_independent_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            primary, mirror, args, size = self._synthetic_byte_pair(temp)
            verification = signed.verify_signed_pair_with_readback(
                primary, mirror, self.keys, **args
            )
            self.assertEqual(
                verification["state"],
                "role-signed-and-independent-byte-readback-verified-not-a-restore",
            )
            self.assertTrue(verification["approvedProducerKeyVerified"])
            self.assertTrue(verification["archiveBytesIndependentlyVerified"])
            self.assertTrue(verification["sourceMirrorByteIdentityVerified"])
            self.assertEqual(verification["archiveBytesRead"], 2 * size)
            for field in (
                "backupJobExecutionVerified",
                "databaseConsistencyVerified",
                "databaseRestoreVerified",
                "deleteAuthorized",
                "hostChangeAuthorized",
            ):
                self.assertFalse(verification[field])
            self.assertEqual(verification["nasWrites"], 0)
            self.assertEqual(verification["modelRequestsSent"], 0)

    def test_replayed_valid_signatures_outside_independent_expected_run_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            primary, mirror, args, _ = self._synthetic_byte_pair(temp)
            for change in (
                {"expected_run_id": "another-run-001"},
                {"not_before_utc": "2026-10-27T07:00:00Z"},
                {"not_after_utc": "2026-10-27T07:30:00Z"},
                {"not_before_utc": "2026-10-26T00:00:00Z"},
                {"not_after_utc": "2026-10-31T00:00:00Z"},
                {"not_before_utc": "malformed"},
                {"not_after_utc": "2030-10-27T08:30:00Z"},
                {"expected_run_id": "../../evil"},
            ):
                with self.subTest(change=change):
                    with mock.patch.object(
                        signed,
                        "compare_independently_read_files",
                        side_effect=AssertionError(
                            "must not read files without independently scoped run/time"
                        ),
                    ) as read:
                        outcome = signed.verify_signed_pair_with_readback(
                            primary, mirror, self.keys, **{**args, **change}
                        )
                    self.assertEqual(outcome["state"], "unverified")
                    self.assertEqual(outcome["archiveBytesRead"], 0)
                    read.assert_not_called()

    def test_exact_day_boundary_is_acceptable_but_long_window_is_not(self):
        with tempfile.TemporaryDirectory() as temp:
            primary, mirror, args, _ = self._synthetic_byte_pair(temp)
            window = {
                **args,
                "not_before_utc": "2026-10-27T06:00:00Z",
                "not_after_utc": "2026-10-28T06:00:00Z",
            }
            result = signed.verify_signed_pair_with_readback(primary, mirror, self.keys, **window)
            self.assertTrue(result["sourceMirrorByteIdentityVerified"])
            with mock.patch.object(
                signed,
                "compare_independently_read_files",
                side_effect=AssertionError("excessive window must avoid archive IO"),
            ) as read:
                oversized = signed.verify_signed_pair_with_readback(
                    primary,
                    mirror,
                    self.keys,
                    **{**window, "not_after_utc": "2026-10-28T06:00:01Z"},
                )
            self.assertEqual(oversized["state"], "unverified")
            read.assert_not_called()

    def test_invalid_role_signatures_never_trigger_archive_read(self):
        with tempfile.TemporaryDirectory() as temp:
            primary, mirror, args, _ = self._synthetic_byte_pair(temp)
            bad = {**mirror, "signature": "A" * 86}
            with mock.patch.object(
                signed,
                "compare_independently_read_files",
                side_effect=AssertionError("must verify signatures before data reads"),
            ) as read:
                outcome = signed.verify_signed_pair_with_readback(primary, bad, self.keys, **args)
            self.assertEqual(outcome["state"], "unverified")
            self.assertFalse(outcome["approvedProducerKeyVerified"])
            read.assert_not_called()

    def test_byte_readback_fail_closed_if_archive_changed_after_signing(self):
        with tempfile.TemporaryDirectory() as temp:
            primary, mirror, args, size = self._synthetic_byte_pair(temp)
            Path(args["mirror_path"]).write_bytes(b"x" * size)
            result = signed.verify_signed_pair_with_readback(primary, mirror, self.keys, **args)
            self.assertEqual(result["state"], "unverified")
            self.assertFalse(result["archiveBytesIndependentlyVerified"])
            self.assertFalse(result["sourceMirrorByteIdentityVerified"])
            self.assertEqual(result["archiveBytesRead"], 0)
            self.assertFalse(result["databaseRestoreVerified"])

    def test_same_inode_and_symlink_mirror_cannot_prove_two_independent_copies(self):
        with tempfile.TemporaryDirectory() as temp:
            primary, mirror, args, _ = self._synthetic_byte_pair(temp, same_inode=True)
            outcome = signed.verify_signed_pair_with_readback(primary, mirror, self.keys, **args)
            self.assertEqual(outcome["state"], "unverified")
            Path(args["mirror_path"]).unlink()
            Path(args["mirror_path"]).symlink_to(Path(args["primary_path"]))
            outcome = signed.verify_signed_pair_with_readback(primary, mirror, self.keys, **args)
            self.assertEqual(outcome["state"], "unverified")

    def test_explicit_byte_cap_prevents_implicit_large_archive_reads(self):
        with tempfile.TemporaryDirectory() as temp:
            primary, mirror, args, size = self._synthetic_byte_pair(temp)
            for bad_cap in (True, 0, size - 1, 32 * 1024**3 + 1):
                with self.subTest(cap=bad_cap):
                    outcome = signed.verify_signed_pair_with_readback(
                        primary, mirror, self.keys, **{**args, "max_bytes": bad_cap}
                    )
                    self.assertEqual(outcome["state"], "unverified")
                    self.assertFalse(outcome["archiveBytesIndependentlyVerified"])
                    self.assertEqual(outcome["archiveBytesRead"], 0)

    def test_no_implicit_archive_reads_when_only_signed_pair_queried(self):
        with mock.patch.object(
            signed,
            "compare_independently_read_files",
            side_effect=AssertionError("a signed pair check is never an implicit archive read"),
        ) as read:
            outcome = signed.verify_signed_pair(self.a, self.b, self.keys)
        self.assertEqual(
            outcome["state"], "two-approved-keys-signed-matching-claims-not-byte-proof"
        )
        read.assert_not_called()

    def test_crypto_unavailable_reports_failure_not_unsigned_fallback(self):
        with mock.patch.dict(sys.modules, {"cryptography.exceptions": None}):
            r = signed.verify_signed_claim(self.a, self.keys)
        self.assertEqual(r["state"], "unverified")
        self.assertFalse(r["approvedProducerKeyVerified"])


if __name__ == "__main__":
    unittest.main()
