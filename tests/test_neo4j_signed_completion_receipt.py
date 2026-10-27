"""Synthetic-only Ed25519 backup role-claim tests; no production key or backup."""

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

    def test_crypto_unavailable_reports_failure_not_unsigned_fallback(self):
        with mock.patch.dict(sys.modules, {"cryptography.exceptions": None}):
            r = signed.verify_signed_claim(self.a, self.keys)
        self.assertEqual(r["state"], "unverified")
        self.assertFalse(r["approvedProducerKeyVerified"])


if __name__ == "__main__":
    unittest.main()
