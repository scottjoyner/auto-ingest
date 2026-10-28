"""Synthetic-only owner job integration; never runs a real backup or loads keys."""

import datetime as dt
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from neo4j_owner_job_adapter import run_owner_stage
from neo4j_signed_completion_receipt import (
    load_signed_envelope,
    verify_signed_claim,
    verify_signed_pair,
)

UTC = dt.timezone.utc
START = dt.datetime(2026, 10, 27, 6, 27, tzinfo=UTC)
PAYLOAD = b"synthetic-backup-data-only" * 40


class OwnerJobAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.job_root = self.root / "jobs"
        self.archives = self.root / "archive"
        self.receipts = self.root / "receipts"
        for path in (self.job_root, self.archives, self.receipts):
            path.mkdir(mode=0o700)
        self.executable = self.job_root / "synthetic-job"
        self.executable.write_text("#!/bin/sh\nexit 0\n")
        self.executable.chmod(0o700)
        self.archive = self.archives / "synthetic.backup"
        self.primary_key = Ed25519PrivateKey.generate()
        self.mirror_key = Ed25519PrivateKey.generate()

        def pub(key):
            return key.public_key().public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )

        self.approved_keys = {
            "primary": {"keyId": "approved-primary", "publicKeyRaw": pub(self.primary_key)},
            "mirror": {"keyId": "approved-mirror", "publicKeyRaw": pub(self.mirror_key)},
        }
        self.invocations = []
        self.role = "primary"
        self.now = START
        self.skip_create = False
        self.retcode = 0
        self.preexisting = False

    def clock(self):
        value = self.now
        self.now = self.now + dt.timedelta(seconds=1)
        return value

    def runner(self, argv, **kwargs):
        self.invocations.append((argv, kwargs))
        if not self.skip_create:
            self.archive.write_bytes(PAYLOAD)
        return mock.Mock(returncode=self.retcode)

    def call(self, **updates):
        data = dict(
            owner_execute=True,
            role=self.role,
            run_id="synthetic-run-001",
            executable=self.executable,
            executable_root=self.job_root,
            archive=self.archive,
            archive_root=self.archives,
            receipt_directory=self.receipts,
            key_id="approved-primary" if self.role == "primary" else "approved-mirror",
            owner_private_key=self.primary_key if self.role == "primary" else self.mirror_key,
            max_archive_bytes=1024 * 1024,
            timeout_seconds=10,
            runner=self.runner,
            clock=self.clock,
        )
        data.update(updates)
        return run_owner_stage(**data)

    def test_no_optin_no_execution_no_receipt(self):
        r = self.call(owner_execute=False)
        self.assertEqual(r["state"], "unverified")
        self.assertFalse(r["jobProcessInvoked"])
        self.assertFalse(r["signedReceiptPublished"])
        self.assertEqual(self.invocations, [])
        self.assertEqual(list(self.receipts.iterdir()), [])

    def test_success_observes_direct_exit_and_hashes_new_bytes_only(self):
        r = self.call()
        self.assertEqual(
            r["state"], "owner-observed-exit-and-archive-claim-published-not-certified"
        )
        self.assertTrue(r["ownerExecutionOptIn"])
        self.assertTrue(r["jobProcessInvoked"])
        self.assertTrue(r["directJobExitObservedZero"])
        self.assertTrue(r["archiveReadbackSha256Completed"])
        self.assertTrue(r["signedReceiptPublished"])
        self.assertEqual(r["archiveBytesRead"], len(PAYLOAD))
        self.assertFalse(r["backupCertified"])
        self.assertFalse(r["databaseRestoreVerified"])
        self.assertFalse(r["deleteAuthorized"])
        argv, options = self.invocations[0]
        self.assertEqual(argv, [str(self.executable)])
        self.assertIs(options["shell"], False)
        self.assertIs(options["check"], False)
        self.assertIs(options["stdin"], __import__("subprocess").DEVNULL)
        receipt = load_signed_envelope(self.receipts / "synthetic-run-001-primary.signed.json")
        self.assertEqual(receipt["claim"]["archiveBytes"], len(PAYLOAD))
        self.assertEqual(receipt["claim"]["archiveSha256"], hashlib.sha256(PAYLOAD).hexdigest())
        self.assertTrue(
            verify_signed_claim(receipt, self.approved_keys)["approvedProducerKeyVerified"]
        )
        self.assertFalse(
            verify_signed_claim(receipt, self.approved_keys)["backupJobExecutionVerified"]
        )

    def test_nonzero_direct_exit_must_not_sign_or_claim_completed_job(self):
        self.retcode = 17
        r = self.call()
        self.assertEqual(r["state"], "held-job-exit-not-zero-or-unverified")
        self.assertTrue(r["jobProcessInvoked"])
        self.assertFalse(r["directJobExitObservedZero"])
        self.assertFalse(r["signedReceiptPublished"])
        self.assertEqual(list(self.receipts.iterdir()), [])

    def test_refuse_reused_archive_even_if_new_job_reports_exit_zero(self):
        self.archive.write_bytes(PAYLOAD)
        r = self.call()
        self.assertEqual(r["reason"], "refused-to-reuse-existing-archive")
        self.assertFalse(r["jobProcessInvoked"])
        self.assertEqual(self.invocations, [])
        self.assertEqual(list(self.receipts.iterdir()), [])

    def test_refuse_missing_archive_after_zero_exit(self):
        self.skip_create = True
        r = self.call()
        self.assertTrue(r["jobProcessInvoked"])
        self.assertTrue(r["directJobExitObservedZero"])
        self.assertFalse(r["signedReceiptPublished"])
        self.assertEqual(list(self.receipts.iterdir()), [])

    def test_oversized_archive_does_not_hash_or_sign(self):
        def runner(argv, **kwargs):
            self.archive.write_bytes(PAYLOAD * 10)
            return mock.Mock(returncode=0)

        r = self.call(runner=runner, max_archive_bytes=len(PAYLOAD))
        self.assertTrue(r["jobProcessInvoked"])
        self.assertFalse(r["archiveReadbackSha256Completed"])
        self.assertFalse(r["signedReceiptPublished"])
        self.assertEqual(list(self.receipts.iterdir()), [])

    def test_archive_symlink_cannot_be_read_or_signed(self):
        target = self.root / "target"
        target.write_bytes(PAYLOAD)

        def runner(argv, **kwargs):
            self.archive.symlink_to(target)
            return mock.Mock(returncode=0)

        r = self.call(runner=runner)
        self.assertFalse(r["signedReceiptPublished"])
        self.assertEqual(list(self.receipts.iterdir()), [])

    def test_wrong_owner_or_world_writable_executable_denies_job(self):
        self.executable.chmod(0o777)
        r = self.call()
        self.assertEqual(r["reason"], "approved-executable-identity-unverified")
        self.assertEqual(self.invocations, [])

    def test_disabled_without_valid_signer_does_not_run_job(self):
        r = self.call(owner_private_key=None)
        self.assertEqual(r["reason"], "invalid-owner-job-contract")
        self.assertEqual(self.invocations, [])

    def test_actual_disposable_subprocess_exit_and_archive_capture(self):
        import shlex
        import subprocess

        # The only real process executed by this test is a local disposable
        # script writing a 100% synthetic archive in a temporary directory.
        self.executable.write_text(
            "#!/bin/sh\n"
            + "printf %s synthetic-fixture-bytes-only > "
            + shlex.quote(str(self.archive))
            + "\nexit 0\n"
        )
        self.executable.chmod(0o700)
        out = self.call(runner=subprocess.run)
        self.assertEqual(
            out["state"],
            "owner-observed-exit-and-archive-claim-published-not-certified",
        )
        self.assertEqual(self.archive.read_bytes(), b"synthetic-fixture-bytes-only")
        self.assertTrue(out["jobProcessInvoked"])
        self.assertFalse(out["backupCertified"])

    def test_actual_disposable_nonzero_subprocess_cannot_publish(self):
        import subprocess

        self.executable.write_text("#!/bin/sh\nexit 23\n")
        self.executable.chmod(0o700)
        out = self.call(runner=subprocess.run)
        self.assertEqual(out["state"], "held-job-exit-not-zero-or-unverified")
        self.assertTrue(out["jobProcessInvoked"])
        self.assertFalse(out["directJobExitObservedZero"])
        self.assertFalse(out["signedReceiptPublished"])
        self.assertEqual(list(self.receipts.iterdir()), [])

    def test_partial_stream_failure_must_not_claim_zero_archive_io(self):
        with mock.patch(
            "neo4j_owner_job_adapter._read_new_archive",
            side_effect=OSError("synthetic-stream-stopped-after-read"),
        ):
            out = self.call()
        self.assertTrue(out["directJobExitObservedZero"])
        self.assertIsNone(out["archiveBytesRead"])
        self.assertFalse(out["signedReceiptPublished"])
        self.assertNotIn("synthetic-stream-stopped", str(out))

    def test_mirror_requires_independently_approved_primary_claim(self):
        self.role = "mirror"
        r = self.call()
        self.assertEqual(r["reason"], "primary-claim-not-independently-approved")
        self.assertFalse(r["jobProcessInvoked"])

    def test_mirror_preapproved_primary_pair_and_archive_digest(self):
        initial = self.call()
        self.assertTrue(initial["signedReceiptPublished"])
        prior = load_signed_envelope(self.receipts / "synthetic-run-001-primary.signed.json")
        self.archive.unlink()
        self.role = "mirror"
        # Different destination for new mirror, not reuse primary archive bytes.
        self.archive = self.archives / "synthetic-mirror.backup"
        out = self.call(approved_primary_envelope=prior, approved_primary_keys=self.approved_keys)
        self.assertEqual(
            out["state"], "owner-observed-exit-and-archive-claim-published-not-certified"
        )
        self.assertTrue((self.receipts / "synthetic-run-001-mirror.signed.json").is_file())
        mirrored = load_signed_envelope(self.receipts / "synthetic-run-001-mirror.signed.json")
        verified = verify_signed_pair(prior, mirrored, self.approved_keys)
        self.assertEqual(
            verified["state"],
            "two-approved-keys-signed-matching-claims-not-byte-proof",
        )
        self.assertFalse(verified["databaseRestoreVerified"])

    def test_forged_primary_claim_does_not_authorize_mirror_execution(self):
        self.role = "mirror"
        tampered = {
            "schema": "neo4j-backup-signed-claim-v1",
            "keyId": "approved-primary",
            "claim": {
                "schema": "neo4j-backup-completion-evidence-v1",
                "role": "primary",
                "runId": "synthetic-run-001",
                "startUtc": "2026-10-27T06:00:00Z",
                "finishUtc": "2026-10-27T06:01:00Z",
                "exitCode": 0,
                "archiveBytes": len(PAYLOAD),
                "archiveSha256": "0" * 64,
            },
            "signature": "A" * 86,
        }
        out = self.call(
            approved_primary_envelope=tampered, approved_primary_keys=self.approved_keys
        )
        self.assertEqual(out["reason"], "primary-claim-not-independently-approved")
        self.assertFalse(out["jobProcessInvoked"])
        self.assertEqual(self.invocations, [])

    def test_mirror_mismatched_digest_rejects_signing(self):
        r = self.call()
        self.assertTrue(r["signedReceiptPublished"])
        prior = load_signed_envelope(self.receipts / "synthetic-run-001-primary.signed.json")
        self.archive = self.archives / "mirror.backup"
        self.role = "mirror"

        def runner(argv, **kwargs):
            self.archive.write_bytes(b"different")
            return mock.Mock(returncode=0)

        out = self.call(
            approved_primary_envelope=prior, approved_primary_keys=self.approved_keys, runner=runner
        )
        self.assertEqual(out["state"], "hold-mirror-primary-byte-claim-mismatch")
        self.assertFalse(out["signedReceiptPublished"])

    def test_publication_failure_cannot_erase_recorded_job_attempt(self):
        def publish_fail(*args, **kwargs):
            raise OSError("synthetic private-disk error")

        with mock.patch(
            "neo4j_owner_job_adapter.publish_signed_envelope", side_effect=publish_fail
        ):
            out = self.call()
        self.assertTrue(out["jobProcessInvoked"])
        self.assertTrue(out["directJobExitObservedZero"])
        self.assertTrue(out["archiveReadbackSha256Completed"])
        self.assertEqual(out["archiveBytesRead"], len(PAYLOAD))
        self.assertFalse(out["signedReceiptPublished"])
        self.assertNotIn("private-disk", str(out))

    def test_timeout_retains_attempt_and_no_success_claim(self):
        import subprocess

        def timeout(*args, **kwargs):
            raise subprocess.TimeoutExpired(args[0], 3)

        out = self.call(runner=timeout)
        self.assertTrue(out["jobProcessInvoked"])
        self.assertFalse(out["directJobExitObservedZero"])
        self.assertFalse(out["signedReceiptPublished"])

    def test_refuse_path_traversal_and_intermediate_symlink(self):
        bad = self.call(archive=self.archives / ".." / "archive" / "fake.backup")
        self.assertEqual(bad["reason"], "owner-path-contract-invalid")
        self.assertEqual(self.invocations, [])
        alias = self.root / "alias"
        alias.symlink_to(self.job_root, target_is_directory=True)
        bad = self.call(executable_root=alias, executable=alias / "synthetic-job")
        self.assertFalse(bad["jobProcessInvoked"])
        self.assertEqual(self.invocations, [])


if __name__ == "__main__":
    unittest.main()
