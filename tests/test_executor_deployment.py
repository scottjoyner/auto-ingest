"""Dependency-free tests for deployment inventory assessment."""
import unittest
from datetime import datetime, timedelta, timezone

from auto_ingest.executor_deployment import assess_deployments

NOW = datetime(2026, 11, 26, tzinfo=timezone.utc)
CANDIDATE = "a" * 40
LEGACY = "b" * 40


def row(identity="node/worker", **changes):
    return dict(deployment_id=identity, observed_at=NOW, source_blob=CANDIDATE,
                process_state="QUIESCENT", launch_state="DISABLED") | changes


def assess(rows, expected=("node/worker",)):
    return assess_deployments(expected, rows, candidate_blob=CANDIDATE, now=NOW)


class DeploymentTests(unittest.TestCase):
    def test_clean_does_not_authorize(self):
        report = assess([row()])
        self.assertEqual(report["inventory_status"], "DECLARED_INVENTORY_CONSISTENT")
        self.assertEqual(report["inventory_completeness"], "UNVERIFIED")
        self.assertEqual(report["observation_authenticity"], "UNVERIFIED")
        self.assertFalse(report["activation_allowed"])
        self.assertFalse(report["replay_allowed"])

    def test_legacy_and_enabled_writer(self):
        report = assess([row(source_blob=LEGACY, process_state="ACTIVE", launch_state="ENABLED")])
        self.assertEqual(report["findings"][0]["blockers"],
                         ["SOURCE_VERSION_MISMATCH", "PROCESS_ACTIVE", "LAUNCH_ENABLED"])

    def test_inaccessible_and_unknown(self):
        report = assess([row(source_blob=None, process_state="UNKNOWN", launch_state="UNKNOWN")])
        self.assertEqual(report["findings"][0]["blockers"],
                         ["SOURCE_UNVERIFIED", "PROCESS_UNVERIFIED", "LAUNCH_UNVERIFIED"])

    def test_missing_unexpected_and_duplicates(self):
        report = assess([row("other/worker")])
        self.assertEqual(report["findings"], [
            {"deployment_id": "node/worker", "blockers": ["MISSING_OBSERVATION"]},
            {"deployment_id": "other/worker", "blockers": ["UNEXPECTED_DEPLOYMENT"]}])
        with self.assertRaises(ValueError):
            assess([row(), row()])
        with self.assertRaises(ValueError):
            assess([row()], expected=("node/worker", "node/worker"))

    def test_time_bounds(self):
        for stamp in (None, NOW.replace(tzinfo=None), NOW + timedelta(seconds=1),
                      NOW - timedelta(minutes=15, seconds=1)):
            with self.subTest(stamp=stamp):
                self.assertIn("STALE_OR_INVALID_OBSERVATION",
                              assess([row(observed_at=stamp)])["findings"][0]["blockers"])
        self.assertEqual(assess([row(observed_at=NOW-timedelta(minutes=15))])[
            "findings"][0]["blockers"], [])

    def test_invalid_inputs(self):
        for expected in ([], ["invalid space"], ["a"]*101):
            with self.subTest(expected=expected), self.assertRaises(ValueError):
                assess([], expected=expected)
        for rows in (None, [{}], [row()]*101):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                assess(rows)
        with self.assertRaises(ValueError):
            assess_deployments(["node"], [], candidate_blob="short", now=NOW)

    def test_fixture_mixed_fleet(self):
        expected = ("xwing/legacy", "destroyer/modified", "deathstar/supervisor")
        rows = [row(expected[0], source_blob=LEGACY, launch_state="UNKNOWN"),
                row(expected[1], source_blob="c"*40, process_state="UNKNOWN"),
                row(expected[2], source_blob=None, launch_state="ENABLED")]
        report = assess(rows, expected=expected)
        self.assertEqual(report["inventory_status"], "BLOCKERS_OBSERVED")
        self.assertTrue(all(item["blockers"] for item in report["findings"]))
        self.assertFalse(report["activation_allowed"])
