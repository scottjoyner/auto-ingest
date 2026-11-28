"""Mixed-version component assessment, no live host or graph operations."""
import unittest
from datetime import datetime, timedelta, timezone

from auto_ingest.executor_deployment import assess_component_deployments

NOW = datetime(2026, 11, 26, tzinfo=timezone.utc)
PINS = {"launcher.py": "a" * 40, "auto_ingest/evidence.py": "b" * 40,
        "auto_ingest/report.py": "c" * 40}


def row(**changes):
    return dict(deployment_id="node/worker", observed_at=NOW,
                component_blobs=dict(PINS), process_state="QUIESCENT",
                launch_state="DISABLED") | changes


def assess(rows, expected=("node/worker",), pins=PINS):
    return assess_component_deployments(expected, rows, candidate_components=pins, now=NOW)


class ComponentTests(unittest.TestCase):
    def test_exact_components_never_authorize(self):
        result = assess([row()])
        self.assertEqual(result["inventory_status"], "DECLARED_INVENTORY_CONSISTENT")
        self.assertEqual(result["loaded_code_identity"], "UNVERIFIED")
        self.assertFalse(result["activation_allowed"])
        self.assertFalse(result["replay_allowed"])

    def test_current_launcher_stale_recovery(self):
        result = assess([row(component_blobs=dict(PINS, **{"auto_ingest/evidence.py": "d"*40}))])
        self.assertEqual(result["findings"][0]["blockers"],
                         ["COMPONENT_VERSION_MISMATCH:auto_ingest/evidence.py"])
        self.assertEqual(result["inventory_status"], "BLOCKERS_OBSERVED")

    def test_missing_unexpected_and_invalid(self):
        result = assess([row(component_blobs={"launcher.py": None, "old.py": "a"*40})])
        self.assertEqual(set(result["findings"][0]["blockers"]), {
            "MISSING_COMPONENT:auto_ingest/evidence.py", "MISSING_COMPONENT:auto_ingest/report.py",
            "COMPONENT_UNVERIFIED:launcher.py", "UNEXPECTED_COMPONENT:old.py"})

    def test_missing_manifest_not_clean(self):
        for components in (None, [], {"bad key": "a"*40}):
            self.assertIn("COMPONENT_INVENTORY_UNVERIFIED",
                          assess([row(component_blobs=components)])["findings"][0]["blockers"])

    def test_process_launch_and_staleness_still_block(self):
        result = assess([row(process_state="ACTIVE", launch_state="ENABLED",
                             observed_at=NOW-timedelta(hours=1))])
        self.assertEqual(result["findings"][0]["blockers"],
                         ["STALE_OR_INVALID_OBSERVATION", "PROCESS_ACTIVE", "LAUNCH_ENABLED"])

    def test_missing_deployment_and_duplicates(self):
        self.assertEqual(assess([])["findings"][0]["blockers"], ["MISSING_OBSERVATION"])
        with self.assertRaises(ValueError):
            assess([row(), row()])

    def test_invalid_pins(self):
        for pins in ({}, {"bad key": "a"*40}, {"valid.py": "short"}):
            with self.assertRaises(ValueError):
                assess([], pins=pins)


if __name__ == "__main__":
    unittest.main()
