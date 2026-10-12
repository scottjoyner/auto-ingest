"""Synthetic Hermes no-agent wrapper audit: no commands or private prompts."""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_hermes_wrapper_completion",
    ROOT / "scripts" / "audit_hermes_wrapper_completion.py",
)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class TestHermesWrapperCompletion(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir="/tmp")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.wrapper = self.root / "scripts"
        self.wrapper.mkdir()
        (self.wrapper / "kg-arxiv-ingest.py").write_text(
            'import subprocess\n'
            'subprocess.Popen(["python3", "/media/x/SSD_4TB/auto-ingest/'
            'scripts/arxiv_kg_bridge.py"], start_new_session=True)\n'
        )
        (self.wrapper / "kg-paper768-backfill.py").write_text(
            'import subprocess\n'
            'raise SystemExit(subprocess.call(["python3", '
            '"/media/x/SSD_4TB/auto-ingest/scripts/backfill_paper_768.py"]))\n'
        )
        (self.wrapper / "kg-signal-bridge.py").write_text(
            'import subprocess\n'
            'raise SystemExit(subprocess.call(["python3", '
            '"/media/x/SSD_4TB/auto-ingest/scripts/signal_kg_bridge.py"]))\n'
        )
        self.jobs = {
            "jobs": [
                {
                    "name": "kg-arxiv-ingest",
                    "script": "kg-arxiv-ingest.py", "no_agent": True,
                    "enabled": True, "last_status": "ok",
                    "prompt": "SUPER_SECRET_PRIVATE_TASK_BODY",
                },
                {
                    "name": "kg-paper768-backfill",
                    "script": "kg-paper768-backfill.py", "no_agent": True,
                    "enabled": True, "last_status": "ok",
                },
                {
                    "name": "kg-signal-bridge",
                    "script": "kg-signal-bridge.py", "no_agent": True,
                    "enabled": True, "last_status": "ok",
                },
                {
                    "name": "Neo4j enrichment semantic re-embed (weekly)",
                    "no_agent": False, "enabled": True,
                    "last_status": "error", "prompt": "SECRET_SEMANTIC_PROMPT",
                },
            ]
        }

    def inspect(self):
        return M.inspect(self.jobs, self.wrapper)

    def test_detects_three_no_agent_and_one_agent(self):
        r = self.inspect()
        self.assertEqual(r["enabled_known_jobs"], 4)
        self.assertEqual(r["configured_no_agent_script_jobs"], 3)
        self.assertEqual(r["agent_driven_jobs"], 1)
        self.assertFalse(r["can_deploy"])

    def test_arxiv_detached_child_is_not_completion_receipt(self):
        r = self.inspect()
        task = next(x for x in r["jobs"] if x["job_name"] == "kg-arxiv-ingest")
        self.assertEqual(task["launch_kind"], "detached_popen_no_completion_wait")
        self.assertEqual(r["detached_without_completion_wait"], 1)
        self.assertEqual(task["last_status_observed"], "ok")
        self.assertFalse(task["child_completion_attested"])
        self.assertFalse(task["child_wait_or_completion_check_detected"])

    def test_blocking_wrappers_not_signer_receipts(self):
        r = self.inspect()
        for name in ("kg-signal-bridge", "kg-paper768-backfill"):
            job = next(x for x in r["jobs"] if x["job_name"] == name)
            self.assertEqual(job["launch_kind"], "blocking_child_call_candidate")
            self.assertTrue(job["child_wait_or_completion_check_detected"])
            self.assertFalse(job["child_completion_attested"])
            self.assertFalse(job["owner_or_approved_digest_attested"])

    def test_private_job_prompts_and_paths_never_disclosed(self):
        result = json.dumps(self.inspect())
        for forbidden in (
            "SUPER_SECRET", "SECRET_SEMANTIC", str(self.root),
            "/media/x/", "start_new_session=True", "http://",
        ):
            self.assertNotIn(forbidden, result)

    def test_symlink_wrapper_is_not_followed(self):
        source = self.wrapper / "kg-arxiv-ingest.py"
        source.unlink()
        source.symlink_to(self.wrapper / "kg-signal-bridge.py")
        row = next(x for x in self.inspect()["jobs"]
                   if x["job_name"] == "kg-arxiv-ingest")
        self.assertFalse(row["wrapper_file_observed"])
        self.assertEqual(row["launch_kind"], "missing_or_invalid_wrapper")

    def test_wrong_script_reference_fails_closed(self):
        self.jobs["jobs"][0]["script"] = "../../outside.py"
        row = next(x for x in self.inspect()["jobs"]
                   if x["job_name"] == "kg-arxiv-ingest")
        self.assertEqual(row["job_mode"], "unverified")
        self.assertFalse(row["wrapper_file_observed"])

    def test_agent_job_does_not_receive_no_agent_status(self):
        self.jobs["jobs"][-1]["script"] = "kg-arxiv-ingest.py"
        row = next(x for x in self.inspect()["jobs"]
                   if x["job_name"].startswith("Neo4j"))
        self.assertEqual(row["job_mode"], "agent_driven_no_direct_script")
        self.assertFalse(row["child_completion_attested"])

    def test_large_wrapper_is_rejected(self):
        (self.wrapper / "kg-arxiv-ingest.py").write_bytes(b"x" * (M.MAX_BYTES + 1))
        row = next(x for x in self.inspect()["jobs"]
                   if x["job_name"] == "kg-arxiv-ingest")
        self.assertEqual(row["launch_kind"], "missing_or_invalid_wrapper")
        self.assertFalse(row["wrapper_file_observed"])

    def test_unbound_expected_child_does_not_prove_caller(self):
        (self.wrapper / "kg-paper768-backfill.py").write_text(
            'import subprocess\nsubprocess.call(["python3", "other.py"])\n'
        )
        row = next(x for x in self.inspect()["jobs"]
                   if x["job_name"] == "kg-paper768-backfill")
        self.assertFalse(row["expected_caller_referenced"])
        self.assertFalse(row["can_dispatch"])

    def test_missing_job_file_refused(self):
        with self.assertRaises(ValueError):
            M.read_bounded(self.wrapper / "missing.py", M.MAX_BYTES)

    def test_oversized_jobs_refused(self):
        target = self.root / "oversize.json"
        target.write_bytes(b"x" * (M.MAX_JOBS_BYTES + 1))
        with self.assertRaises(ValueError):
            M.read_bounded(target, M.MAX_JOBS_BYTES)

    def test_malformed_jobs_payload_refused(self):
        for bad in ([], {"jobs": "not-list"}, {"jobs": [{}] * 513}):
            with self.subTest(type=str(type(bad))):
                with self.assertRaises(ValueError):
                    self.assertIsNone(M.inspect(bad, self.wrapper))

    def test_invalid_enabled_state_fails_closed(self):
        self.jobs["jobs"][0]["enabled"] = "true"
        with self.assertRaisesRegex(ValueError, "invalid bool"):
            self.inspect()

    def test_even_waited_popen_is_not_execution_attestation(self):
        (self.wrapper / "kg-arxiv-ingest.py").write_text(
            'import subprocess\n'
            'p=subprocess.Popen(["python3","arxiv_kg_bridge.py"])\n'
            'p.wait()\n'
        )
        row = next(x for x in self.inspect()["jobs"]
                   if x["job_name"] == "kg-arxiv-ingest")
        self.assertNotEqual(row["launch_kind"], "detached_popen_no_completion_wait")
        self.assertFalse(row["child_completion_attested"])

    def test_duplicate_known_jobs_cannot_approve(self):
        self.jobs["jobs"].append(dict(self.jobs["jobs"][0]))
        r = self.inspect()
        self.assertEqual(r["enabled_known_jobs"], 5)
        self.assertFalse(r["can_deploy"])

    def test_all_reports_fail_closed(self):
        r = self.inspect()
        self.assertEqual(r["status"], "BLOCKED_UNVERIFIED_HERMES_CHILD_COMPLETION")
        for key in ("runtime_child_execution_verified",
                    "source_digest_independently_approved",
                    "operator_change_approved", "can_change_scheduler",
                    "can_deploy", "can_dispatch"):
            self.assertFalse(r[key])


if __name__ == "__main__":
    unittest.main()
