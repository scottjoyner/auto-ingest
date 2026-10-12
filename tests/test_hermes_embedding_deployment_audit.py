"""Hermes active-job + checkout drift acceptance: no job prompt disclosed."""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_hermes_embedding_deployment",
    ROOT / "scripts" / "audit_hermes_embedding_deployment.py",
)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class TestHermesDeploymentAudit(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(dir="/tmp")
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name)
        self.jobs = self.base / "jobs.json"
        self.a = self.base / "shared"
        self.b = self.base / "home"
        for root in (self.a, self.b):
            (root / "scripts").mkdir(parents=True)
            (root / "embedding_endpoints.json").write_text(json.dumps({
                "model": "text-embedding-nomic-embed-text-v1.5",
                "embedding_dimensions": 768,
            }))
            (root / "scripts" / "embedding_client.py").write_text(
                "def embed_texts(texts): return []\n"
            )
            for name in M.EXPECTED:
                (root / "scripts" / name).write_text(
                    "from scripts.embedding_client import embed_texts\n"
                )
        self.write_jobs()

    def write_jobs(self, enabled=True):
        self.jobs.write_text(json.dumps({
            "jobs": [
                {"name": "kg-arxiv-ingest", "enabled": enabled,
                 "prompt": "SECRET_PROMPT_ABC: private endpoint and account"},
                {"name": "kg-paper768-backfill", "enabled": enabled,
                 "prompt": "SECRET_PROMPT_XYZ: message content"},
                {"name": "unrelated-automation", "enabled": True,
                 "prompt": "private unrelated job content"},
            ]
        }))

    def test_enabled_jobs_unbound_and_denied(self):
        r = M.collect(self.jobs, [self.a, self.b])
        self.assertEqual(r["scheduler"]["total_scheduler_records"], 3)
        self.assertEqual(r["scheduler"]["known_embedding_job_count"], 2)
        self.assertEqual(r["scheduler"]["enabled_execution_paths_unbound"], 2)
        self.assertFalse(r["active_execution_path_verified"])
        self.assertEqual(r["status"], "BLOCKED_UNBOUND_SCHEDULED_EXECUTION")
        self.assertTrue(r["source_copies"]["all_copies_migrated"])
        for key in ("operator_approval_present", "can_deploy", "can_dispatch",
                    "authenticated_embedding_owner_proven", "source_provenance_verified"):
            self.assertFalse(r[key], key)

    def test_job_prompt_is_never_serialized(self):
        r = json.dumps(M.collect(self.jobs, [self.a]))
        self.assertNotIn("SECRET_PROMPT_", r)
        self.assertNotIn("private endpoint", r)
        self.assertNotIn("message content", r)
        self.assertNotIn(str(self.base), r)

    def test_disabled_jobs_do_not_count_as_bound_execution(self):
        self.write_jobs(enabled=False)
        r = M.collect(self.jobs, [self.a])
        self.assertEqual(r["scheduler"]["enabled_execution_paths_unbound"], 0)
        self.assertFalse(r["can_deploy"])

    def test_malformed_scheduler_entries_refused(self):
        for payload in ([], {}, {"jobs": "bad"}, {"jobs": [{}] * 513}):
            with self.subTest(payload_type=str(type(payload))):
                self.jobs.write_text(json.dumps(payload))
                with self.assertRaises(ValueError):
                    M.collect(self.jobs, [self.a])

    def test_unknown_job_enabled_state_refused(self):
        self.write_jobs(enabled=None)
        with self.assertRaisesRegex(ValueError, "enabled state"):
            M.collect(self.jobs, [self.a])

    def test_symlink_scheduler_file_refused(self):
        link = self.base / "jobs-link.json"
        link.symlink_to(self.jobs)
        with self.assertRaisesRegex(ValueError, "symlink"):
            M.collect(link, [self.a])

    def test_oversized_scheduler_file_refused(self):
        self.jobs.write_bytes(b"x" * (M.MAX_JOBS_BYTES + 1))
        with self.assertRaisesRegex(ValueError, "budget"):
            M.collect(self.jobs, [self.a])

    def test_unmigrated_script_and_different_copies_detected(self):
        (self.b / "scripts" / "arxiv_kg_bridge.py").write_text(
            'LM_STUDIO_URL = "http://127.0.0.1/v1/embeddings"\n'
        )
        result = M.collect(self.jobs, [self.a, self.b])
        self.assertTrue(result["source_copies"]["script_copies_differ"])
        self.assertFalse(result["source_copies"]["all_copies_migrated"])
        self.assertFalse(result["can_deploy"])

    def test_registry_dimension_not_pinned_denied(self):
        (self.a / "embedding_endpoints.json").write_text(json.dumps({
            "model": "text-embedding-nomic-embed-text-v1.5",
        }))
        r = M.collect(self.jobs, [self.a])
        self.assertFalse(r["source_copies"]["all_copies_migrated"])

    def test_missing_script_denied(self):
        (self.a / "scripts" / "arxiv_kg_bridge.py").unlink()
        r = M.collect(self.jobs, [self.a])
        self.assertFalse(r["source_copies"]["all_copies_migrated"])

    def test_script_symlink_marked_missing(self):
        old = self.a / "scripts" / "signal_kg_bridge.py"
        old.unlink()
        old.symlink_to(self.b / "scripts" / "signal_kg_bridge.py")
        r = M.collect(self.jobs, [self.a])
        target = next(x for x in r["source_copies"]["checkouts"][0]["known_scripts"]
                      if x["script"] == "signal_kg_bridge.py")
        self.assertFalse(target["exists"])
        self.assertFalse(r["source_copies"]["all_copies_migrated"])

    def test_root_symlink_refused(self):
        link = self.base / "shared-link"
        link.symlink_to(self.a, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "checkout root"):
            M.collect(self.jobs, [link])

    def test_too_many_checkout_roots_refused(self):
        with self.assertRaisesRegex(ValueError, "bounded exact"):
            M.collect(self.jobs, [self.a] * 6)

    def test_malformed_json_denied_before_script_scan(self):
        self.jobs.write_text("{not json")
        with self.assertRaises(ValueError):
            M.collect(self.jobs, [self.a])

    def test_job_records_must_be_pure_metadata(self):
        r = M.inspect_jobs({
            "jobs": [{
                "name": "kg-signal-bridge",
                "enabled": True,
                "prompt": "secret Signal phone message",
                "schedule": {"expr": "* * * * *"},
            }]
        })
        self.assertEqual(r["known_jobs"][0]["known_job_name"], "kg-signal-bridge")
        self.assertFalse(r["known_jobs"][0]["execution_path_attested"])
        self.assertNotIn("secret", json.dumps(r))

if __name__ == "__main__":
    unittest.main()
