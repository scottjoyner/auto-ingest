"""Disposable synthetic process tests for the NOT INSTALLED arXiv supervisor."""
from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "hermes_arxiv_supervisor_experiment",
    ROOT / "scripts" / "hermes_arxiv_supervisor_experiment.py",
)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class TestArxivBlockingSupervisor(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(dir="/tmp")
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.checkout = self.root / "checkout"
        scripts = self.checkout / "scripts"
        scripts.mkdir(parents=True)
        self.child = scripts / "arxiv_kg_bridge.py"
        self.child.write_text("import sys\nsys.exit(0)\n")
        (scripts / "embedding_client.py").write_text(
            "# synthetic shared client placeholder\n"
        )
        self.registry = self.checkout / "embedding_endpoints.json"
        self.registry.write_text(json.dumps({
            "model": "text-embedding-nomic-embed-text-v1.5",
            "embedding_dimensions": 768,
        }))
        self.lock = self.root / "job.lock"
        self.py = Path(sys.executable).resolve(strict=True)

    def sha(self, file):
        return hashlib.sha256(file.read_bytes()).hexdigest()

    def call(self, **kwargs):
        values = {
            "checkout": self.checkout,
            "python_executable": self.py,
            "reviewed_source_sha256": self.sha(self.child),
            "reviewed_registry_sha256": self.sha(self.registry),
            "lock_file": self.lock,
            "timeout_seconds": 1,
        }
        values.update(kwargs)
        return M.supervise(**values)

    def test_disposable_child_zero_is_waited_and_not_approved(self):
        result = self.call()
        self.assertEqual(result["status"], "CHILD_EXIT_ZERO")
        self.assertEqual(result["child_exit_code"], 0)
        self.assertTrue(result["child_waited"])
        self.assertFalse(result["can_dispatch"])
        self.assertFalse(result["source_signed_receipt"])
        self.assertFalse(result["per_user_embedding_auth_verified"])

    def test_nonzero_exit_propagates_as_failure(self):
        self.child.write_text("import sys\nsys.exit(17)\n")
        result = self.call()
        self.assertEqual(result["status"], "CHILD_EXIT_NONZERO")
        self.assertEqual(result["child_exit_code"], 17)
        self.assertTrue(result["child_waited"])

    def test_timeout_reaps_synthetic_sleeping_child(self):
        self.child.write_text("import time\ntime.sleep(5)\n")
        start = time.monotonic()
        result = self.call(timeout_seconds=0.12)
        self.assertEqual(result["status"], "CHILD_TIMEOUT_REAPED")
        self.assertTrue(result["child_waited"])
        self.assertLess(time.monotonic() - start, 3)

    def test_overlapping_lock_denied_without_running_child(self):
        self.child.write_text(
            "from pathlib import Path\n"
            f"Path({str(self.root / 'unexpected.txt')!r}).write_text('ran')\n"
        )
        with self.lock.open("a+b") as fd:
            fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = self.call()
            self.assertEqual(result["status"], "DENIED_OVERLAP")
            fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
        self.assertFalse((self.root / "unexpected.txt").exists())

    def test_wrong_source_digest_blocks_launch(self):
        result = self.call(reviewed_source_sha256="0" * 64)
        self.assertEqual(result["status"], "DENIED_PREFLIGHT")
        self.assertFalse(result["source_and_registry_checksum_match"])

    def test_wrong_registry_digest_blocks_launch(self):
        result = self.call(reviewed_registry_sha256="f" * 64)
        self.assertEqual(result["status"], "DENIED_PREFLIGHT")

    def test_missing_dimension_pin_blocks_launch(self):
        self.registry.write_text(json.dumps({
            "model": "text-embedding-nomic-embed-text-v1.5"
        }))
        result = self.call()
        self.assertEqual(result["status"], "DENIED_PREFLIGHT")

    def test_wrong_model_family_blocks_launch(self):
        self.registry.write_text(json.dumps({
            "model": "llama3", "embedding_dimensions": 768,
        }))
        result = self.call()
        self.assertEqual(result["status"], "DENIED_PREFLIGHT")

    def test_missing_shared_client_blocks_launch(self):
        (self.checkout / "scripts" / "embedding_client.py").unlink()
        result = self.call()
        self.assertEqual(result["status"], "DENIED_PREFLIGHT")

    def test_symlinked_child_refused_even_matching_digest(self):
        original = self.root / "real-child.py"
        original.write_bytes(self.child.read_bytes())
        self.child.unlink()
        self.child.symlink_to(original)
        result = self.call()
        self.assertEqual(result["status"], "DENIED_PREFLIGHT")

    def test_invalid_timeout_refused_without_spawn(self):
        for t in (0, -1, 3601, None, "1", True):
            with self.subTest(t=t):
                result = self.call(timeout_seconds=t)
                self.assertEqual(result["status"], "DENIED_TIMEOUT_BUDGET")

    def test_untrusted_lock_symlink_refused(self):
        target = self.root / "target"
        target.touch()
        self.lock.symlink_to(target)
        result = self.call()
        self.assertEqual(result["status"], "DENIED_LOCK_LOCATION")

    def test_result_never_exposes_paths_or_raw_child_output(self):
        self.child.write_text(
            "import sys\n"
            "print('SYNTHETIC_PRIVATE_EMBEDDING_TEXT')\n"
            "print('SYNTHETIC_PRIVATE_ERROR',file=sys.stderr)\n"
            "sys.exit(4)\n"
        )
        result = json.dumps(self.call())
        self.assertNotIn("SYNTHETIC_PRIVATE", result)
        self.assertNotIn(str(self.root), result)
        self.assertIn("CHILD_EXIT_NONZERO", result)

    def test_lock_released_after_nonzero(self):
        self.child.write_text("import sys\nsys.exit(2)\n")
        a = self.call()
        b = self.call()
        self.assertEqual(a["status"], "CHILD_EXIT_NONZERO")
        self.assertEqual(b["status"], "CHILD_EXIT_NONZERO")

    def test_no_cli_or_autorun_entrypoint(self):
        source = (ROOT / "scripts" / "hermes_arxiv_supervisor_experiment.py").read_text()
        self.assertNotIn('if __name__ == "__main__"', source)
        self.assertNotIn("argparse", source)
        self.assertNotIn("requests.post(", source)
        self.assertNotIn("urllib.request", source)


if __name__ == "__main__":
    unittest.main()
