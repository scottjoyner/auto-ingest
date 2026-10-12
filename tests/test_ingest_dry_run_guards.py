"""Offline --dry-run tests: no model, graph DDL, vector or cursor mutation.

Both ingestion modules are loaded without accessing Signal, arXiv, Neo4j or
a real embedding endpoint. Source ingestion happens only through mocked
functions, never via the module's command-line main().
"""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name: str):
    path = ROOT / "scripts" / name
    module_id = "test_dryrun_" + name.replace(".py", "")
    spec = importlib.util.spec_from_file_location(module_id, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_id] = module
    spec.loader.exec_module(module)
    return module


class TestSignalDryRun(unittest.TestCase):
    def setUp(self):
        self.mod = load("signal_kg_bridge.py")
        self.message = {
            "src": "synthetic-user",
            "ts": 1_800_000_000,
            "body": "CONFIDENTIAL_SYNTHETIC_MESSAGE_NEVER_LOG",
        }

    def test_signal_ingest_dry_run_avoids_model_neo4j_constraints_and_cursor(self):
        with (
            patch.object(self.mod, "embed", side_effect=AssertionError("model called")),
            patch.object(self.mod, "get_neo4j", side_effect=AssertionError("DB opened")),
            patch.object(self.mod, "ensure_constraints", side_effect=AssertionError("DDL")),
            patch.object(self.mod, "save_cursor", side_effect=AssertionError("cursor")),
            patch.object(self.mod, "log") as log,
        ):
            self.assertEqual(self.mod.ingest([self.message], dry_run=True), 1)
            messages = " ".join(str(a) for call in log.call_args_list for a in call.args)
            self.assertIn("[dry-run]", messages)
            self.assertNotIn("CONFIDENTIAL_SYNTHETIC_MESSAGE", messages)
            self.assertNotIn("synthetic-user", messages)

    def test_signal_empty_dry_run_never_contacts_model_or_db(self):
        with (
            patch.object(self.mod, "embed", side_effect=AssertionError),
            patch.object(self.mod, "get_neo4j", side_effect=AssertionError),
            patch.object(self.mod, "save_cursor", side_effect=AssertionError),
            patch.object(self.mod, "log"),
        ):
            self.assertEqual(self.mod.ingest([], dry_run=True), 0)

    def test_signal_cli_dry_run_skips_receive_before_queue_access(self):
        with (
            patch.object(sys, "argv", [
                "signal_kg_bridge.py", "--dry-run", "--since", "2026-10-10",
            ]),
            patch.object(self.mod, "receive_signal", side_effect=AssertionError(
                "Signal queue was accessed"
            )),
            patch.object(self.mod, "load_cursor", side_effect=AssertionError(
                "cursor read should not be needed"
            )),
            patch.object(self.mod, "get_neo4j", side_effect=AssertionError(
                "database was opened"
            )),
            patch.object(self.mod, "embed", side_effect=AssertionError(
                "model was invoked"
            )),
            patch.object(self.mod, "log") as log,
        ):
            self.assertEqual(self.mod.main(), 0)
            self.assertIn("queue safety unverified", str(log.call_args_list))

    def test_signal_dry_run_counts_without_reading_private_body(self):
        bad_message = {"src": "synthetic-user", "ts": 1_800_000_001}
        with (
            patch.object(self.mod, "embed", side_effect=AssertionError),
            patch.object(self.mod, "get_neo4j", side_effect=AssertionError),
            patch.object(self.mod, "log") as log,
        ):
            self.assertEqual(self.mod.ingest([bad_message], dry_run=True), 1)
            self.assertNotIn("synthetic-user", str(log.call_args_list))


class TestArxivDryRun(unittest.TestCase):
    def setUp(self):
        self.mod = load("arxiv_kg_bridge.py")
        self.paper = {
            "arxiv_id": "synthetic-paper-0001",
            "title": "SYNTHETIC_PRIVATE_TITLE_DO_NOT_PRINT",
            "abstract": "synthetic bounded dry-run abstract",
        }

    def test_arxiv_dry_run_avoids_embeddings_db_and_seen_cursor(self):
        with (
            patch.object(self.mod, "load_seen", return_value=set()),
            patch.object(self.mod, "embed", side_effect=AssertionError("model called")),
            patch.object(self.mod, "get_neo4j", side_effect=AssertionError("DB opened")),
            patch.object(self.mod, "save_seen", side_effect=AssertionError("cursor")),
            patch.object(self.mod, "log") as log,
        ):
            self.assertEqual(self.mod.ingest([self.paper], dry_run=True), 1)
            messages = " ".join(str(a) for call in log.call_args_list for a in call.args)
            self.assertIn("[dry-run]", messages)
            self.assertNotIn("SYNTHETIC_PRIVATE_TITLE", messages)

    def test_arxiv_dry_run_excludes_already_seen_and_duplicate_ids(self):
        papers = [
            self.paper, dict(self.paper),
            {"arxiv_id": "already-present", "title": "Already", "abstract": "A"},
        ]
        with (
            patch.object(self.mod, "load_seen", return_value={"already-present"}),
            patch.object(self.mod, "embed", side_effect=AssertionError),
            patch.object(self.mod, "get_neo4j", side_effect=AssertionError),
            patch.object(self.mod, "save_seen", side_effect=AssertionError),
            patch.object(self.mod, "log"),
        ):
            self.assertEqual(self.mod.ingest(papers, dry_run=True), 1)

    def test_arxiv_empty_dry_run_never_opens_database_or_writes_cursor(self):
        with (
            patch.object(self.mod, "load_seen", return_value=set()),
            patch.object(self.mod, "get_neo4j", side_effect=AssertionError),
            patch.object(self.mod, "save_seen", side_effect=AssertionError),
            patch.object(self.mod, "log"),
        ):
            self.assertEqual(self.mod.ingest([], dry_run=True), 0)


if __name__ == "__main__":
    unittest.main()
