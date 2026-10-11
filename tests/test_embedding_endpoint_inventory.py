"""Offline fleet model inventory failover tests: no model calls or network."""
from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "embedding_endpoint_selector", ROOT / "scripts" / "get_embedding_endpoint.py",
)
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


class FakeResponse:
    def __init__(self, payload, status=200):
        self.status = status
        self.content = io.BytesIO(
            payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        )

    def read(self, size=-1):
        return self.content.read(size)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.content.close()


class TestEmbeddingInventoryFailover(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(dir="/tmp")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.registry = self.root / "embedding_endpoints.json"
        self.old_env = M.ENV_URL
        self.old_registry = M.REGISTRY
        self.addCleanup(setattr, M, "ENV_URL", self.old_env)
        self.addCleanup(setattr, M, "REGISTRY", self.old_registry)
        M.ENV_URL = None
        M.REGISTRY = str(self.registry)

    def record(self, rank, name):
        return {
            "name": name,
            "priority": rank,
            "url": f"http://example.invalid:{1000+rank}/v1/embeddings",
            "health": f"http://example.invalid:{1000+rank}/v1/models",
        }

    def registry_with(self, *records):
        self.registry.write_text(json.dumps({
            "model": "text-embedding-nomic-embed-text-v1.5",
            "endpoints": list(records),
        }))

    def stub(self, mapping):
        requests = []

        def response(req, timeout):
            assert req.get_method() == "GET", "must not run embeddings or inference"
            assert req.full_url.endswith("/v1/models")
            requests.append(req.full_url)
            value = mapping.get(req.full_url)
            if isinstance(value, Exception):
                raise value
            if value is None:
                raise OSError("simulated offline provider")
            return FakeResponse(value)

        return requests, response

    def test_200_empty_inventory_not_healthy(self):
        with patch.object(M.urllib.request, "urlopen", return_value=FakeResponse(
            {"object": "list", "data": []}
        )):
            self.assertFalse(M._healthy("http://example.invalid/v1/models"))

    def test_one_advertised_model_listed_but_not_execution_attested(self):
        with patch.object(M.urllib.request, "urlopen", return_value=FakeResponse(
            {"object": "list", "data": [{"id": "nomic-embed-text", "object": "model"}]}
        )):
            self.assertTrue(M._healthy("http://example.invalid/v1/models"))

    def test_200_non_json_fails_closed(self):
        with patch.object(M.urllib.request, "urlopen", return_value=FakeResponse(
            b"<html>please authenticate</html>"
        )):
            self.assertFalse(M._healthy("http://example.invalid/v1/models"))

    def test_200_empty_model_id_fails_closed(self):
        with patch.object(M.urllib.request, "urlopen", return_value=FakeResponse(
            {"data": [{"id": "   "}, {"object": "model"}]}
        )):
            self.assertFalse(M._healthy("http://example.invalid/v1/models"))

    def test_200_wrong_schema_fails_closed(self):
        with patch.object(M.urllib.request, "urlopen", return_value=FakeResponse(
            {"models": [{"id": "nomic-embed-text"}]}
        )):
            self.assertFalse(M._healthy("http://example.invalid/v1/models"))

    def test_200_oversize_inventory_fails_closed(self):
        with patch.object(M.urllib.request, "urlopen", return_value=FakeResponse(
            b" " * (M.MAX_INVENTORY_BYTES + 1)
        )):
            self.assertFalse(M._healthy("http://example.invalid/v1/models"))

    def test_http_failure_fails_closed(self):
        with patch.object(M.urllib.request, "urlopen", return_value=FakeResponse(
            {"data": [{"id": "nomic"}]}, status=503
        )):
            self.assertFalse(M._healthy("http://example.invalid/v1/models"))

    def test_only_models_get_paths_accepted(self):
        self.assertFalse(M._healthy("http://example.invalid/v1/embeddings"))
        self.assertFalse(M._healthy("http://example.invalid/health"))
        self.assertFalse(M._healthy(None))

    def test_fleet_live_snapshot_empty_primary_falls_through_to_next_model(self):
        hub = self.record(1, "x1")
        offline = self.record(2, "destroyer")
        alternate = self.record(3, "deathstar")
        fallback = self.record(6, "beelink")
        self.registry_with(hub, offline, alternate, fallback)
        calls, fake = self.stub({
            hub["health"]: {"data": []},
            offline["health"]: OSError("provider offline"),
            alternate["health"]: {"data": [{"id": "nomic-embed-text-v1.5"}]},
            fallback["health"]: {"data": [{"id": "nomic-embed-text-v1.5"}]},
        })
        with patch.object(M.urllib.request, "urlopen", side_effect=fake):
            selected = M.get_embedding_endpoint()
        self.assertEqual(selected, alternate["url"])
        self.assertEqual(calls, [hub["health"], offline["health"], alternate["health"]])
        self.assertNotIn(fallback["health"], calls)

    def test_all_unavailable_returns_none(self):
        self.registry_with(self.record(1, "hub"))
        with patch.object(M.urllib.request, "urlopen", side_effect=OSError("offline")):
            self.assertIsNone(M.get_embedding_endpoint(fail=False))

    def test_all_unavailable_raises_explicit_error(self):
        self.registry_with(self.record(1, "hub"))
        with patch.object(M.urllib.request, "urlopen", side_effect=OSError("offline")):
            with self.assertRaisesRegex(RuntimeError, "No healthy embedding endpoint"):
                M.get_embedding_endpoint()

    def test_missing_registry_does_not_fall_back_to_unknown_host(self):
        self.assertIsNone(M.get_embedding_endpoint(fail=False))

    def test_invalid_registry_entry_does_not_crash(self):
        selected = self.record(3, "usable")
        self.registry_with(None, {"url": None, "priority": 2}, selected)
        with patch.object(M.urllib.request, "urlopen", return_value=FakeResponse(
            {"data": [{"id": "nomic-embed"}]}
        )):
            self.assertEqual(M.get_embedding_endpoint(), selected["url"])

    def test_environment_override_does_not_accept_empty_model_inventory(self):
        M.ENV_URL = "http://example.invalid:1235/v1/embeddings"
        with patch.object(M.urllib.request, "urlopen", return_value=FakeResponse(
            {"data": []}
        )):
            self.assertIsNone(M.get_embedding_endpoint(fail=False))

    def test_no_inference_or_proxy_mutation_in_resolver(self):
        source = (ROOT / "scripts/get_embedding_endpoint.py").read_text()
        self.assertNotIn('method="POST"', source)
        self.assertNotIn("/v1/chat/completions", source)
        self.assertNotIn("subprocess", source)


if __name__ == "__main__":
    unittest.main()
