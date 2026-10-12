"""Offline transport and graph-caller acceptance for centrally pinned embeddings.

All network and source resolution are mocked. Nothing connects to Neo4j,
Signal, arXiv, or a real inference server.
"""
from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import embedding_client as M

ROOT = Path(__file__).resolve().parents[1]


class Response:
    def __init__(self, value, status=200):
        self.status = status
        self.body = (
            value if isinstance(value, bytes)
            else json.dumps(value, allow_nan=True).encode("utf-8")
        )

    def read(self, limit):
        return self.body[:limit]

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        pass


class HTTP:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def open(self, request, timeout):
        self.calls.append((request, timeout))
        return self.response


def reply(indices=(0,), dimensions=768):
    return {
        "data": [
            {
                "index": idx,
                "object": "embedding",
                "embedding": [0.25] * dimensions,
            } for idx in indices
        ]
    }


class TestEmbeddingClient(unittest.TestCase):
    def setUp(self):
        self.selector = patch.object(
            M, "get_embedding_endpoint",
            return_value="http://example.invalid:1235/v1/embeddings",
        )
        self.get_endpoint = self.selector.start()
        self.addCleanup(self.selector.stop)
        self.http = HTTP(Response(reply()))
        self.opener = patch.object(
            M.urllib.request, "build_opener", return_value=self.http,
        )
        self.opener.start()
        self.addCleanup(self.opener.stop)

    def test_bounded_pinned_embedding_result(self):
        vectors = M.embed_texts(["short synthetic canary"])
        self.assertEqual(len(vectors), 1)
        self.assertEqual(len(vectors[0]), 768)
        req, timeout = self.http.calls[0]
        self.assertEqual(req.get_method(), "POST")
        self.assertEqual(timeout, 30)
        data = json.loads(req.data)
        self.assertEqual(data["model"], M.MODEL)
        self.assertEqual(data["input"], ["short synthetic canary"])
        self.get_endpoint.assert_called_once_with(model=M.MODEL, fail=False)

    def test_no_post_when_selector_unavailable(self):
        self.get_endpoint.return_value = None
        with self.assertRaisesRegex(M.EmbeddingUnavailable, "no eligible"):
            M.embed_texts(["synthetic"])
        self.assertEqual(self.http.calls, [])

    def test_no_post_when_selector_exceptions(self):
        self.get_endpoint.side_effect = RuntimeError("no models")
        with self.assertRaisesRegex(M.EmbeddingUnavailable, "resolution"):
            M.embed_texts(["synthetic"])
        self.assertEqual(self.http.calls, [])

    def test_no_direct_other_model_fallback(self):
        with self.assertRaisesRegex(M.EmbeddingUnavailable, "unapproved"):
            M.embed_texts(["synthetic"], model="llama3")
        self.assertEqual(self.http.calls, [])
        self.get_endpoint.assert_not_called()

    def test_no_wrong_dimension_request(self):
        with self.assertRaisesRegex(M.EmbeddingUnavailable, "dimension"):
            M.embed_texts(["synthetic"], expected_dimensions=384)
        self.assertEqual(self.http.calls, [])

    def test_only_bounded_nonempty_batch(self):
        for bad in ([], [""] * 1, ["x"] * 17, ["x" * 8001], [" "], "text"):
            with self.subTest(bad=str(bad)[:30]):
                with self.assertRaises(M.EmbeddingUnavailable):
                    M.embed_texts(bad)
        self.assertEqual(self.http.calls, [])

    def test_batch_length_and_indices_preserved(self):
        self.http.response = Response({
            "data": [
                {"index": 1, "embedding": [0.5] * 768},
                {"index": 0, "embedding": [0.25] * 768},
            ]
        })
        result = M.embed_texts(["first", "second"])
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][0], 0.25)
        self.assertEqual(result[1][0], 0.5)

    def test_wrong_vector_length_rejected(self):
        self.http.response = Response(reply(dimensions=384))
        with self.assertRaisesRegex(M.EmbeddingUnavailable, "dimensions mismatch"):
            M.embed_texts(["synthetic"])

    def test_zero_vector_rejected(self):
        self.http.response = Response({"data": [
            {"index": 0, "embedding": [0] * 768},
        ]})
        with self.assertRaisesRegex(M.EmbeddingUnavailable, "zero vector"):
            M.embed_texts(["synthetic"])

    def test_nonfinite_nan_inf_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=str(value)):
                vector = [1.0] * 768
                vector[5] = value
                self.http.response = Response({"data": [
                    {"index": 0, "embedding": vector},
                ]})
                with self.assertRaisesRegex(M.EmbeddingUnavailable, "nonfinite"):
                    M.embed_texts(["synthetic"])

    def test_boolean_coordinate_rejected(self):
        vector = [1.0] * 768
        vector[5] = True
        self.http.response = Response({"data": [
            {"index": 0, "embedding": vector},
        ]})
        with self.assertRaisesRegex(M.EmbeddingUnavailable, "nonnumeric"):
            M.embed_texts(["synthetic"])

    def test_missing_or_duplicated_indices_rejected(self):
        for rows in (
            [{"embedding": [1.0] * 768}],
            [{"index": True, "embedding": [1.0] * 768}],
            [{"index": 2, "embedding": [1.0] * 768}],
        ):
            with self.subTest(row_keys=list(rows[0])):
                self.http.response = Response({"data": rows})
                with self.assertRaises(M.EmbeddingUnavailable):
                    M.embed_texts(["synthetic"])

    def test_wrong_batch_result_count_rejected(self):
        self.http.response = Response({"data": []})
        with self.assertRaisesRegex(M.EmbeddingUnavailable, "count mismatch"):
            M.embed_texts(["synthetic"])

    def test_malformed_json_rejected(self):
        self.http.response = Response(b"<html>not embeddings</html>")
        with self.assertRaisesRegex(M.EmbeddingUnavailable, "invalid JSON"):
            M.embed_texts(["synthetic"])

    def test_non_200_refused(self):
        self.http.response = Response(reply(), status=503)
        with self.assertRaisesRegex(M.EmbeddingUnavailable, "non-200"):
            M.embed_texts(["synthetic"])

    def test_response_budget_and_missing_body(self):
        for raw in (b"", b"a" * (M.MAX_RESPONSE_BYTES + 1)):
            with self.subTest(n=len(raw)):
                self.http.response = Response(raw)
                with self.assertRaisesRegex(M.EmbeddingUnavailable, "size invalid"):
                    M.embed_texts(["synthetic"])

    def test_timeout_limit_refused_without_network(self):
        for bad in (0, -1, 121, True, 0.5):
            with self.subTest(timeout=bad):
                with self.assertRaises(M.EmbeddingUnavailable):
                    M.embed_texts(["synthetic"], timeout=bad)
        self.assertEqual(self.http.calls, [])

    def test_redirects_are_never_authorized(self):
        self.assertIsNone(M._RejectRedirects().redirect_request(
            None, None, 302, "msg", {}, "http://untrusted.invalid",
        ))

    def test_no_raw_url_or_input_on_failing_resolver(self):
        self.get_endpoint.side_effect = RuntimeError(
            "http://private.invalid:1234/secret/path"
        )
        try:
            M.embed_texts(["VERY_PRIVATE_TEST_INPUT"])
        except M.EmbeddingUnavailable as exc:
            self.assertNotIn("VERY_PRIVATE_TEST_INPUT", str(exc))
            self.assertNotIn("private.invalid", str(exc))


class TestCallerGraphBoundary(unittest.TestCase):
    FILES = (
        "backfill_paper_768.py", "enrich_bridge_semantic.py",
        "signal_kg_bridge.py", "arxiv_kg_bridge.py",
    )

    def test_all_four_callers_route_only_through_shared_client(self):
        for filename in self.FILES:
            with self.subTest(filename=filename):
                code = (ROOT / "scripts" / filename).read_text()
                tree = ast.parse(code)
                self.assertIn("from scripts.embedding_client import", code)
                func = next(
                    node for node in tree.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == "embed"
                )
                calls = [
                    node for node in ast.walk(func)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "embed_texts"
                ]
                self.assertEqual(len(calls), 1)
                self.assertNotIn("LM_STUDIO_URL =", code)
                self.assertNotIn("EMBED_URL =", code)
                self.assertNotIn("legacy URL", code)

    def test_enrichment_does_not_exceed_client_batch_limit(self):
        code = (ROOT / "scripts" / "enrich_bridge_semantic.py").read_text()
        self.assertIn("B = MAX_BATCH", code)
        self.assertNotIn("B = 100", code)
        self.assertNotIn("B = 64", code)

    def test_k_g_failure_does_not_write_invalid_embedding(self):
        for filename in ("signal_kg_bridge.py", "arxiv_kg_bridge.py"):
            code = (ROOT / "scripts" / filename).read_text()
            self.assertIn("return None", code)
            self.assertIn("except Exception as exc:", code)
        backfill = (ROOT / "scripts" / "backfill_paper_768.py").read_text()
        self.assertIn("vec = embed(", backfill)
        self.assertIn("with drv.session(database=db) as ws:", backfill)


if __name__ == "__main__":
    unittest.main()
