"""A bounded, fail-closed embedding client for Auto-Ingest graph writers.

Only use a model-family + embedding-dimension pinned resolver from
scripts.get_embedding_endpoint. No fallback to hardcoded endpoints. This
function performs inference ONLY when explicitly invoked by an approved
caller; its unit tests mock all HTTP and selector calls. It is NOT a
worker-execution attestation, source signature, or admission lease.
"""
from __future__ import annotations

import json
import math
import urllib.request
from typing import Any

from scripts.get_embedding_endpoint import get_embedding_endpoint

MODEL = "text-embedding-nomic-embed-text-v1.5"
DIMENSIONS = 768
MAX_BATCH = 16
MAX_CHARS = 8000
MAX_RESPONSE_BYTES = 1_048_576


class EmbeddingUnavailable(RuntimeError):
    """Endpoint unavailable or model/shape/custody is not verifiable."""


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def _checked_vector(value: Any, expected_dimensions: int) -> list[float]:
    if not isinstance(value, list) or len(value) != expected_dimensions:
        raise EmbeddingUnavailable("embedding dimensions mismatch")
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in value):
        raise EmbeddingUnavailable("embedding contains nonfinite or nonnumeric values")
    numbers = [float(x) for x in value]
    if not any(x != 0.0 for x in numbers):
        raise EmbeddingUnavailable("embedding is a zero vector")
    return numbers


def embed_texts(
    texts: list[str], *, model: str = MODEL, timeout: int = 30,
    expected_dimensions: int = DIMENSIONS,
) -> list[list[float]]:
    """Run bounded embeddings through pinned provider selection.

    Only call from already-authorized ingestion jobs. This does NOT create
    authorization or attest the GGUF model behind a provider endpoint.
    """
    if not isinstance(texts, list) or not 1 <= len(texts) <= MAX_BATCH:
        raise EmbeddingUnavailable("invalid embedding batch")
    if not isinstance(model, str) or model != MODEL:
        raise EmbeddingUnavailable("unapproved embedding model")
    if type(expected_dimensions) is not int or expected_dimensions != DIMENSIONS:
        raise EmbeddingUnavailable("unapproved vector dimension")
    if type(timeout) is not int or not 1 <= timeout <= 120:
        raise EmbeddingUnavailable("unbounded request timeout")
    if any(
        not isinstance(t, str) or not t.strip() or len(t) > MAX_CHARS
        for t in texts
    ):
        raise EmbeddingUnavailable("blank or oversized embedding input")
    # This lookup requires model-family + dimensionality pins from an
    # installed registry; no implicit legacy URL or other-model fallback.
    try:
        url = get_embedding_endpoint(model=model, fail=False)
    except (OSError, ValueError, RuntimeError) as exc:
        raise EmbeddingUnavailable("embedding provider resolution unavailable") from exc
    if not isinstance(url, str) or not url.endswith("/v1/embeddings"):
        raise EmbeddingUnavailable("no eligible pinned embedding endpoint")

    # Deliberately do not log URL, prompt or response on failures.
    body = json.dumps(
        {"model": model, "input": texts},
        ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")
    if len(body) > MAX_BATCH * MAX_CHARS * 4 + 512:
        raise EmbeddingUnavailable("embedding request too large")
    request = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}), _RejectRedirects()
    )
    try:
        with opener.open(request, timeout=timeout) as response:
            if response.status != 200:
                raise EmbeddingUnavailable("embedding provider returned non-200")
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except (OSError, ValueError) as exc:
        raise EmbeddingUnavailable("embedding response unavailable") from exc
    if not raw or len(raw) > MAX_RESPONSE_BYTES:
        raise EmbeddingUnavailable("embedding response size invalid")
    try:
        payload = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise EmbeddingUnavailable("embedding response invalid JSON") from exc
    if not isinstance(payload, dict):
        raise EmbeddingUnavailable("embedding response schema invalid")
    values = payload.get("data")
    if not isinstance(values, list) or len(values) != len(texts):
        raise EmbeddingUnavailable("embedding result count mismatch")

    output: list[list[float] | None] = [None] * len(texts)
    used: set[int] = set()
    for row in values:
        if not isinstance(row, dict):
            raise EmbeddingUnavailable("embedding row not object")
        idx = row.get("index")
        if type(idx) is not int or not 0 <= idx < len(texts) or idx in used:
            raise EmbeddingUnavailable("embedding index missing, duplicated or invalid")
        used.add(idx)
        output[idx] = _checked_vector(row.get("embedding"), expected_dimensions)
    if len(used) != len(texts) or any(row is None for row in output):
        raise EmbeddingUnavailable("missing embedding result")
    return [row for row in output if row is not None]
