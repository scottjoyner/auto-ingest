#!/usr/bin/env python3
"""
get_embedding_endpoint.py — fleet-aware embedding endpoint resolver.

KG embedding crons (backfill_paper_768, enrich_bridge_semantic, ...) should call this
instead of hardcoding a single LM Studio URL. It reads the shared registry
(embedding_endpoints.json on SSD), health-checks each endpoint in priority order, and
returns the first provider listing at least one model. HTTP 200 alone is not
model readiness or worker execution proof. This makes embeddings resilient: if the hub's LM Studio
is down (e.g. benchmarking), the cron fails over to a standby node (destroyer, etc).

Usage (in a cron script):
    from get_embedding_endpoint import get_embedding_endpoint
    url = get_embedding_endpoint()          # raises RuntimeError if all down
    url = get_embedding_endpoint(fail=False) # returns None if all down

Or as a CLI: prints the chosen URL (or nothing) for shell scripts.
"""
import json
import os
import sys
import urllib.request

MAX_INVENTORY_BYTES = 65_536

REGISTRY = os.environ.get(
    "EMBEDDING_REGISTRY",
    "/media/scott/SSD_4TB/hermes-home/auto-ingest/embedding_endpoints.json",
)
# Env override takes precedence over the whole registry (single-endpoint mode).
ENV_URL = os.environ.get("EMBEDDING_URL")
HEALTH_TIMEOUT = float(os.environ.get("EMBEDDING_HEALTH_TIMEOUT", "4"))


def _matches_expected_model(model_id, expected_model):
    """Match canonical model family, not arbitrary substring in a path.

    Registry: text-embedding-nomic-embed-text-v1.5
    llama.cpp: .../nomic-embed-text-v1.5.Q4_K_M.gguf
    We accept only exact family or a separated quantization/file suffix.
    """
    if not isinstance(model_id, str) or not model_id.strip():
        return False
    if expected_model is None:
        return True
    if not isinstance(expected_model, str) or not expected_model.strip():
        return False
    expected = expected_model.strip().lower()
    if expected.startswith("text-embedding-"):
        expected = expected[len("text-embedding-"):]
    expected = expected.replace("_", "-")
    advertised = model_id.rsplit("/", 1)[-1].strip().lower().replace("_", "-")
    return (
        advertised == expected
        or advertised.startswith(expected + ".")
        or any(
            advertised.startswith(expected + "-" + quant)
            for quant in ("q2", "q3", "q4", "q5", "q6", "q8", "f16", "f32")
        )
    )


def _healthy(health_url, expected_model=None, expected_dimensions=None):
    """Read-only model LISTING acceptance; not proof of LOADED or owner auth.

    A 200 from /v1/models with zero advertised models is not a usable
    embedding endpoint. The caller's separate production trust policy must
    still enforce model identity, source authentication, and owner admission.
    """
    if not isinstance(health_url, str) or not health_url.endswith("/v1/models"):
        return False
    if expected_dimensions is not None and (
        type(expected_dimensions) is not int or expected_dimensions <= 0
    ):
        return False
    try:
        req = urllib.request.Request(
            health_url, method="GET", headers={"Accept": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=HEALTH_TIMEOUT) as response:
            if response.status != 200:
                return False
            raw = response.read(MAX_INVENTORY_BYTES + 1)
        if not raw or len(raw) > MAX_INVENTORY_BYTES:
            return False
        inventory = json.loads(raw)
        if not isinstance(inventory, dict) or not isinstance(
            inventory.get("data"), list
        ):
            return False
        for item in inventory["data"]:
            if not isinstance(item, dict) or not _matches_expected_model(
                item.get("id"), expected_model
            ):
                continue
            if expected_dimensions is not None:
                metadata = item.get("meta")
                if not isinstance(metadata, dict) or (
                    type(metadata.get("n_embd")) is not int
                    or metadata["n_embd"] != expected_dimensions
                ):
                    continue
            return True
        return False
    except (OSError, ValueError, TypeError, UnicodeError):
        return False


def get_embedding_endpoint(fail=True, model=None):
    """Return the embeddings URL of the first healthy endpoint, or None/raise."""
    if ENV_URL:
        # A manual endpoint override must not bypass the *installed* model
        # family and embedding dimension pin. Never include URL in errors.
        try:
            with open(REGISTRY) as config_file:
                override_registry = json.load(config_file)
        except (OSError, ValueError, UnicodeError):
            override_registry = {}
        if not isinstance(override_registry, dict):
            override_registry = {}
        pinned_model = model or override_registry.get("model")
        pinned_dimensions = override_registry.get("embedding_dimensions")
        if (
            pinned_model and type(pinned_dimensions) is int
            and pinned_dimensions > 0
            and isinstance(ENV_URL, str)
            and ENV_URL.endswith("/v1/embeddings")
            and _healthy(
                ENV_URL.replace("/v1/embeddings", "/v1/models"),
                expected_model=pinned_model,
                expected_dimensions=pinned_dimensions,
            )
        ):
            return ENV_URL
        if fail:
            raise RuntimeError("Embedding override unavailable or not model-pinned")
        return None

    try:
        with open(REGISTRY) as f:
            reg = json.load(f)
    except FileNotFoundError:
        if fail:
            raise RuntimeError(f"Embedding registry not found: {REGISTRY}")
        return None

    endpoints = reg.get("endpoints", [])
    if not isinstance(endpoints, list):
        endpoints = []
    endpoints = [e for e in endpoints if isinstance(e, dict)]
    if model:
        endpoints = [
            e for e in endpoints
            if reg.get("model") == model or e.get("model") == model
        ]

    def priority(entry):
        number = entry.get("priority", 99)
        return number if type(number) is int and number >= 0 else 99

    endpoints.sort(key=priority)

    for e in endpoints:
        if not isinstance(e, dict):
            continue
        url = e.get("url")
        if not isinstance(url, str) or not url.endswith("/v1/embeddings"):
            continue
        health = e.get("health") or url.replace("/v1/embeddings", "/v1/models")
        expected_model = model or e.get("model") or reg.get("model")
        expected_dimensions = e.get(
            "embedding_dimensions", reg.get("embedding_dimensions")
        )
        if _healthy(
            health, expected_model=expected_model,
            expected_dimensions=expected_dimensions,
        ):
            return url

    if fail:
        raise RuntimeError("No healthy embedding endpoint in registry")
    return None


if __name__ == "__main__":
    url = get_embedding_endpoint(fail=False)
    if url:
        print(url)
        sys.exit(0)
    sys.exit(1)
