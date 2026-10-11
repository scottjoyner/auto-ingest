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


def _healthy(health_url):
    """Read-only model LISTING acceptance; not proof of LOADED or owner auth.

    A 200 from /v1/models with zero advertised models is not a usable
    embedding endpoint. The caller's separate production trust policy must
    still enforce model identity, source authentication, and owner admission.
    """
    if not isinstance(health_url, str) or not health_url.endswith("/v1/models"):
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
        return any(
            isinstance(item, dict) and isinstance(item.get("id"), str)
            and bool(item["id"].strip())
            for item in inventory["data"]
        )
    except (OSError, ValueError, TypeError, UnicodeError):
        return False


def get_embedding_endpoint(fail=True, model=None):
    """Return the embeddings URL of the first healthy endpoint, or None/raise."""
    if ENV_URL:
        # Single-endpoint mode: still health-check it.
        if _healthy(ENV_URL.replace("/v1/embeddings", "/v1/models")):
            return ENV_URL
        if fail:
            raise RuntimeError(f"EMBEDDING_URL {ENV_URL} is not reachable")
        return None

    try:
        with open(REGISTRY) as f:
            reg = json.load(f)
    except FileNotFoundError:
        if fail:
            raise RuntimeError(f"Embedding registry not found: {REGISTRY}")
        return None

    endpoints = reg.get("endpoints", [])
    if model:
        endpoints = [e for e in endpoints if reg.get("model") == model or e.get("model") == model]
    endpoints.sort(key=lambda e: e.get("priority", 99))

    for e in endpoints:
        if not isinstance(e, dict):
            continue
        url = e.get("url")
        if not isinstance(url, str) or not url.endswith("/v1/embeddings"):
            continue
        health = e.get("health") or url.replace("/v1/embeddings", "/v1/models")
        if _healthy(health):
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
