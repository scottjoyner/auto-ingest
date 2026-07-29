#!/usr/bin/env python3
"""
get_embedding_endpoint.py — fleet-aware embedding endpoint resolver.

KG embedding crons (backfill_paper_768, enrich_bridge_semantic, ...) should call this
instead of hardcoding a single LM Studio URL. It reads the shared registry
(embedding_endpoints.json on SSD), health-checks each endpoint in priority order, and
returns the first reachable one. This makes embeddings resilient: if the hub's LM Studio
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

REGISTRY = os.environ.get(
    "EMBEDDING_REGISTRY",
    "/media/scott/SSD_4TB/hermes-home/auto-ingest/embedding_endpoints.json",
)
# Env override takes precedence over the whole registry (single-endpoint mode).
ENV_URL = os.environ.get("EMBEDDING_URL")
HEALTH_TIMEOUT = float(os.environ.get("EMBEDDING_HEALTH_TIMEOUT", "4"))


def _healthy(health_url):
    try:
        req = urllib.request.Request(health_url, method="GET")
        with urllib.request.urlopen(req, timeout=HEALTH_TIMEOUT) as r:
            return r.status == 200
    except Exception:
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
        url = e.get("url")
        health = e.get("health", url.replace("/v1/embeddings", "/v1/models"))
        if url and _healthy(health):
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
