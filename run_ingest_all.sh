#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Compatibility entrypoint only. Configuration, supervision, lease heartbeats,
# retries, quarantine, and transcript process execution now live in Python.
exec python3 -m auto_ingest.transcript_orchestration "$@"
