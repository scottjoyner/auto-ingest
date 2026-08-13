#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Compatibility entrypoint for the old NAS drop worker. Queue items are now
# typed *.job.json documents naming an approved orchestration profile; arbitrary
# *.job shell files are rejected and moved aside rather than executed.
exec python3 -m auto_ingest.file_queue work --once "$@"
