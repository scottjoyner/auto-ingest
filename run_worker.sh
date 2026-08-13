#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Compatibility entrypoint. The worker lifecycle now lives in Python so cron,
# services, and interactive workers share the same fenced orchestration,
# heartbeat, retry/quarantine, and resource-admission contracts.
exec python3 -m auto_ingest.worker_loop "$@"
