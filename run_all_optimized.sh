#!/bin/bash
# DEPRECATED flat orchestrator — superseded by `bin/auto-ingest run-all`.
# Kept as a convenience shim. All config/credentials come from config.yaml + env (never hardcoded here).
# Usage: ./run_all_optimized.sh [extra args passed to `auto-ingest run-all`]
set -euo pipefail
cd "$(dirname "$0")"
exec python3 bin/auto-ingest run-all "$@"
