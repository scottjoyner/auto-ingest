#!/usr/bin/env bash
set -euo pipefail

DROP_ROOT="${DROP_ROOT:-/nas/drop}"
CLAIMED_DIR="${CLAIMED_DIR:-$DROP_ROOT/claimed}"
DONE_DIR="${DONE_DIR:-$DROP_ROOT/done}"
FAILED_DIR="${FAILED_DIR:-$DROP_ROOT/failed}"
PATTERN="${PATTERN:-*.job}"

mkdir -p "$DROP_ROOT" "$CLAIMED_DIR" "$DONE_DIR" "$FAILED_DIR"

shopt -s nullglob
jobs=("$DROP_ROOT"/$PATTERN)
if (( ${#jobs[@]} == 0 )); then
  echo "No job files found in $DROP_ROOT"
  exit 0
fi

host="$(hostname)"
for job in "${jobs[@]}"; do
  base="$(basename "$job")"
  claim="$CLAIMED_DIR/${base}.${host}.$$"

  if ! mv "$job" "$claim" 2>/dev/null; then
    continue
  fi

  echo "Claimed job: $claim"

  # Secondary, queryable manifest lock (best-effort; the mv above is authoritative).
  # A down Neo4j must never block the run, so we capture the rc and ignore failures.
  KEY="${base%.job}"
  python3 "$(dirname "$0")/../scripts/claim_job.py" claim "$KEY" "$host" \
    --ttl-sec 3600 || echo "graph claim skipped (neo4j unreachable)" >&2

  if bash "$claim"; then
    mv "$claim" "$DONE_DIR/${base}.done"
    # Mark terminal stage + release so the manifest is resumable/queryable.
    python3 "$(dirname "$0")/../scripts/claim_job.py" stage "$KEY" graph_written \
      --owner "$host" || true
    python3 "$(dirname "$0")/../scripts/claim_job.py" release "$KEY" "$host" || true
    echo "Completed job: $base"
  else
    mv "$claim" "$FAILED_DIR/${base}.failed"
    # Release the graph claim on failure so another worker can retry.
    python3 "$(dirname "$0")/../scripts/claim_job.py" release "$KEY" "$host" || true
    echo "Failed job: $base" >&2
  fi
done
