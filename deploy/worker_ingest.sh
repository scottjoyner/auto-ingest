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

  if bash "$claim"; then
    mv "$claim" "$DONE_DIR/${base}.done"
    echo "Completed job: $base"
  else
    mv "$claim" "$FAILED_DIR/${base}.failed"
    echo "Failed job: $base" >&2
  fi
done
