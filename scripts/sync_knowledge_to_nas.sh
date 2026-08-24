#!/usr/bin/env bash
set -euo pipefail

VAULT_PATH="${KNOWLEDGE_VAULT_PATH:-/home/scott/knowledge}"
# NAS5 recovering lost files (2026-08) - mirror to NAS4 until restored.
# Revert to: /media/scott/NAS5/shared-knowledge
MIRROR_PATH="${KNOWLEDGE_MIRROR_PATH:-/media/scott/NAS4/shared-knowledge-mirror}"

mkdir -p "$MIRROR_PATH"

# First pass: detect whether anything changed. Keep silent if nothing is pending.
DIFF_OUTPUT="$(rsync -rlt --delete --no-perms --no-owner --no-group --exclude '.git/' -ni "$VAULT_PATH"/ "$MIRROR_PATH"/ || true)"
if [[ -z "$DIFF_OUTPUT" ]]; then
  exit 0
fi

# Second pass: apply changes and emit a concise summary.
rsync -rlt --delete --no-perms --no-owner --no-group --exclude '.git/' "$VAULT_PATH"/ "$MIRROR_PATH"/

echo "Mirrored $VAULT_PATH -> $MIRROR_PATH"
echo "$DIFF_OUTPUT"
