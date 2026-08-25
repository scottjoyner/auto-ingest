#!/usr/bin/env bash
set -euo pipefail

VAULT_PATH="${KNOWLEDGE_VAULT_PATH:-/home/scott/knowledge}"
# NAS5 recovering lost files (2026-08) - mirror to NAS4 until restored.
# Revert to: /media/scott/NAS5/shared-knowledge
MIRROR_PATH="${KNOWLEDGE_MIRROR_PATH:-/media/scott/NAS4/shared-knowledge-mirror}"

mkdir -p "$MIRROR_PATH"

# 2026-08-25: exclude heavy non-vault payloads. The stale 9.1G neo4j artifacts
# in knowledge/backups/ made every tick walk+diff gigabytes over CIFS.
# Excludes also protect matching receiver-side files from --delete, so nothing
# already mirrored is removed; future ticks just skip rescanning that payload.
EXCLUDES=(
  --exclude '.git/'
  --exclude '/backups/'
  --exclude '/photo_summaries/'
  --exclude '*.q*'
  --exclude '*.gguf'
)

# First pass: detect whether anything changed. Keep silent if nothing is pending.
DIFF_OUTPUT="$(timeout 240 rsync -rlt --delete --no-perms --no-owner --no-group "${EXCLUDES[@]}" -ni "$VAULT_PATH"/ "$MIRROR_PATH"/ || true)"
if [[ -z "$DIFF_OUTPUT" ]]; then
  exit 0
fi

# Second pass: apply changes and emit a concise summary.
timeout 600 rsync -rlt --delete --no-perms --no-owner --no-group "${EXCLUDES[@]}" "$VAULT_PATH"/ "$MIRROR_PATH"/

echo "Mirrored $VAULT_PATH -> $MIRROR_PATH"
echo "$DIFF_OUTPUT"
