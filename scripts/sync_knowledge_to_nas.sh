#!/usr/bin/env bash
set -euo pipefail

VAULT_PATH="${KNOWLEDGE_VAULT_PATH:-/home/scott/knowledge}"
MIRROR_PATH="${KNOWLEDGE_MIRROR_PATH:-/media/scott/NAS4/shared-knowledge-mirror}"
NAS4_MOUNTPOINT="${NAS4_MOUNTPOINT:-/media/scott/NAS4}"
EXPECTED_NAS4_SOURCE="${EXPECTED_NAS4_SOURCE:-}"
EXPECTED_NAS4_FSTYPE="${EXPECTED_NAS4_FSTYPE:-cifs}"

# Fail closed before any mkdir, rsync, or destination traversal.
if [[ -z "$EXPECTED_NAS4_SOURCE" ]]; then
  printf 'ERROR: NAS4 identity gate failed: canonical expected source is not configured; refusing mirror\n' >&2
  exit 2
fi
if ! mountpoint -q "$NAS4_MOUNTPOINT"; then
  printf 'ERROR: NAS4 mount gate failed: mountpoint missing: %s (expected source=%s fstype=%s)\n' "$NAS4_MOUNTPOINT" "$EXPECTED_NAS4_SOURCE" "$EXPECTED_NAS4_FSTYPE" >&2
  exit 3
fi
read -r OBSERVED_SOURCE OBSERVED_FSTYPE OBSERVED_TARGET < <(findmnt -T "$NAS4_MOUNTPOINT" -no SOURCE,FSTYPE,TARGET)
if [[ "$OBSERVED_SOURCE" != "$EXPECTED_NAS4_SOURCE" || "$OBSERVED_FSTYPE" != "$EXPECTED_NAS4_FSTYPE" ]]; then
  printf 'ERROR: NAS4 identity gate failed: expected source=%s fstype=%s; observed source=%s fstype=%s target=%s\n' "$EXPECTED_NAS4_SOURCE" "$EXPECTED_NAS4_FSTYPE" "$OBSERVED_SOURCE" "$OBSERVED_FSTYPE" "$OBSERVED_TARGET" >&2
  exit 4
fi
if [[ ! -d "$MIRROR_PATH" ]]; then
  printf 'ERROR: NAS4 destination directory is absent after identity verification: %s\n' "$MIRROR_PATH" >&2
  exit 5
fi

EXCLUDES=(
  --exclude '.git/'
  --exclude '/backups/'
  --exclude '/photo_summaries/'
  --exclude '*.q*'
  --exclude '*.gguf'
)

DIFF_OUTPUT="$(timeout 240 rsync -rlt --delete --no-perms --no-owner --no-group "${EXCLUDES[@]}" -ni "$VAULT_PATH"/ "$MIRROR_PATH"/ )"
if [[ -z "$DIFF_OUTPUT" ]]; then
  exit 0
fi

timeout 600 rsync -rlt --delete --no-perms --no-owner --no-group "${EXCLUDES[@]}" "$VAULT_PATH"/ "$MIRROR_PATH"/
printf '%s\n' "Mirrored $VAULT_PATH -> $MIRROR_PATH"
printf '%s\n' "$DIFF_OUTPUT"
