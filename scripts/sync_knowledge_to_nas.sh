#!/usr/bin/env bash
set -euo pipefail

VAULT_PATH="${KNOWLEDGE_VAULT_PATH:-/home/scott/nas-knowledge}"
MIRROR_PATH="${KNOWLEDGE_MIRROR_PATH:-/media/scott/NAS4/fileserver/shared-knowledge}"

SOURCE_REV_FILE="$HOME/.hermes/state/knowledge_vault_nas_mirror.rev"
mkdir -p "$MIRROR_PATH" "$(dirname "$SOURCE_REV_FILE")"

CURRENT_REV="$(git -C "$VAULT_PATH" rev-parse HEAD 2>/dev/null || true)"
if [[ -n "$CURRENT_REV" && -f "$SOURCE_REV_FILE" && "$(cat "$SOURCE_REV_FILE")" == "$CURRENT_REV" ]]; then
  exit 0
fi

# Single-pass rsync: itemized output stays silent when nothing changed,
# and avoids the second tree walk that was causing cron timeouts.
DIFF_OUTPUT="$(rsync -rlt --delete --no-perms --no-owner --no-group --exclude '.git/' -i "$VAULT_PATH"/ "$MIRROR_PATH"/ || true)"
if [[ -z "$DIFF_OUTPUT" ]]; then
  [[ -n "$CURRENT_REV" ]] && printf '%s' "$CURRENT_REV" > "$SOURCE_REV_FILE"
  exit 0
fi

echo "Mirrored $VAULT_PATH -> $MIRROR_PATH"
echo "$DIFF_OUTPUT"
[[ -n "$CURRENT_REV" ]] && printf '%s' "$CURRENT_REV" > "$SOURCE_REV_FILE"
