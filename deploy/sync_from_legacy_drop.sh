#!/usr/bin/env bash
set -euo pipefail

# Sync files dropped by legacy host (deathstar-xps) into canonical roots on x1-370.
# Safe for repeated runs; uses rsync --ignore-existing by default.

LEGACY_DROP_ROOT="${LEGACY_DROP_ROOT:-/nas/fileserver/incoming/deathstar}"

# Exit gracefully if legacy drop root doesn't exist
if [[ ! -d "$LEGACY_DROP_ROOT" ]]; then
    echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') Legacy drop root not found: $LEGACY_DROP_ROOT — skipping sync"
    exit 0
fi
AUDIO_ROOT="${AUDIO_ROOT:-/nas/fileserver/audio}"
DASHCAM_ROOT="${DASHCAM_ROOT:-/nas/fileserver/dashcam}"
BODYCAM_ROOT="${BODYCAM_ROOT:-/nas/fileserver/bodycam}"
TRANSCRIPT_ROOT="${TRANSCRIPT_ROOT:-/nas/fileserver/audio/transcriptions}"
YOLO_ROOT="${YOLO_ROOT:-/nas/fileserver/dashcam/yolo}"
META_ROOT="${META_ROOT:-/nas/fileserver/dashcam/metadata}"

RSYNC_FLAGS=${RSYNC_FLAGS:---archive --human-readable --ignore-existing}

mkdir -p "$AUDIO_ROOT" "$DASHCAM_ROOT" "$BODYCAM_ROOT" "$TRANSCRIPT_ROOT" "$YOLO_ROOT" "$META_ROOT"

if [[ ! -d "$LEGACY_DROP_ROOT" ]]; then
  echo "Legacy drop root not found: $LEGACY_DROP_ROOT" >&2
  exit 2
fi

sync_dir() {
  local src="$1"
  local dst="$2"
  if [[ -d "$src" ]]; then
    echo "Syncing $src -> $dst"
    rsync $RSYNC_FLAGS "$src/" "$dst/"
  fi
}

# Expected optional folder layout from legacy host drop:
# incoming/deathstar/{audio,dashcam,bodycam,transcripts,yolo,metadata}
sync_dir "$LEGACY_DROP_ROOT/audio" "$AUDIO_ROOT"
sync_dir "$LEGACY_DROP_ROOT/dashcam" "$DASHCAM_ROOT"
sync_dir "$LEGACY_DROP_ROOT/bodycam" "$BODYCAM_ROOT"
sync_dir "$LEGACY_DROP_ROOT/transcripts" "$TRANSCRIPT_ROOT"
sync_dir "$LEGACY_DROP_ROOT/yolo" "$YOLO_ROOT"
sync_dir "$LEGACY_DROP_ROOT/metadata" "$META_ROOT"

# Also copy flat sidecars from drop root by extension when present.
for ext in srt vtt txt json csv; do
  find "$LEGACY_DROP_ROOT" -maxdepth 2 -type f -name "*.${ext}" -print0 \
    | xargs -0 -I{} rsync $RSYNC_FLAGS "{}" "$TRANSCRIPT_ROOT/" || true
done

echo "Legacy drop sync complete."
