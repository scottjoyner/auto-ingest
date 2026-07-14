#!/usr/bin/env bash
set -euo pipefail

# Sync files dropped by legacy host (deathstar-xps) into canonical roots on x1-370.
# Safe for repeated runs; uses rsync --ignore-existing by default.

LEGACY_DROP_ROOT="${LEGACY_DROP_ROOT:-/nas/fileserver/incoming/deathstar}"
REMOTE_DROP_HOST="${REMOTE_DROP_HOST:-deathstar}"
REMOTE_DROP_ROOT="${REMOTE_DROP_ROOT:-/mnt/8TB_2025/fileserver}"
REMOTE_PULL="${REMOTE_PULL:-1}"
LOCAL_FILESERVER_ROOT="${LOCAL_FILESERVER_ROOT:-/nas/fileserver}"

# If the local legacy drop root is missing, keep going when remote pull is enabled.
if [[ ! -d "$LEGACY_DROP_ROOT" ]]; then
  if [[ "$REMOTE_PULL" == "1" ]]; then
    echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') Local legacy drop root not found: $LEGACY_DROP_ROOT — continuing with remote source"
  else
    echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') Legacy drop root not found: $LEGACY_DROP_ROOT — skipping sync"
    exit 0
  fi
fi

AUDIO_ROOT="${AUDIO_ROOT:-$LOCAL_FILESERVER_ROOT/audio}"
DASHCAM_ROOT="${DASHCAM_ROOT:-$LOCAL_FILESERVER_ROOT/dashcam}"
BODYCAM_ROOT="${BODYCAM_ROOT:-$LOCAL_FILESERVER_ROOT/bodycam}"
HEADCAM_ROOT="${HEADCAM_ROOT:-$LOCAL_FILESERVER_ROOT/headcam}"
NEO4J_BKPS_ROOT="${NEO4J_BKPS_ROOT:-$LOCAL_FILESERVER_ROOT/neo4j-bkps}"
TRANSCRIPT_ROOT="${TRANSCRIPT_ROOT:-$AUDIO_ROOT/transcriptions}"
YOLO_ROOT="${YOLO_ROOT:-$DASHCAM_ROOT/yolo}"
META_ROOT="${META_ROOT:-$DASHCAM_ROOT/metadata}"

# exFAT on NAS4 cannot preserve unix owner/group/perms; make those metadata losses explicit
# instead of letting rsync exit 23 and crash-loop the container.
RSYNC_FLAGS=${RSYNC_FLAGS:---archive --human-readable --ignore-existing --no-perms --no-owner --no-group --omit-dir-times}

mkdir -p "$LOCAL_FILESERVER_ROOT" "$AUDIO_ROOT" "$DASHCAM_ROOT" "$BODYCAM_ROOT" "$HEADCAM_ROOT" "$NEO4J_BKPS_ROOT" "$TRANSCRIPT_ROOT" "$YOLO_ROOT" "$META_ROOT"

if [[ ! -d "$LEGACY_DROP_ROOT" && "$REMOTE_PULL" != "1" ]]; then
  echo "Legacy drop root not found: $LEGACY_DROP_ROOT" >&2
  exit 2
fi

sync_dir() {
  local src="$1"
  local dst="$2"
  if [[ -d "$src" ]]; then
    echo "Syncing $src -> $dst"
    local rc=0
    rsync $RSYNC_FLAGS "$src/" "$dst/" || rc=$?
    if [[ "$rc" -ne 0 ]]; then
      if [[ "$rc" -eq 23 || "$rc" -eq 24 ]]; then
        echo "Warning: rsync returned $rc for $src -> $dst; continuing because NAS4 may not preserve all metadata"
      else
        return "$rc"
      fi
    fi
  fi
}

sync_remote_dir() {
  local src="$1"
  local dst="$2"
  local remote_host="${src%%:*}"
  local remote_path="${src#*:}"
  if ssh -n "$remote_host" "test -e '$remote_path'" >/dev/null 2>&1; then
    echo "Syncing remote $src -> $dst"
    local rc=0
    rsync $RSYNC_FLAGS -e ssh "$src/" "$dst/" || rc=$?
    if [[ "$rc" -ne 0 ]]; then
      if [[ "$rc" -eq 23 || "$rc" -eq 24 ]]; then
        echo "Warning: rsync returned $rc for $src -> $dst; continuing because NAS4 may not preserve all metadata"
      else
        return "$rc"
      fi
    fi
  else
    echo "Skipping missing remote source: $src"
  fi
}

sync_remote_top_level() {
  local saw_any=0
  while IFS= read -r name; do
    saw_any=1
    [[ -z "$name" ]] && continue
    case "$name" in
      incoming)
        continue
        ;;
    esac
    sync_remote_dir "$REMOTE_DROP_HOST:$REMOTE_DROP_ROOT/$name" "$LOCAL_FILESERVER_ROOT/$name"
  done < <(ssh -n "$REMOTE_DROP_HOST" "cd '$REMOTE_DROP_ROOT' && find . -mindepth 1 -maxdepth 1 -type d -printf '%f\\n' | sort" 2>/dev/null || true)

  if [[ "$saw_any" -eq 0 ]]; then
    echo "No remote top-level directories found under ${REMOTE_DROP_HOST}:${REMOTE_DROP_ROOT}"
  fi
}

# Legacy local drop layout: mirror known folders into the canonical fileserver tree.
sync_dir "$LEGACY_DROP_ROOT/audio" "$AUDIO_ROOT"
sync_dir "$LEGACY_DROP_ROOT/dashcam" "$DASHCAM_ROOT"
sync_dir "$LEGACY_DROP_ROOT/bodycam" "$BODYCAM_ROOT"
sync_dir "$LEGACY_DROP_ROOT/headcam" "$HEADCAM_ROOT"
sync_dir "$LEGACY_DROP_ROOT/neo4j-bkps" "$NEO4J_BKPS_ROOT"
sync_dir "$LEGACY_DROP_ROOT/transcripts" "$TRANSCRIPT_ROOT"

# Older sidecar layouts sometimes stored yolo/metadata at the drop root or nested
# directly under dashcam. The canonical base for both is the dashcam tree.
sync_dir "$LEGACY_DROP_ROOT/yolo" "$YOLO_ROOT"
sync_dir "$LEGACY_DROP_ROOT/metadata" "$META_ROOT"
sync_dir "$LEGACY_DROP_ROOT/dashcam/yolo" "$YOLO_ROOT"
sync_dir "$LEGACY_DROP_ROOT/dashcam/metadata" "$META_ROOT"

# Secondary source: deathstar 8TB fileserver mounted locally on deathstar.
# This pulls every top-level directory found on the drive, so newly discovered
# roots are included automatically.
if [[ "$REMOTE_PULL" == "1" ]]; then
  if ssh "$REMOTE_DROP_HOST" "test -d '$REMOTE_DROP_ROOT'" >/dev/null 2>&1; then
    sync_remote_top_level
  else
    echo "Remote drop root not reachable: ${REMOTE_DROP_HOST}:${REMOTE_DROP_ROOT}"
  fi
fi

# Also copy flat sidecars from drop root by extension when present.
if [[ -d "$LEGACY_DROP_ROOT" ]]; then
  for ext in srt vtt txt json csv; do
    find "$LEGACY_DROP_ROOT" -maxdepth 2 -type f -name "*.${ext}" -print0 \
      | xargs -0 -I{} rsync $RSYNC_FLAGS "{}" "$TRANSCRIPT_ROOT/" || true
  done
fi

echo "Legacy drop sync complete."
