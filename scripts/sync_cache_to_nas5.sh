#!/usr/bin/env bash
set -euo pipefail

# Mirror deathstar's local first-cache/processing layer to NAS5 cold storage.
# Source is intentionally local so processing is not done over CIFS.

LOCAL_ROOT="${LOCAL_CACHE_ROOT:-/mnt/8TB_2025/fileserver}"
NAS5_ROOT="${NAS5_ROOT:-/media/scott/NAS5}"
DRY_RUN="${DRY_RUN:-0}"

if ! findmnt -rn "$NAS5_ROOT" >/dev/null; then
  echo "❌ NAS5 is not mounted at $NAS5_ROOT; refusing to mirror into an unmounted directory."
  exit 1
fi

if [[ ! -d "$LOCAL_ROOT" ]]; then
  echo "❌ Local cache root does not exist: $LOCAL_ROOT"
  exit 1
fi

RSYNC_FLAGS=(-rlt --ignore-existing --no-perms --no-owner --no-group --info=stats1,progress2)
if [[ "$DRY_RUN" == "1" ]]; then
  RSYNC_FLAGS=(-rlt --ignore-existing --no-perms --no-owner --no-group -ni)
fi

for subdir in audio dashcam bodycam; do
  src="$LOCAL_ROOT/$subdir"
  dst="$NAS5_ROOT/$subdir"
  if [[ ! -d "$src" ]]; then
    echo "ℹ️  Skipping missing local cache subtree: $src"
    continue
  fi
  mkdir -p "$dst"
  echo "▶️ Mirror cache: $src/ -> $dst/"
  rsync "${RSYNC_FLAGS[@]}" "$src/" "$dst/"
done

echo "✅ Local cache mirror to NAS5 complete."
