#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

# =========================
# Config
# =========================
DEST_ROOT="/mnt/8TB_2025/fileserver/dashcam"
SRC_FOLDERS=("DCIM/Movie" "DCIM/Movie/RO")

# Sidecar/associated file extensions to bring along (case-insensitive)
# Add more if you create other artifacts.
ASSOC_EXTS=("mp4" "csv" "png" "json" "srt" "txt" "wav" "mp3" "jpg" "jpeg" "webp")

# Set DRY_RUN=1 to preview without copying
DRY_RUN="${DRY_RUN:-0}"

# =========================
# Helpers
# =========================
join_by() { local IFS="$1"; shift; echo "$*"; }

copy_file() {
  local src="$1" dest_dir="$2"
  mkdir -p "$dest_dir"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "🧪 DRY-RUN: cp -np \"$src\" \"$dest_dir/\""
  else
    cp -np "$src" "$dest_dir/"
  fi
}

copy_related_for_key() {
  local src_dir="$1" key="$2" dest_dir="$3" ext_regex="$4"

  # Find any files in the same directory that begin with the key, optionally have _SUFFIX,
  # and end with an allowed extension (case-insensitive).
  # Example matches: 2025_0101_173933_F.MP4, 2025_0101_173933_FR.MP4, 2025_0101_173933_R_YOLOv8n.csv, etc.
  while IFS= read -r -d '' sidecar; do
    copy_file "$sidecar" "$dest_dir"
  done < <(
    find "$src_dir" -maxdepth 1 -type f \
      -iregex ".*/${key}(_[^/]*)?\.(${ext_regex})" -print0
  )
}

# Build case-insensitive regex for extensions
# We use -iregex so we only need lowercase here.
EXT_REGEX="$(join_by '|' "${ASSOC_EXTS[@]}")"

# =========================
# Discover mounts (skip system mounts)
# =========================
MOUNTS=$(lsblk -o MOUNTPOINT -nr | grep -Ev '^$|^/proc|^/sys|^/dev|^/run' || true)
if [[ -z "${MOUNTS// }" ]]; then
  echo "❌ No mounted drives found."
  exit 1
fi

# =========================
# Main
# =========================
for MOUNTPOINT in $MOUNTS; do
  for src_rel in "${SRC_FOLDERS[@]}"; do
    FULL_SRC="$MOUNTPOINT/$src_rel"
    [[ -d "$FULL_SRC" ]] || continue

    echo "📂 Scanning: $FULL_SRC"

    # Find MP4s (case-insensitive). Use null-delimited to handle odd filenames.
    while IFS= read -r -d '' file; do
      filename=$(basename "$file")

      # Extract: YEAR, MONTH, DAY, TIME, plus full key "YYYY_MMDD_HHMMSS"
      # Accepts names like: 2025_0101_173933_F.MP4 or 2025_0101_173933.MP4
      #                     ^^^^ ^^^^ ^^^^^^
      if [[ "$filename" =~ ^([0-9]{4})_([0-9]{2})([0-9]{2})_([0-9]{6}) ]]; then
        year="${BASH_REMATCH[1]}"
        month="${BASH_REMATCH[2]}"
        day="${BASH_REMATCH[3]}"
        key="${year}_${month}${day}_${BASH_REMATCH[4]}"

        dest_dir="$DEST_ROOT/$year/$month/$day"

        echo "➡️  Key: $key  →  $dest_dir"
        # Copy the MP4 itself
        copy_file "$file" "$dest_dir"

        # Copy any sidecars in the same directory that share the key (F/R/FR, YOLO CSVs, heatmaps, transcripts, etc.)
        copy_related_for_key "$(dirname "$file")" "$key" "$dest_dir" "$EXT_REGEX"
      else
        echo "⚠️  Skipping (unrecognized date format): $filename"
      fi
    done < <(find "$FULL_SRC" -type f -iregex '.*\.mp4' -print0)
  done
done

echo "✅ Copy complete."
