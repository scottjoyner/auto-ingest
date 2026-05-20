#!/bin/bash
set -euo pipefail

# ======= CONFIG =======
DEST_ROOT="/mnt/8TB_2025/fileserver/audio"   # where to store organized audio
TRANS_SUBDIR="transcriptions"                # subfolder under DEST_ROOT for transcripts/diarization
SRC_FOLDER_NAME="RECORD"                     # folder on the removable drive(s)
DELETE_AFTER_COPY=false                      # true to move (delete source after copy)
LOG_PREFIX="[audio_copy]"
# ======================

# What gets copied:
# 1) Audio/video: common recorder media (*.wav, *.mp3, *.mp4, *.m4a, etc.; case-insensitive)
# 2) Audio sidecars (same folder as the audio), e.g.:
#    <basename>.*            (e.g., 20250823030047.WAV.music.json)
#    <stem>.{json,cue,md5,sha1,sha256,yaml,yml,ini,xml,info,tag}
# 3) Transcription/diarization anywhere under the source mount (by 14-digit stamp):
#    ${STAMP}_transcription.{txt,csv,json}
#    ${STAMP}_*transcription.{txt,csv,json}
#    ${STAMP}*.{rttm,srt,vtt,ass,tsv,csv,txt}
#    ${STAMP}*diariz*.{rttm,txt,csv}
#    ${STAMP}*speaker*.{rttm,txt,csv}
#
# Example:
#  - /.../20250823030047.WAV                      → DEST_ROOT/YYYY/MM/DD/
#  - /.../20250823030047.WAV.music.json           → DEST_ROOT/YYYY/MM/DD/
#  - /.../20250823050742_transcription.csv        → DEST_ROOT/transcriptions/YYYY/MM/DD/
#  - /.../20250823050742_large-v3_transcription.txt → DEST_ROOT/transcriptions/YYYY/MM/DD/

echo "$LOG_PREFIX Starting…"

shopt -s nullglob dotglob nocaseglob nocasematch

# -------- Helpers --------
copy_one() {
  local src="$1" dest_dir="$2"
  mkdir -p "$dest_dir"
  local bname; bname="$(basename "$src")"
  local dest_path="$dest_dir/$bname"
  if [ -e "$dest_path" ]; then
    echo "$LOG_PREFIX ⚠️  Exists, skipping: $dest_path"
    return 1
  fi
  echo "$LOG_PREFIX ➡️  Copying: $src → $dest_dir/"
  if rsync -t --ignore-existing -- "$src" "$dest_dir/"; then
    if [ "$DELETE_AFTER_COPY" = true ]; then
      rm -f -- "$src" || echo "$LOG_PREFIX ⚠️  Could not delete source: $src"
    fi
    return 0
  else
    echo "$LOG_PREFIX ❌ Error copying: $src"
    return 2
  fi
}

# Copy sidecars that live NEXT TO the audio file (same directory)
copy_audio_sidecars() {
  local audio_path="$1" dest_dir="$2"
  local dir bname stem
  dir="$(dirname "$audio_path")"
  bname="$(basename "$audio_path")"
  stem="${bname%.*}"

  # candidates: exact bname + extra extension(s), and common stem-based metadata
  declare -a candidates=()
  # e.g. 20250823030047.WAV.music.json
  candidates+=( "$dir/$bname".* )
  # e.g. 20250823030047.json, .cue, checksums, etc.
  candidates+=( "$dir/$stem".json "$dir/$stem".cue "$dir/$stem".md5 "$dir/$stem".sha1 "$dir/$stem".sha256 \
                "$dir/$stem".yaml "$dir/$stem".yml "$dir/$stem".ini "$dir/$stem".xml "$dir/$stem".info "$dir/$stem".tag )

  local c res
  for c in "${candidates[@]}"; do
    # Skip if it IS the audio file itself or doesn't exist
    [ -e "$c" ] || continue
    if [[ "$(basename "$c")" == "$bname" ]]; then
      continue
    fi
    # Don't accidentally grab other audio files
    if [[ "$c" =~ \.(3gp|aac|aiff|amr|avi|flac|m4a|m4v|mkv|mov|mp3|mp4|mpeg|mpg|oga|ogg|opus|wav|webm|wma)$ ]]; then
      continue
    fi
    copy_one "$c" "$dest_dir" || true
  done
}


parse_media_date() {
  local bname="$1"
  python3 - "$bname" <<'PY'
import re
import sys
from datetime import datetime
from pathlib import Path

stem = Path(sys.argv[1]).stem
patterns = [
    re.compile(r"(?<!\d)(?:[A-Za-z]{1,8})?(?P<year>(?:19|20)\d{2})[_-]?(?P<month>\d{2})(?P<day>\d{2})[_-]?(?P<hour>\d{2})(?P<minute>\d{2})(?P<second>\d{2})(?!\d)"),
    re.compile(r"(?<!\d)(?:[A-Za-z]{1,8}[_-]?)?(?P<year>(?:19|20)\d{2})[_-](?P<month>\d{2})[_-](?P<day>\d{2})[ T_-]+(?P<hour>\d{2})[:._-]?(?P<minute>\d{2})[:._-]?(?P<second>\d{2})(?!\d)"),
    re.compile(r"(?<!\d)(?:[A-Za-z]{1,8}[_-]?)?(?P<year>(?:19|20)\d{2})[_-](?P<month>\d{2})[_-](?P<day>\d{2})(?!\d)"),
]
for pattern in patterns:
    for match in pattern.finditer(stem):
        parts = match.groupdict()
        try:
            dt = datetime(
                int(parts["year"]), int(parts["month"]), int(parts["day"]),
                int(parts.get("hour") or 0), int(parts.get("minute") or 0), int(parts.get("second") or 0),
            )
        except ValueError:
            continue
        print(dt.strftime("%Y%m%d%H%M%S %Y %m %d"))
        sys.exit(0)
sys.exit(1)
PY
}

# Find & copy transcripts/diarization anywhere under SRC_BASE that match the stamp
copy_transcripts_by_stamp() {
  local src_root="$1" stamp="$2" year="$3" month="$4" day="$5"
  local tdest="$DEST_ROOT/$TRANS_SUBDIR/$year/$month/$day"
  mkdir -p "$tdest"

  # Build a space-safe list using find -print0
  # Patterns are grouped to keep find fast and specific to the stamp
  while IFS= read -r -d '' f; do
    copy_one "$f" "$tdest" || true
  done < <(
    find "$src_root" -type f \( \
      -iname "${stamp}_transcription.txt" -o \
      -iname "${stamp}_transcription.csv" -o \
      -iname "${stamp}_transcription.json" -o \
      -iname "${stamp}_*transcription.txt" -o \
      -iname "${stamp}_*transcription.csv" -o \
      -iname "${stamp}_*transcription.json" -o \
      -iname "${stamp}*.rttm" -o \
      -iname "${stamp}*diariz*.rttm" -o \
      -iname "${stamp}*diariz*.txt" -o \
      -iname "${stamp}*diariz*.csv" -o \
      -iname "${stamp}*speaker*.rttm" -o \
      -iname "${stamp}*speaker*.txt" -o \
      -iname "${stamp}*speaker*.csv" -o \
      -iname "${stamp}*.srt" -o \
      -iname "${stamp}*.vtt" -o \
      -iname "${stamp}*.ass" -o \
      -iname "${stamp}*.tsv" -o \
      -iname "${stamp}*.csv" -o \
      -iname "${stamp}*.txt" \
    \) -print0
  )
}

# Build a robust, de-duplicated list of candidate mount points.
get_mounts() {
  lsblk -pnro MOUNTPOINT,RM 2>/dev/null | awk '$1!="" && $2=="1"{print $1}'
  [ -d "/media/$USER" ] && find "/media/$USER" -mindepth 1 -maxdepth 1 -type d 2>/dev/null
  [ -d "/run/media/$USER" ] && find "/run/media/$USER" -mindepth 1 -maxdepth 1 -type d 2>/dev/null
  lsblk -pnro MOUNTPOINT 2>/dev/null | grep -Ev '^$|^/proc|^/sys|^/dev($|/)' || true
}

echo "$LOG_PREFIX Scanning mounts…"
mapfile -t CANDIDATES < <(get_mounts | sed 's:/*$::' | sort -u)

if [ "${#CANDIDATES[@]}" -eq 0 ]; then
  echo "$LOG_PREFIX ❌ No mounted drives found."
  exit 0
fi

copied=0
skipped=0
errors=0

for MOUNTPOINT in "${CANDIDATES[@]}"; do
  SRC_BASE="$MOUNTPOINT/$SRC_FOLDER_NAME"
  [[ -d "$SRC_BASE" ]] || continue
  echo "$LOG_PREFIX 📂 Found source directory: $SRC_BASE"

  # Find recorder media (case-insensitive), recursively, space-safe
  while IFS= read -r -d '' file; do
    bname="$(basename "$file")"

    # Parse a recorder timestamp from filename when present (e.g. R2025_0911_223045.mp3,
    # 20250911223045.WAV, REC_2025-09-11_22-30-45.m4a); otherwise use mtime.
    stamp=""
    year=""; month=""; day=""
    if parsed_date=$(parse_media_date "$bname" 2>/dev/null); then
      read -r stamp year month day <<< "$parsed_date"
    else
      mod_date="$(stat -c %y "$file" 2>/dev/null | cut -d' ' -f1 || true)"
      if [[ "$mod_date" =~ ^([0-9]{4})-([0-9]{2})-([0-9]{2})$ ]]; then
        year="${BASH_REMATCH[1]}"; month="${BASH_REMATCH[2]}"; day="${BASH_REMATCH[3]}"
      else
        year="unknown"; month="00"; day="00"
      fi
    fi

    dest_dir="$DEST_ROOT/$year/$month/$day"
    if copy_one "$file" "$dest_dir"; then
      ((copied++)) || true
    else
      # copy_one returns 1 when skipped, 2 when error
      last=$?
      if [ $last -eq 1 ]; then ((skipped++)) || true; else ((errors++)) || true; fi
      # Even if audio existed, still try to bring over sidecars & transcripts
    fi

    # Copy sidecars living next to the audio
    copy_audio_sidecars "$file" "$dest_dir" || true

    # Copy transcripts/diarization matching the 14-digit stamp (if we have one)
    if [ -n "$stamp" ]; then
      copy_transcripts_by_stamp "$SRC_BASE" "$stamp" "$year" "$month" "$day" || true
    fi

  done < <(find "$SRC_BASE" -type f \( -iname "*.3gp" -o -iname "*.aac" -o -iname "*.aiff" -o -iname "*.amr" -o -iname "*.avi" -o -iname "*.flac" -o -iname "*.m4a" -o -iname "*.m4v" -o -iname "*.mkv" -o -iname "*.mov" -o -iname "*.mp3" -o -iname "*.mp4" -o -iname "*.mpeg" -o -iname "*.mpg" -o -iname "*.oga" -o -iname "*.ogg" -o -iname "*.opus" -o -iname "*.wav" -o -iname "*.webm" -o -iname "*.wma" \) -print0)

done

echo "$LOG_PREFIX ✅ Done. Copied: $copied, Skipped: $skipped, Errors: $errors"
