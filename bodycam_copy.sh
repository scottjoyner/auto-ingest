#!/bin/bash
set -euo pipefail

# ======= CONFIG =======
DEST_BODYCAM_ROOT="/mnt/8TB_2025/fileserver/bodycam"   # where video/photo go (dated)
DEST_AUDIO_ROOT="/mnt/8TB_2025/fileserver/audio"       # where audio (and extracted audio) go (dated)
TRANS_SUBDIR="transcriptions"                          # under DEST_AUDIO_ROOT for transcripts/diarization

# Source folder names (case-insensitive) expected at device root
SRC_DIRS=("RECORD" "VIDEO" "PHOTO")

# Delete source files after successful copy & (if enabled) successful audio extraction
DELETE_AFTER_COPY=false

# Extract an .mp3 for every video (.mp4) and .m4a found in VIDEO
EXTRACT_AUDIO_FROM_VIDEO=true
AUDIO_MP3_QUALITY="-q:a 2"   # VBR quality (~190 kbps). Alternative: "-b:a 192k"
LOG_PREFIX="[media_copy]"
# ======================

echo "$LOG_PREFIX Starting…"

shopt -s nullglob dotglob nocaseglob

# --- Helpers ---

get_mounts() {
  # Prefer removable devices
  lsblk -pnro MOUNTPOINT,RM 2>/dev/null | awk '$1!="" && $2=="1"{print $1}'
  # Common user media roots
  [ -d "/media/$USER" ] && find "/media/$USER" -mindepth 1 -maxdepth 1 -type d 2>/dev/null
  [ -d "/run/media/$USER" ] && find "/run/media/$USER" -mindepth 1 -maxdepth 1 -type d 2>/dev/null
  # Fallback: all non-system mounts (keep /run so /run/media works)
  lsblk -pnro MOUNTPOINT 2>/dev/null | grep -Ev '^$|^/proc|^/sys|^/dev($|/)'
}

# Extract Y M D from filename if it starts with 14 digits (YYYYMMDDhhmmss[...]),
# else fallback to file mtime (YYYY-MM-DD). Echoes: "YYYY MM DD"
date_parts_for() {
  local path="$1"
  local bname="$(basename "$path")"

  # Match: YYYYMMDDhhmmss.ext OR YYYYMMDDhhmmss_######.ext (case-insensitive ext)
  if [[ "$bname" =~ ^([0-9]{4})([0-9]{2})([0-9]{2})([0-9]{6})(_[0-9]{6})?\.[^.]+$ ]]; then
    echo "${BASH_REMATCH[1]} ${BASH_REMATCH[2]} ${BASH_REMATCH[3]}"
    return 0
  fi

  # Fallback to file mtime
  local mod_date
  mod_date="$(stat -c %y "$path" 2>/dev/null | cut -d' ' -f1 || true)"
  if [[ "$mod_date" =~ ^([0-9]{4})-([0-9]{2})-([0-9]{2})$ ]]; then
    echo "${BASH_REMATCH[1]} ${BASH_REMATCH[2]} ${BASH_REMATCH[3]}"
  else
    echo "unknown 00 00"
  fi
}

# Return 14-digit stamp (YYYYMMDDhhmmss) if filename starts with it; else empty
stamp_for() {
  local path="$1"
  local bname="$(basename "$path")"
  if [[ "$bname" =~ ^([0-9]{14})(_[0-9]{6})?\.[^.]+$ ]]; then
    echo "${BASH_REMATCH[1]}"
  else
    echo ""
  fi
}

ensure_dir() {
  mkdir -p "$1"
}

copy_file() {
  local src="$1" dest_dir="$2" label="$3"
  ensure_dir "$dest_dir"
  local bname dest_path
  bname="$(basename "$src")"
  dest_path="$dest_dir/$bname"

  if [ -e "$dest_path" ]; then
    echo "$LOG_PREFIX ⚠️  Exists ($label), skipping: $dest_path"
    return 2
  fi

  echo "$LOG_PREFIX ➡️  Copying ($label): $bname → $dest_dir"
  if rsync -t --ignore-existing --progress -- "$src" "$dest_dir/" >/dev/null 2>&1; then
    return 0
  else
    echo "$LOG_PREFIX ❌ Error copying ($label): $src"
    return 1
  fi
}

maybe_delete() {
  local src="$1"
  if [ "$DELETE_AFTER_COPY" = true ]; then
    rm -f -- "$src" || echo "$LOG_PREFIX ⚠️  Could not delete source: $src"
  fi
}

extract_mp3() {
  local input="$1" y="$2" m="$3" d="$4"
  local base="${input##*/}"
  local stem="${base%.*}"
  local out_dir="$DEST_AUDIO_ROOT/$y/$m/$d"
  local out_path="$out_dir/$stem.mp3"

  ensure_dir "$out_dir"
  if [ -e "$out_path" ]; then
    echo "$LOG_PREFIX 🎧 MP3 exists, skipping: $out_path"
    return 0
  fi

  if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "$LOG_PREFIX ❌ ffmpeg not found; cannot extract audio."
    return 1
  fi

  echo "$LOG_PREFIX 🎧 Extracting MP3: $base → $out_path"
  if ffmpeg -hide_banner -loglevel error -nostdin -i "$input" -vn -acodec libmp3lame $AUDIO_MP3_QUALITY "$out_path"; then
    return 0
  else
    echo "$LOG_PREFIX ❌ ffmpeg failed on: $input"
    return 1
  fi
}

# ---- Sidecars & metadata ----

# Copy sidecars sitting NEXT TO a media file, avoiding re-copy of media itself
# mode = AUDIO|VIDEO|PHOTO (affects which patterns we consider)
copy_sidecars_next_to() {
  local media="$1" dest_dir="$2" mode="$3"
  local dir bname stem
  dir="$(dirname "$media")"
  bname="$(basename "$media")"
  stem="${bname%.*}"

  # Assemble candidate patterns
  declare -a candidates=()
  case "$mode" in
    AUDIO)
      # e.g. 20250823030047.WAV.music.json, and stem-based metadata
      candidates+=( "$dir/$bname".* )
      candidates+=( "$dir/$stem".json "$dir/$stem".yaml "$dir/$stem".yml "$dir/$stem".md5 "$dir/$stem".sha1 "$dir/$stem".sha256 "$dir/$stem".cue "$dir/$stem".xml "$dir/$stem".info "$dir/$stem".tag )
      ;;
    VIDEO)
      # common bodycam outputs placed alongside video
      candidates+=( "$dir/${stem}"*_yolo*.csv "$dir/${stem}"*_yolov*.csv "$dir/${stem}"*YOLO*.csv )
      candidates+=( "$dir/${stem}"*_heatmap.* "$dir/${stem}"*_bbox*.csv "$dir/${stem}"*_detections*.csv "$dir/${stem}"*_tracks*.csv )
      candidates+=( "$dir/${stem}"*.json "$dir/${stem}"*.jsonl "$dir/${stem}"*.yaml "$dir/${stem}"*.yml )
      candidates+=( "$dir/${stem}"*.srt "$dir/${stem}"*.vtt "$dir/${stem}"*.ass )   # captions next to video
      ;;
    PHOTO)
      candidates+=( "$dir/${stem}".xmp "$dir/${stem}".json "$dir/${stem}".xml "$dir/${stem}"*.txt )
      ;;
  esac

  local c rc
  for c in "${candidates[@]}"; do
    [ -e "$c" ] || continue
    # Skip if it is the media file itself or another primary media
    if [[ "$(basename "$c")" == "$bname" ]]; then continue; fi
    if [[ "$c" =~ \.(wav|mp3|mp4|m4a|jpg|jpeg|png|heic)$ ]]; then continue; fi

    if copy_file "$c" "$dest_dir" "${mode}_SIDECAR"; then
      ((meta_copied++)) || true
      maybe_delete "$c"
    else
      rc=$?
      if [ $rc -eq 2 ]; then ((skipped++)) || true; else ((errors++)) || true; fi
    fi
  done
}

# Copy transcripts/diarization by 14-digit STAMP anywhere under SRC_ROOT → AUDIO/transcriptions/YYYY/MM/DD
copy_transcripts_by_stamp() {
  local src_root="$1" stamp="$2" y="$3" m="$4" d="$5"
  local tdest="$DEST_AUDIO_ROOT/$TRANS_SUBDIR/$y/$m/$d"
  ensure_dir "$tdest"

  while IFS= read -r -d '' f; do
    if copy_file "$f" "$tdest" "TRANSCRIPT"; then
      ((transcripts_copied++)) || true
      maybe_delete "$f"
    else
      rc=$?
      if [ $rc -eq 2 ]; then ((skipped++)) || true; else ((errors++)) || true; fi
    fi
  done < <(
    find "$src_root" -type f \( \
      -iname "${stamp}_transcription.txt" -o \
      -iname "${stamp}_transcription.csv" -o \
      -iname "${stamp}_transcription.json" -o \
      -iname "${stamp}_*transcription.txt" -o \
      -iname "${stamp}_*transcription.csv" -o \
      -iname "${stamp}_*transcription.json" -o \
      -iname "${stamp}*diariz*.rttm" -o \
      -iname "${stamp}*diariz*.txt" -o \
      -iname "${stamp}*diariz*.csv" -o \
      -iname "${stamp}*speaker*.rttm" -o \
      -iname "${stamp}*speaker*.txt" -o \
      -iname "${stamp}*speaker*.csv" -o \
      -iname "${stamp}*.rttm" -o \
      -iname "${stamp}*.tsv" -o \
      -iname "${stamp}*.csv" -o \
      -iname "${stamp}*.txt" \
    \) -print0
  )
}

# Copy captions/subtitles by STAMP (SRT/VTT/ASS) → BODYCAM/YYYY/MM/DD/captions
copy_captions_by_stamp() {
  local src_root="$1" stamp="$2" y="$3" m="$4" d="$5"
  local cdest="$DEST_BODYCAM_ROOT/$y/$m/$d/captions"
  ensure_dir "$cdest"

  while IFS= read -r -d '' f; do
    if copy_file "$f" "$cdest" "CAPTION"; then
      ((captions_copied++)) || true
      maybe_delete "$f"
    else
      rc=$?
      if [ $rc -eq 2 ]; then ((skipped++)) || true; else ((errors++)) || true; fi
    fi
  done < <(
    find "$src_root" -type f \( \
      -iname "${stamp}*.srt" -o \
      -iname "${stamp}*.vtt" -o \
      -iname "${stamp}*.ass" \
    \) -print0
  )
}

# Search for YOLO/heatmap/etc. by STAMP anywhere under SRC_ROOT → BODYCAM/YYYY/MM/DD/video
copy_video_analytics_by_stamp() {
  local src_root="$1" stamp="$2" y="$3" m="$4" d="$5"
  local vdest="$DEST_BODYCAM_ROOT/$y/$m/$d/video"
  ensure_dir "$vdest"

  while IFS= read -r -d '' f; do
    if copy_file "$f" "$vdest" "VIDEO_ANALYTICS"; then
      ((video_meta_copied++)) || true
      maybe_delete "$f"
    else
      rc=$?
      if [ $rc -eq 2 ]; then ((skipped++)) || true; else ((errors++)) || true; fi
    fi
  done < <(
    find "$src_root" -type f \( \
      -iname "${stamp}_yolo*.csv" -o \
      -iname "${stamp}_yolov*.csv" -o \
      -iname "${stamp}*YOLO*.csv" -o \
      -iname "${stamp}*heatmap*.png" -o \
      -iname "${stamp}*bbox*.csv" -o \
      -iname "${stamp}*detections*.csv" -o \
      -iname "${stamp}*tracks*.csv" -o \
      -iname "${stamp}*.json" -o \
      -iname "${stamp}*.jsonl" -o \
      -iname "${stamp}*.yaml" -o \
      -iname "${stamp}*.yml" \
    \) -print0
  )
}

# --- Discover mounts ---
mapfile -t CANDIDATES < <(get_mounts | sed 's:/*$::' | sort -u)
if [ "${#CANDIDATES[@]}" -eq 0 ]; then
  echo "$LOG_PREFIX ❌ No mounted drives found."
  exit 0
fi

video_copied=0
photo_copied=0
audio_copied=0
audio_extracted=0
meta_copied=0
video_meta_copied=0
captions_copied=0
transcripts_copied=0
skipped=0
errors=0

for MOUNTPOINT in "${CANDIDATES[@]}"; do
  # Look for RECORD/VIDEO/PHOTO at mount root (case-insensitive)
  while IFS= read -r -d '' SRC_BASE; do
    base_name="$(basename "$SRC_BASE")"
    case "${base_name^^}" in
      "RECORD")
        LABEL="AUDIO_RAW"
        exts=(-iname "*.wav" -o -iname "*.mp3")
        while IFS= read -r -d '' file; do
          read -r year month day < <(date_parts_for "$file")
          dest_dir="$DEST_AUDIO_ROOT/$year/$month/$day"

          if copy_file "$file" "$dest_dir" "$LABEL"; then
            ((audio_copied++)) || true

            # Copy audio sidecars next to the audio in destination
            copy_sidecars_next_to "$file" "$dest_dir" "AUDIO"

            # Transcripts/diarization/captions by STAMP
            stamp="$(stamp_for "$file")"
            if [ -n "$stamp" ]; then
              copy_transcripts_by_stamp "$SRC_BASE" "$stamp" "$year" "$month" "$day"
              copy_captions_by_stamp "$SRC_BASE" "$stamp" "$year" "$month" "$day"
            fi

            maybe_delete "$file"
          else
            rc=$?
            if [ $rc -eq 2 ]; then ((skipped++)) || true; else ((errors++)) || true; fi
          fi
        done < <(find "$SRC_BASE" -type f \( "${exts[@]}" \) -print0)
        ;;
      "VIDEO")
        LABEL="VIDEO"
        exts=(-iname "*.mp4" -o -iname "*.m4a")
        while IFS= read -r -d '' file; do
          read -r year month day < <(date_parts_for "$file")
          dest_dir="$DEST_BODYCAM_ROOT/$year/$month/$day/video"

          if copy_file "$file" "$dest_dir" "$LABEL"; then
            ((video_copied++)) || true

            # Sidecars that live next to the video (e.g., YOLO CSVs, heatmaps, JSON)
            copy_sidecars_next_to "$file" "$dest_dir" "VIDEO"

            # Analytics & captions discovered anywhere in SRC_BASE by STAMP
            stamp="$(stamp_for "$file")"
            if [ -n "$stamp" ]; then
              copy_video_analytics_by_stamp "$SRC_BASE" "$stamp" "$year" "$month" "$day"
              copy_captions_by_stamp "$SRC_BASE" "$stamp" "$year" "$month" "$day"
            fi

            # Extract audio (MP3) if enabled
            extraction_ok=true
            if [ "$EXTRACT_AUDIO_FROM_VIDEO" = true ]; then
              if extract_mp3 "$file" "$year" "$month" "$day"; then
                ((audio_extracted++)) || true
                # Also pull transcripts/diarization for the same STAMP into audio/transcriptions
                if [ -n "$stamp" ]; then
                  copy_transcripts_by_stamp "$SRC_BASE" "$stamp" "$year" "$month" "$day"
                fi
              else
                extraction_ok=false
                ((errors++)) || true
              fi
            fi

            # Only delete source video if copy (and extraction if enabled) succeeded
            if [ "$DELETE_AFTER_COPY" = true ] && [ "$extraction_ok" = true ]; then
              rm -f -- "$file" || echo "$LOG_PREFIX ⚠️  Could not delete source: $file"
            fi
          else
            rc=$?
            if [ $rc -eq 2 ]; then ((skipped++)) || true; else ((errors++)) || true; fi
          fi
        done < <(find "$SRC_BASE" -type f \( "${exts[@]}" \) -print0)
        ;;
      "PHOTO")
        LABEL="PHOTO"
        exts=(-iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.heic")
        while IFS= read -r -d '' file; do
          read -r year month day < <(date_parts_for "$file")
          dest_dir="$DEST_BODYCAM_ROOT/$year/$month/$day/photo"
          if copy_file "$file" "$dest_dir" "$LABEL"; then
            ((photo_copied++)) || true
            copy_sidecars_next_to "$file" "$dest_dir" "PHOTO"
            maybe_delete "$file"
          else
            rc=$?
            if [ $rc -eq 2 ]; then ((skipped++)) || true; else ((errors++)) || true; fi
          fi
        done < <(find "$SRC_BASE" -type f \( "${exts[@]}" \) -print0)
        ;;
      *)
        continue
        ;;
    esac
  done < <(find "$MOUNTPOINT" -maxdepth 1 -type d \( -iname "RECORD" -o -iname "VIDEO" -o -iname "PHOTO" \) -print0)
done

echo "$LOG_PREFIX ✅ Done."
echo "$LOG_PREFIX ▶ Video copied: $video_copied"
echo "$LOG_PREFIX 🖼  Photo copied: $photo_copied"
echo "$LOG_PREFIX 🎙  Audio copied (RECORD): $audio_copied"
echo "$LOG_PREFIX 🎧 Audio extracted from video: $audio_extracted"
echo "$LOG_PREFIX 🧾 Metadata sidecars copied (all): $meta_copied"
echo "$LOG_PREFIX 📊 Video analytics copied: $video_meta_copied"
echo "$LOG_PREFIX 🗣️  Transcripts/diarization copied: $transcripts_copied"
echo "$LOG_PREFIX 💬 Captions copied: $captions_copied"
echo "$LOG_PREFIX ⚠️ Skipped (already existed): $skipped"
echo "$LOG_PREFIX ❌ Errors: $errors"
