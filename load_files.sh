#!/bin/bash
set -euo pipefail
shopt -s nocasematch

# Organize common media files from one or more source directories into:
#   <output_base_path>/YYYY/MM/DD/<original_filename>
#
# Timestamp parsing is intentionally flexible for recorder/camera filenames such as:
#   2025_0911_223045_F.MP4
#   20250911223045.WAV
#   R2025_0911_223045.mp3
#   REC_2025-09-11_22-30-45.m4a
# If no valid timestamp is found in the name, the file modified date is used.

MEDIA_EXT_REGEX='\.(3gp|aac|aiff|amr|avi|flac|m4a|m4v|mkv|mov|mp3|mp4|mpeg|mpg|oga|ogg|opus|wav|webm|wma)$'

usage() {
  echo "Usage: $0 <output_base_path> <source_dir1> <source_dir2> ... <source_dirN>"
}

parse_date_from_name() {
  local filename="$1"
  local parsed

  parsed=$(python3 - "$filename" <<'PY'
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
        print(dt.strftime("%Y/%m/%d"))
        sys.exit(0)
sys.exit(1)
PY
  ) && printf '%s' "$parsed"
}

file_date_path() {
  local file="$1"
  local filename date_path
  filename=$(basename "$file")

  if date_path=$(parse_date_from_name "$filename"); then
    printf '%s' "$date_path"
    return 0
  fi

  # Fallback to mtime when filename is not parseable.
  if date -r "$file" '+%Y/%m/%d' >/dev/null 2>&1; then
    date -r "$file" '+%Y/%m/%d'
  else
    stat -c '%y' "$file" | cut -d' ' -f1 | tr '-' '/'
  fi
}

validate_files() {
  local missing=0 file filename date_path dest_file
  for file in "${all_files[@]}"; do
    filename=$(basename "$file")
    if [[ ! "$filename" =~ $MEDIA_EXT_REGEX ]]; then
      continue
    fi
    date_path=$(file_date_path "$file")
    dest_file="$output_base_path/$date_path/$filename"

    if [ ! -f "$dest_file" ]; then
      echo "File $filename is missing in the output directory: $dest_file"
      missing=1
    fi
  done

  if [ "$missing" -eq 0 ]; then
    echo "All files are accounted for in the output directory."
    return 0
  fi
  return 1
}

if [ "$#" -lt 2 ]; then
  usage
  exit 1
fi

output_base_path="$1"
shift

all_files=()
for source_dir in "$@"; do
  if [ ! -d "$source_dir" ]; then
    echo "Source directory $source_dir does not exist."
    exit 1
  fi

  while IFS= read -r -d '' file; do
    all_files+=("$file")
  done < <(find "$source_dir" -maxdepth 1 -type f -print0)
done

for file in "${all_files[@]}"; do
  filename=$(basename "$file")
  if [[ ! "$filename" =~ $MEDIA_EXT_REGEX ]]; then
    echo "Skipping non-media file: $file"
    continue
  fi

  date_path=$(file_date_path "$file")
  dest_dir="$output_base_path/$date_path"
  mkdir -p "$dest_dir"

  if [ -e "$dest_dir/$filename" ]; then
    echo "Exists, skipping: $dest_dir/$filename"
  else
    cp -- "$file" "$dest_dir/"
    echo "Copied $file to $dest_dir"
  fi
done

validate_files
