#!/usr/bin/env bash
#
# combine_metadata.sh
# Searches all subdirectories for files named "*_metadata.csv" 
# and concatenates them into one output file.

# -- Configuration --
ROOT_DIR="/media/deathstar/8TBHDD/fileserver/dashcam"   # Update this path
OUTPUT_FILE="merged_yolo.csv"        # Name of your concatenated file

# --- Script Start ---

# 1. Go to the root directory
cd "$ROOT_DIR" || {
  echo "ERROR: Could not change to directory $ROOT_DIR"
  exit 1
}

# 2. Find and concatenate
#    -type f        => look for files only
#    -name "*_metadata.csv" => pattern to match
#    -exec cat {}   => cat each found file
#    >>             => append to the output file
#
# The parentheses and + combine all found files in as few commands as possible
# for performance. Using \; instead of + runs cat once per file (slower).
#
find . -type f -name "*_YOLOv8n.csv" -exec cat {} + >> "$OUTPUT_FILE"

echo "All *_YOLOv8n.csv files have been concatenated into '$OUTPUT_FILE'."
