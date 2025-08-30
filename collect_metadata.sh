#!/usr/bin/env bash
#
# combine_metadata_with_header.sh
# Recursively finds "*_metadata.csv" and concatenates into a single CSV, with headers only once.

ROOT_DIR="/media/deathstar/8TB_2025/fileserver/dashcam/"  
OUTPUT_FILE="part_2_merged_metadata.csv"

# 1. Change to the root directory
cd "$ROOT_DIR" || {
  echo "ERROR: Could not change to directory $ROOT_DIR"
  exit 1
}

# 2. Capture a list of all relevant files
files=$(find . -type f -name "*_metadata.csv" | sort)

# If no files were found, exit
if [ -z "$files" ]; then
  echo "No files found matching *_metadata.csv in $ROOT_DIR."
  exit 0
fi

# 3. Get the first file to copy the header
first_file=$(echo "$files" | head -n 1)

# Write the header (first line of first CSV) to the output file
head -n 1 "$first_file" > "$OUTPUT_FILE"

# 4. For each file, skip the first line (header) and append the rest
#    tail -n +2 => start from line 2 onward
while IFS= read -r file; do
  tail -n +2 "$file" >> "$OUTPUT_FILE"
done <<< "$files"

echo "All *_metadata.csv files have been concatenated into '$OUTPUT_FILE' with a single header."
