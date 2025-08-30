#!/bin/bash

# Set the current directory and target directory
current_dir=/d
target_dir="/d/raw"

# Create the target directory
mkdir -p "$target_dir"

# Find all files with pattern _FR.MP4 or _FRA.MP4
files=$(find "$current_dir" -type f \( -name "*_FR.MP4" -o -name "*_FRA.MP4" \))

# Iterate over each file
for file in $files; do
    # Extract the base name without extension
    base_name=$(basename -- "$file" .MP4)

    # Check if _F.MP4 or _R.MP4 exists
    if [[ -f "${base_name}_F.MP4" && -f "${base_name}_R.MP4" ]]; then
        # Move _F.MP4 and _R.MP4 to the raw directory
        mv "${base_name}_F.MP4" "${base_name}_R.MP4" "$target_dir/"
    fi
done
