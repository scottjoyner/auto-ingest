#!/bin/bash

# Function to check if all files in input directories are accounted for in the output directory
validate_files() {
    for file in "${all_files[@]}"; do
        filename=$(basename "$file")
        IFS='_' read -r -a parts <<< "$filename"
        year="${parts[0]}"
        month="${parts[1]:0:2}"
        day="${parts[1]:2:2}"

        dest_dir="$output_base_path/$year/$month/$day"
        dest_file="$dest_dir/$filename"

        if [ ! -f "$dest_file" ]; then
            echo "File $filename is missing in the output directory."
            return 1
        fi
    done
    echo "All files are accounted for in the output directory."
    return 0
}

# Check if at least two arguments are provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <output_base_path> <source_dir1> <source_dir2> ... <source_dirN>"
    exit 1
fi

output_base_path="$1"
shift  # Remove the first argument, leaving only source directories

# Collect all files from the multiple input directories
all_files=()
for source_dir in "$@"; do
    if [ ! -d "$source_dir" ]; then
        echo "Source directory $source_dir does not exist."
        exit 1
    fi

    for file in "$source_dir"/*; do
        if [ -f "$file" ]; then
            all_files+=("$file")
        fi
    done
done

# Loop through each collected file
for file in "${all_files[@]}"; do
    if [ -f "$file" ]; then
        # Extract year, month, and day from the filename
        filename=$(basename "$file")
        IFS='_' read -r -a parts <<< "$filename"
        year="${parts[0]}"
        month="${parts[1]:0:2}"
        day="${parts[1]:2:2}"

        # Create directory structure if it doesn't exist in the output base path
        dest_dir="$output_base_path/$year/$month/$day"
        mkdir -p "$dest_dir"

        # Copy the file to the appropriate directory in the output base path
        cp "$file" "$dest_dir"
        echo "Copied $file to $dest_dir"
    fi
done

# Validate that all files are accounted for in the output directory
validate_files

