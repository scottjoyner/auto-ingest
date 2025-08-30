#!/bin/bash

# Directory containing the MP4 files
dir="/Volumes/Untitled/DCIM/Movie"

# Navigate to the directory
cd "$dir"

# Loop through files ending with "_FR.MP4"
for fr_file in *"_FR.MP4"; do
  # Check if the file actually exists (in case there are no matches)
  if [ ! -f "$fr_file" ]; then
    continue
  fi

  # Extract the base name by removing "_FR.MP4" suffix
  base_name="${fr_file%_FR.MP4}"

  # Construct the name of the corresponding "_F.MP4" file
  f_file="${base_name}_F.MP4"

  # Check if the corresponding "_F.MP4" file exists
  if [ -f "$f_file" ]; then
    # Construct the output file name
    output_file="${base_name}_FRA.mp4"

    # Run the FFmpeg command
    ffmpeg -i "$fr_file" -i "$f_file" -c:v copy -map 0:v:0 -map 1:a:0 "$output_file"
  else
    echo "Corresponding file $f_file not found for $fr_file"
  fi
done
