#!/usr/bin/env python3
"""Extract audio from dashcam video files, producing one MP3 per frame (F/R)."""

import os, re, uuid, csv, json, time, subprocess as sp

from auto_ingest_config import get_dashcam_root

def list_files(directory):
    """Find all valid video keys in a directory.
    
    Supports legacy formats (_F.MP4), numeric timestamp formats with frame suffixes
    (2025_0101_002327_000008F.MP4, 2025_0101_002427_000011R.MP4), and plain numeric
    timestamps without frame suffixes (20250713105005_000013.MP4)."""
    file_keys = set()
    for filename in os.listdir(directory):
        # New numeric format patterns with F/R frame indicators
        if re.search(r"_\d+F\.mp4$", filename, re.IGNORECASE):
            file_keys.add(filename.rsplit('.', 1)[0])
        elif re.search(r"_\d+R\.mp4$", filename, re.IGNORECASE):
            file_keys.add(filename.rsplit('.', 1)[0])
        # Legacy pattern (backward compatible)
        if re.search("_F.MP4", filename):
            file_keys.add(filename.rsplit('.', 1)[0])
        # Generic numeric timestamp without frame suffix — catches bodycam and other formats
        # e.g. 20250713105005_000013.MP4 or 2025_0101_173933.MP4
        if re.search(r"_\d+\.mp4$", filename, re.IGNORECASE):
            file_keys.add(filename.rsplit('.', 1)[0])
    return sorted(file_keys)

def is_valid_date_structure(dir_name):
    """Check if directory name is a valid YYYY/MM/DD date structure."""
    try:
        from datetime import datetime
        datetime.strptime(dir_name, "%Y/%m/%d")
        return True
    except ValueError:
        return False

def list_directories(base_path):
    for root, dirs, files in os.walk(base_path):
        path_parts = root[len(base_path):].split(os.sep)
        normalized_path = "/".join(path_parts)
        temp_path = root[len(base_path):]
        if is_valid_date_structure(temp_path):
            print(f"Valid directory structure found: {root}")
            file_path = root
            dashcam_root = get_dashcam_root()  # canonical local cache/processing dashcam root
            transcriptions = os.path.join(dashcam_root, "transcriptions")
            audio_dir = os.path.join(dashcam_root, "audio")
            
            key_list = list_files(file_path)
            total = len(key_list)
            current = 1
            print(f"{total} Videos left to process")
            for x in range(len(key_list)):
                if os.path.exists(os.path.join(transcriptions, f"{key_list[x]}_medium_transcription.txt")):
                    with open(os.path.join(transcriptions, f"{key_list[x]}_medium_transcription.txt"), 'r') as file:
                        trans = file.read()
                        mp3_path = os.path.join(audio_dir, f"{key_list[x]}.mp3")
                        if not os.path.exists(mp3_path):
                            try:
                                print(f"Extracting audio for {key_list[x]}...")
                                video_path = os.path.join(file_path, f"{key_list[x]}.MP4")
                                sp.run(
                                    ["ffmpeg", "-y", "-i", video_path,
                                     "-ar", "16000", "-ac", "mono", "-c:a", "libmp3lame", 
                                     mp3_path],
                                    stdout=sp.PIPE, stderr=sp.PIPE
                                )
                                if os.path.exists(mp3_path):
                                    print(f"✓ Extracted: {key_list[x]} -> {mp3_path}")
                                else:
                                    print(f"✗ Failed: {key_list[x]}")
                            except Exception as e:
                                print(f"  Error extracting {key_list[x]}: {e}")
                current += 1

def main():
    dashcam_root = get_dashcam_root()
    
    # Collect all valid date directories
    for root, dirs, files in os.walk(dashcam_root):
        temp_path = root[len(dashcam_root):]
        
        if is_valid_date_structure(temp_path):
            print(f"Valid directory structure found: {root}")
            
            # List and process video files
            file_keys = list_files(root)
            
            for key in file_keys:
                mp3_dir = os.path.join(dashcam_root, "audio")
                trans_dir = os.path.join(dashcam_root, "transcriptions")
                
                if not os.path.exists(os.path.join(trans_dir, f"{key}_medium_transcription.txt")):
                    continue
                    
                with open(os.path.join(trans_dir, f"{key}_medium_transcription.txt"), 'r') as file:
                    trans = file.read()
                    
                mp3_path = os.path.join(mp3_dir, f"{key}.mp3")
                if not os.path.exists(mp3_path):
                    try:
                        video_path = os.path.join(root, f"{key}.MP4")
                        sp.run(
                            ["ffmpeg", "-y", "-i", video_path,
                             "-ar", "16000", "-ac", "mono", "-c:a", "libmp3lame", 
                             mp3_path],
                            stdout=sp.PIPE, stderr=sp.PIPE
                        )
                        if os.path.exists(mp3_path):
                            print(f"✓ Extracted: {key}")
                        else:
                            print(f"✗ Failed: {key}")
                    except Exception as e:
                        print(f"  Error extracting {key}: {e}")

if __name__ == "__main__":
    main()
