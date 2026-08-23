#!/usr/bin/env python3
"""
transcribe_usb_batch.py - Batch transcribe WAV files from USB device using whisper.

Processes files in batches, saving progress so it can be resumed.
Uses the ryzen-ai-venv with whisper installed.

Usage:
    python3 transcribe_usb_batch.py [--batch-size N] [--start-from N]
"""

import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample

WHISPER_MODEL = "base"
WHISPER_DOWNLOAD_ROOT = "/tmp/whisper_models"
AUDIO_ROOT = "/media/scott/SSD_4TB/audio"
CORRECTION_FILE = "/media/scott/USB DISK/DATETIME_CORRECTION.json"
PROGRESS_FILE = "/media/scott/data/git/auto-ingest/transcribe_progress.json"


def get_usb_wav_files():
    """Find WAV files from the USB device that need transcription."""
    with open(CORRECTION_FILE) as f:
        data = json.load(f)
    files = data.get("files", {})
    corrected_names = set(info["corrected_filename"] for info in files.values())

    wav_files = []
    for root, _, file_list in os.walk(AUDIO_ROOT):
        for name in file_list:
            if name.lower().endswith(".wav") and name in corrected_names:
                base = os.path.splitext(name)[0]
                csv_path = os.path.join(root, f"{base}_transcription.csv")
                if not os.path.isfile(csv_path):
                    wav_files.append((os.path.join(root, name), root, base))
    return sorted(wav_files, key=lambda x: x[2])


def load_progress():
    """Load progress from file."""
    if os.path.isfile(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"completed": [], "failed": [], "started_at": None}


def save_progress(progress):
    """Save progress to file."""
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def transcribe_wav(wav_path, output_dir, base_name, model):
    """Transcribe a WAV file using whisper and save outputs."""
    # Load WAV using scipy (no ffmpeg needed)
    sample_rate, audio_data = wavfile.read(wav_path)

    # Convert to mono if stereo
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)

    # Convert to float32
    audio_float = audio_data.astype(np.float32) / 32768.0

    # Resample to 16kHz if needed
    if sample_rate != 16000:
        num_samples = int(len(audio_float) * 16000 / sample_rate)
        audio_float = resample(audio_float, num_samples)

    result = model.transcribe(audio_float, language="en")

    # Save _medium_transcription.txt (JSONL format)
    txt_path = os.path.join(output_dir, f"{base_name}_medium_transcription.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for seg in result["segments"]:
            f.write(json.dumps(seg) + "\n")

    # Save _transcription.csv
    csv_path = os.path.join(output_dir, f"{base_name}_transcription.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["SegmentIndex", "StartTime", "EndTime", "Text"])
        for i, seg in enumerate(result["segments"]):
            writer.writerow([i, seg["start"], seg["end"], seg["text"]])

    # Save plain text
    plain_path = os.path.join(output_dir, f"{base_name}.txt")
    with open(plain_path, "w", encoding="utf-8") as f:
        f.write(result["text"])

    # Save whisper JSON
    json_path = os.path.join(output_dir, f"{base_name}_whisper.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return len(result["segments"]), result["segments"][-1]["end"]


def main():
    parser = argparse.ArgumentParser(description="Batch transcribe WAV files from USB device")
    parser.add_argument("--batch-size", type=int, default=10, help="Number of files to process before saving progress")
    parser.add_argument("--start-from", type=int, default=0, help="Start from this file index")
    args = parser.parse_args()

    print("Loading progress...")
    progress = load_progress()
    if progress["started_at"] is None:
        progress["started_at"] = datetime.now().isoformat()
    
    completed = set(progress.get("completed", []))
    failed = set(progress.get("failed", []))

    print("Finding WAV files...")
    wav_files = get_usb_wav_files()
    print(f"Found {len(wav_files)} WAV files needing transcription")
    
    # Skip already completed
    remaining = [(p, d, b) for p, d, b in wav_files if b not in completed and b not in failed]
    if args.start_from > 0:
        remaining = remaining[args.start_from:]
    
    print(f"Remaining to process: {len(remaining)}")
    if not remaining:
        print("All files already processed!")
        return

    print(f"Loading whisper {WHISPER_MODEL} model...")
    import whisper
    model = whisper.load_model(WHISPER_MODEL, download_root=WHISPER_DOWNLOAD_ROOT)
    print("Model loaded!")

    for i, (wav_path, output_dir, base_name) in enumerate(remaining):
        print(f"\n[{i+1}/{len(remaining)}] Transcribing: {os.path.basename(wav_path)}")
        try:
            seg_count, duration = transcribe_wav(wav_path, output_dir, base_name, model)
            completed.add(base_name)
            progress["completed"] = list(completed)
            print(f"  OK: {seg_count} segments, {duration:.1f}s")
            
            if (i + 1) % args.batch_size == 0:
                save_progress(progress)
                print(f"  Progress saved: {len(completed)} completed, {len(failed)} failed")
        except Exception as e:
            failed.add(base_name)
            progress["failed"] = list(failed)
            print(f"  FAILED: {e}")
            save_progress(progress)

    save_progress(progress)
    print(f"\nDone! Completed: {len(completed)}, Failed: {len(failed)}")


if __name__ == "__main__":
    main()