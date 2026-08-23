#!/usr/bin/env python3
"""
process_usb_device.py - Process all WAV files from a USB device with a stuck clock.

This script:
1. Reads DATETIME_CORRECTION.json from the USB device to get the timestamp offset
2. Renames all WAV files on the USB device using corrected timestamps
3. Copies corrected files to SSD_4TB/audio/YYYY/MM/DD/ directories
4. Transcribes each WAV file using whisper (base model on CPU)
5. Runs the auto-ingest pipeline to ingest transcriptions into Neo4j

Usage:
    python3 process_usb_device.py [--usb-path PATH] [--audio-root PATH] [--dry-run]
"""

import argparse
import csv
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample

# --- Config ---
DEFAULT_USB_PATH = "/media/scott/USB DISK"
DEFAULT_AUDIO_ROOT = "/media/scott/SSD_4TB/audio"
WHISPER_MODEL = "base"
WHISPER_DOWNLOAD_ROOT = "/tmp/whisper_models"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
NEO4J_URI = "bolt://100.64.43.123:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "knowledge_graph_2026"


def load_datetime_correction(usb_path):
    """Load the DATETIME_CORRECTION.json from the USB device."""
    correction_path = os.path.join(usb_path, "DATETIME_CORRECTION.json")
    if not os.path.isfile(correction_path):
        print(f"ERROR: {correction_path} not found. Run the datetime correction first.")
        sys.exit(1)
    with open(correction_path) as f:
        data = json.load(f)
    files = data.get("files", {})
    meta = data.get("_meta", {})
    print(f"Loaded DATETIME_CORRECTION.json: {len(files)} files, offset={meta.get('offset_days')} days")
    return files, meta


def get_existing_neo4j_keys():
    """Query Neo4j for existing transcription keys."""
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as session:
            result = session.run("MATCH (t:Transcription) RETURN t.key")
            keys = set(r["t.key"] for r in result)
        driver.close()
        return keys
    except Exception as e:
        print(f"  WARNING: Could not query Neo4j: {e}")
        return set()


def rename_usb_files(usb_record_path, correction_files, dry_run=False):
    """Rename WAV files on the USB device using corrected timestamps."""
    renamed = []
    for old_name, info in correction_files.items():
        old_path = os.path.join(usb_record_path, old_name)
        new_name = info["corrected_filename"]
        new_path = os.path.join(usb_record_path, new_name)
        if not os.path.isfile(old_path):
            if os.path.isfile(new_path):
                print(f"  Already renamed: {old_name} -> {new_name}")
                renamed.append((old_name, new_name))
                continue
            print(f"  WARNING: {old_path} not found, skipping")
            continue
        if dry_run:
            print(f"  [DRY-RUN] Would rename: {old_name} -> {new_name}")
        else:
            os.rename(old_path, new_path)
            print(f"  Renamed: {old_name} -> {new_name}")
        renamed.append((old_name, new_name))
    return renamed


def copy_to_audio_root(usb_record_path, audio_root, correction_files, existing_keys, dry_run=False):
    """Copy corrected files to audio_root/YYYY/MM/DD/ directories.
    Skips files that are already in Neo4j (by key)."""
    # Inline key derivation to avoid importing auto_ingest.ingest.transcripts
    # (which loads the embedding model and is slow)
    import re as _re
    from datetime import datetime, timezone

    def _parse_key_datetime_utc_from_string(s):
        m = _re.match(r"(\d{4})_(\d{2})_(\d{2})_(\d{2})(\d{2})(\d{2})", s)
        if not m:
            m = _re.match(r"(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})", s)
            if m:
                y, mo, d, h, mi, se = m.groups()
            else:
                return None
        else:
            y, mo, d, h, mi, se = m.groups()
        try:
            return datetime(int(y), int(mo), int(d), int(h), int(mi), int(se), tzinfo=timezone.utc)
        except ValueError:
            return None

    def _canonicalize_key(name_without_suffix):
        dt = _parse_key_datetime_utc_from_string(name_without_suffix)
        if dt:
            return dt.strftime("%Y_%m%d_%H%M%S")
        return name_without_suffix

    def _file_key_from_name(name):
        base = _re.sub(r"\.(json|txt|csv|rttm)$", "", name, flags=_re.IGNORECASE)
        base = _re.sub(r"_([A-Za-z0-9\-\._]+)_transcription(_(entites|entities))?$", "", base, flags=_re.IGNORECASE)
        base = _re.sub(r"_transcription(_(entites|entities))?$", "", base, flags=_re.IGNORECASE)
        base = _re.sub(r"_speakers$", "", base, flags=_re.IGNORECASE)
        base = _re.sub(r"_metadata$", "", base, flags=_re.IGNORECASE)
        return base

    copied = []
    skipped = []
    for old_name, info in correction_files.items():
        new_name = info["corrected_filename"]
        src_path = os.path.join(usb_record_path, new_name)
        if not os.path.isfile(src_path):
            print(f"  WARNING: {src_path} not found, skipping")
            continue
        # Parse date from corrected filename: YYYYMMDDHHMMSS.WAV
        match = re.match(r"(\d{4})(\d{2})(\d{2})\d{6}\.WAV", new_name, re.IGNORECASE)
        if not match:
            print(f"  WARNING: Cannot parse date from {new_name}, skipping")
            continue
        year, month, day = match.groups()
        dest_dir = os.path.join(audio_root, year, month, day)
        dest_path = os.path.join(dest_dir, new_name)

        # Check if already in Neo4j
        base = _file_key_from_name(new_name)
        key = _canonicalize_key(base)
        if key in existing_keys:
            print(f"  Already in Neo4j: {new_name} (key={key})")
            skipped.append((new_name, dest_path, key))
            continue

        if os.path.isfile(dest_path):
            print(f"  Already copied: {new_name} -> {dest_dir}/")
            copied.append((new_name, dest_path, key))
            continue
        if dry_run:
            print(f"  [DRY-RUN] Would copy: {src_path} -> {dest_dir}/")
        else:
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy2(src_path, dest_path)
            print(f"  Copied: {new_name} -> {dest_dir}/")
        copied.append((new_name, dest_path, key))
    return copied, skipped


def transcribe_wav(wav_path, output_dir, base_name, model=None):
    """Transcribe a WAV file using whisper and save outputs."""
    import whisper

    if model is None:
        print(f"  Loading whisper {WHISPER_MODEL} model...")
        model = whisper.load_model(WHISPER_MODEL, download_root=WHISPER_DOWNLOAD_ROOT)

    # Load WAV using scipy (no ffmpeg needed)
    sample_rate, audio_data = wavfile.read(wav_path)
    print(f"  Sample rate: {sample_rate}, shape: {audio_data.shape}")

    # Convert to mono if stereo
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)

    # Convert to float32
    audio_float = audio_data.astype(np.float32) / 32768.0

    # Resample to 16kHz if needed
    if sample_rate != 16000:
        num_samples = int(len(audio_float) * 16000 / sample_rate)
        audio_float = resample(audio_float, num_samples)
        print(f"  Resampled to 16kHz ({len(audio_float)} samples)")

    print(f"  Transcribing with whisper {WHISPER_MODEL}...")
    result = model.transcribe(audio_float, language="en")

    # Save outputs
    # _medium_transcription.txt - JSONL format (one JSON segment per line)
    # This is what the auto-ingest pipeline expects
    txt_path = os.path.join(output_dir, f"{base_name}_medium_transcription.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for seg in result["segments"]:
            f.write(json.dumps(seg) + "\n")

    # _transcription.csv - CSV format (what the pipeline reads)
    csv_path = os.path.join(output_dir, f"{base_name}_transcription.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["SegmentIndex", "StartTime", "EndTime", "Text"])
        for i, seg in enumerate(result["segments"]):
            writer.writerow([i, seg["start"], seg["end"], seg["text"]])

    # Plain text
    plain_path = os.path.join(output_dir, f"{base_name}.txt")
    with open(plain_path, "w", encoding="utf-8") as f:
        f.write(result["text"])

    # Whisper JSON (full result)
    json_path = os.path.join(output_dir, f"{base_name}_whisper.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"  Transcribed {len(result['segments'])} segments, duration: {result['segments'][-1]['end']:.1f}s")
    return result


def run_ingest(audio_root, limit=None, force=False, dry_run=False):
    """Run the auto-ingest pipeline."""
    import subprocess

    env = os.environ.copy()
    env["AUDIO_ROOT"] = audio_root
    env["FILESERVER_ROOT"] = "/media/scott/SSD_4TB/fileserver"
    env["DASHCAM_ROOT"] = "/media/scott/SSD_4TB/fileserver/dashcam"
    env["NEO4J_URI"] = NEO4J_URI
    env["NEO4J_USER"] = NEO4J_USER
    env["NEO4J_PASSWORD"] = NEO4J_PASSWORD
    env["LOG_LEVEL"] = "INFO"
    env["LOCAL_TZ"] = "America/New_York"

    cmd = [
        sys.executable,
        "/media/scott/data/git/auto-ingest/ingest_transcriptsv5_3.py",
        "--log-level", "INFO",
        "--force",
    ]
    if limit:
        cmd += ["--limit", str(limit)]
    if dry_run:
        cmd.append("--dry-run")

    print(f"\nRunning ingest: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, cwd="/media/scott/data/git/auto-ingest")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Process all WAV files from a USB device with a stuck clock.")
    parser.add_argument("--usb-path", default=DEFAULT_USB_PATH, help="Path to USB device mount point")
    parser.add_argument("--audio-root", default=DEFAULT_AUDIO_ROOT, help="Path to audio storage root")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without doing it")
    parser.add_argument("--skip-rename", action="store_true", help="Skip renaming files on USB device")
    parser.add_argument("--skip-copy", action="store_true", help="Skip copying files to audio root")
    parser.add_argument("--skip-transcribe", action="store_true", help="Skip transcription")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip Neo4j ingestion")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of files to process")
    parser.add_argument("--force", action="store_true", help="Force re-ingest")
    args = parser.parse_args()

    usb_record_path = os.path.join(args.usb_path, "RECORD")
    if not os.path.isdir(usb_record_path):
        print(f"ERROR: USB RECORD directory not found: {usb_record_path}")
        sys.exit(1)

    print("=" * 60)
    print("USB Device Processing Pipeline")
    print("=" * 60)

    # Step 1: Load datetime correction
    print("\n[1/5] Loading DATETIME_CORRECTION.json...")
    correction_files, meta = load_datetime_correction(args.usb_path)
    print(f"  Device clock range: {meta.get('device_clock_range', {}).get('earliest', 'N/A')} to {meta.get('device_clock_range', {}).get('latest', 'N/A')}")
    print(f"  Offset: {meta.get('offset_days', 'N/A')} days")

    # Step 1b: Check existing Neo4j keys
    print("\n[1b/5] Checking existing transcriptions in Neo4j...")
    existing_keys = get_existing_neo4j_keys()
    print(f"  Found {len(existing_keys)} existing transcription keys in Neo4j")

    # Step 2: Rename files on USB device
    if not args.skip_rename:
        print("\n[2/5] Renaming files on USB device...")
        renamed = rename_usb_files(usb_record_path, correction_files, dry_run=args.dry_run)
        print(f"  Renamed {len(renamed)} files")
    else:
        print("\n[2/5] Skipping rename (already done)")

    # Step 3: Copy to audio root (skip already-ingested files)
    if not args.skip_copy:
        print("\n[3/5] Copying files to audio root (skipping already-ingested)...")
        copied, skipped = copy_to_audio_root(usb_record_path, args.audio_root, correction_files, existing_keys, dry_run=args.dry_run)
        print(f"  Copied {len(copied)} files, skipped {len(skipped)} (already in Neo4j)")
    else:
        print("\n[3/5] Skipping copy")

    # Step 4: Transcribe
    if not args.skip_transcribe:
        print("\n[4/5] Transcribing files...")
        # Find all WAV files that don't have transcription files yet
        wav_files = []
        for root, _, files in os.walk(args.audio_root):
            for name in files:
                if name.lower().endswith(".wav"):
                    base = os.path.splitext(name)[0]
                    csv_path = os.path.join(root, f"{base}_transcription.csv")
                    if not os.path.isfile(csv_path):
                        wav_files.append((os.path.join(root, name), root, base))
        
        if args.limit:
            wav_files = wav_files[:args.limit]
        
        print(f"  Found {len(wav_files)} WAV files needing transcription")
        if args.dry_run:
            for wav_path, _, base in wav_files[:5]:
                print(f"  [DRY-RUN] Would transcribe: {wav_path}")
        else:
            import whisper
            model = whisper.load_model(WHISPER_MODEL, download_root=WHISPER_DOWNLOAD_ROOT)
            for wav_path, output_dir, base in wav_files:
                print(f"\n  Transcribing: {os.path.basename(wav_path)}")
                transcribe_wav(wav_path, output_dir, base, model=model)
    else:
        print("\n[4/5] Skipping transcription")

    # Step 5: Ingest into Neo4j
    if not args.skip_ingest:
        print("\n[5/5] Running auto-ingest pipeline...")
        rc = run_ingest(args.audio_root, limit=args.limit, force=True, dry_run=args.dry_run)
        if rc != 0:
            print(f"  Ingest pipeline returned code {rc}")
    else:
        print("\n[5/5] Skipping ingest")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()