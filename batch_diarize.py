#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path
from tqdm import tqdm
import argparse

from diarize_and_transcribe import (  # import our previous functions
    pick_device,
    ensure_mono16k,
    build_diarization_pipeline,
    diarize_audio,
    stitch_per_speaker_audio,
    load_whisper,
    transcribe_chunk,
    SegmentResult,
    write_outputs,
    write_wav,
    segment_to_frames
)

import soundfile as sf

def find_wav_files(root: Path):
    """
    Recursively find all .wav files under root.
    """
    return sorted(p for p in root.rglob("*.wav") if p.is_file())


def process_one_file(
    wav_path: Path,
    out_root: Path,
    diar_pipe,
    asr,
    sr_target: int,
    args
):
    rel_path = wav_path.relative_to(args.indir)
    out_dir = out_root / rel_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    base_name = wav_path.stem
    artifact_dir = out_dir / base_name
    artifact_dir.mkdir(parents=True, exist_ok=True)

    out_base = artifact_dir / f"{base_name}_speaker_transcript"

    # Skip if already done
    if args.skip_existing and out_base.with_suffix(".json").exists():
        if args.verbose:
            print(f"⏩ Skipping {wav_path} (already processed)")
        return

    if args.verbose:
        print(f"🎧 Processing {wav_path}")

    # Load & normalize
    waveform, sr = ensure_mono16k(wav_path)

    # Diarization
    diarization = diarize_audio(
        input_wav=wav_path,
        diar_pipeline=diar_pipe,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
    )

    # Export per-speaker WAVs
    per_spk_dir = artifact_dir / "speakers"
    stitch_per_speaker_audio(diarization, waveform, sr, per_spk_dir, min_len_sec=args.min_segment_len)

    # Transcribe each segment
    results = []
    seg_dir = artifact_dir / "segments"
    if args.export_segments:
        seg_dir.mkdir(parents=True, exist_ok=True)

    iterables = list(diarization.itertracks(yield_label=True))
    for (segment, _, label) in iterables:
        start_f, end_f = segment_to_frames(segment, sr)
        dur = (end_f - start_f) / sr
        if dur < args.min_segment_len:
            continue

        chunk = waveform[start_f:end_f]
        text, words = transcribe_chunk(
            model=asr,
            audio=chunk,
            sr=sr,
            language=args.asr_lang,
            beam_size=args.asr_beam,
            vad_filter=args.vad_filter,
        )

        if args.export_segments:
            seg_name = f"{label}_{segment.start:.3f}-{segment.end:.3f}.wav"
            write_wav(seg_dir / seg_name, chunk, sr)

        results.append(SegmentResult(
            speaker=label,
            start=float(segment.start),
            end=float(segment.end),
            text=text,
            words=words
        ))

    results.sort(key=lambda r: (r.start, r.end))
    write_outputs(results, out_base)

    if args.verbose:
        print(f"✅ Finished {wav_path}")


def main():
    parser = argparse.ArgumentParser(description="Batch diarization & transcription on YYYY/MM/DD structure.")
    parser.add_argument("--indir", type=str, required=True, help="Input root directory containing YYYY/MM/DD folders")
    parser.add_argument("--outdir", type=str, required=True, help="Output root directory for results")
    parser.add_argument("--hf_token", type=str, default=os.getenv("HUGGINGFACE_TOKEN"), help="Hugging Face token")
    parser.add_argument("--asr_model", type=str, default="medium", help="faster-whisper model name")
    parser.add_argument("--asr_lang", type=str, default=None)
    parser.add_argument("--asr_beam", type=int, default=5)
    parser.add_argument("--asr_compute", type=str, default="float16")
    parser.add_argument("--num_speakers", type=int, default=None)
    parser.add_argument("--min_speakers", type=int, default=None)
    parser.add_argument("--max_speakers", type=int, default=None)
    parser.add_argument("--export_segments", action="store_true")
    parser.add_argument("--min_segment_len", type=float, default=0.0)
    parser.add_argument("--vad_filter", action="store_true")
    parser.add_argument("--skip_existing", action="store_true", help="Skip files with existing JSON output")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    indir = Path(args.indir).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()

    if not indir.exists():
        print(f"❌ Input directory not found: {indir}", file=sys.stderr)
        sys.exit(1)
    if not args.hf_token:
        print("❌ Missing Hugging Face token.", file=sys.stderr)
        sys.exit(1)

    device = pick_device(prefer_gpu=True)
    if args.verbose:
        print(f"🖥️ Using device: {device}")

    # Load pipelines & ASR once
    diar_pipe = build_diarization_pipeline(args.hf_token, device)
    asr = load_whisper(args.asr_model, device, args.asr_compute)

    wav_files = find_wav_files(indir)
    if not wav_files:
        print("⚠️ No .wav files found.")
        return

    print(f"📂 Found {len(wav_files)} WAV files under {indir}")

    for wav in tqdm(wav_files):
        process_one_file(wav, outdir, diar_pipe, asr, 16000, args)


if __name__ == "__main__":
    main()

