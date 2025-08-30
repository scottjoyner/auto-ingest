#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import io
import sys
import json
import math
import argparse
import tempfile
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import soundfile as sf
import torchaudio
from tqdm import tqdm
import pandas as pd

# pyannote
from pyannote.audio import Pipeline, Model
from pyannote.core import Annotation, Segment
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.audio.pipelines import VoiceActivityDetection, OverlappedSpeechDetection

# ASR (fast, GPU-capable)
from faster_whisper import WhisperModel


# -----------------------------
# Utility & IO
# -----------------------------

def pick_device(prefer_gpu: bool = True) -> torch.device:
    """
    Choose the best-available Torch device.
    Notes:
      - pyannote pipelines support CUDA/ROCm (torch.cuda.is_available()) and Apple MPS.
      - Vulkan backend is not supported by pyannote; use ROCm for AMD or CUDA for NVIDIA.
    """
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    try:
        if prefer_gpu and torch.backends.mps.is_available():  # Apple Silicon
            return torch.device("mps")
    except Exception:
        pass
    return torch.device("cpu")


def ensure_mono16k(path: Path) -> Tuple[np.ndarray, int]:
    """
    Load audio, downmix to mono, and resample to 16kHz (pyannote & Whisper-friendly).
    Returns (waveform_float32, sample_rate)
    """
    waveform, sr = torchaudio.load(str(path))
    if waveform.ndim == 2 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        sr = 16000
    # (channels=1, samples) -> (samples,)
    return waveform.squeeze(0).numpy().astype(np.float32), sr


def segment_to_frames(segment: Segment, sr: int) -> Tuple[int, int]:
    """Convert a pyannote Segment (seconds) to sample frame indices."""
    start = max(0, int(round(segment.start * sr)))
    end = max(start, int(round(segment.end * sr)))
    return start, end


def write_wav(path: Path, waveform: np.ndarray, sr: int):
    """Write float32 waveform to WAV preserving sr."""
    sf.write(str(path), waveform, sr, subtype="PCM_16")


@dataclass
class SegmentResult:
    speaker: str
    start: float
    end: float
    text: str
    words: Optional[List[Dict]] = None
    # Optionally include confidence or avg word prob later


# -----------------------------
# Diarization & VAD/OSD
# -----------------------------

def build_diarization_pipeline(hf_token: str, device: torch.device) -> Pipeline:
    """
    Load pyannote speaker-diarization 3.1 pipeline.
    """
    pipe = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token
    )
    pipe.to(device)
    return pipe


def optional_vad_osd(segmentation_model_token: str, hf_token: str):
    """
    Create VAD/OSD pipelines using the segmentation model (optional).
    Useful if you need custom on/off thresholds or to pre-trim long non-speech regions.
    """
    model = Model.from_pretrained(
        "pyannote/segmentation-3.0",
        use_auth_token=hf_token
    )
    vad = VoiceActivityDetection(segmentation=model)
    osd = OverlappedSpeechDetection(segmentation=model)
    # Default hyperparams can be tuned here:
    vad_params = {"min_duration_on": 0.0, "min_duration_off": 0.0}
    osd_params = {"min_duration_on": 0.0, "min_duration_off": 0.0}
    vad.instantiate(vad_params)
    osd.instantiate(osd_params)
    return vad, osd


def diarize_audio(
    input_wav: Path,
    diar_pipeline: Pipeline,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> Annotation:
    """
    Run diarization on the given audio file with progress monitoring.
    """
    kwargs = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    if min_speakers is not None:
        kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        kwargs["max_speakers"] = max_speakers

    with ProgressHook() as hook:
        diar = diar_pipeline(str(input_wav), hook=hook, **kwargs)
    return diar


# -----------------------------
# ASR with Whisper (faster-whisper)
# -----------------------------

def load_whisper(model_name: str, device: torch.device, compute_type: str):
    """
    Load a faster-whisper model.
    compute_type examples: "float16", "float32", "int8", "int8_float16".
    """
    # Map torch device to faster-whisper device string
    if device.type == "cuda":
        dev = "cuda"
    elif device.type == "mps":
        # faster-whisper does not use MPS; fallback to CPU here.
        dev = "cpu"
    else:
        dev = "cpu"

    return WhisperModel(model_name, device=dev, compute_type=compute_type)


def transcribe_chunk(
    model: WhisperModel,
    audio: np.ndarray,
    sr: int,
    language: Optional[str],
    beam_size: int,
    vad_filter: bool,
) -> Tuple[str, List[Dict]]:
    """
    Transcribe a single audio chunk (float32 mono at sr).
    Returns (text, words) where words include timestamps.
    """
    # faster-whisper expects float32 in [-1, 1]; we already have float32.
    # Use word timestamps for alignment with diarization
    segments, info = model.transcribe(
        audio,
        language=language,
        beam_size=beam_size,
        vad_filter=vad_filter,
        word_timestamps=True,
    )
    text_parts, words = [], []
    for seg in segments:
        if seg.text:
            text_parts.append(seg.text.strip())
        if seg.words:
            for w in seg.words:
                words.append({
                    "word": w.word,
                    "start": float(w.start) if w.start is not None else None,
                    "end": float(w.end) if w.end is not None else None,
                    "prob": float(w.probability) if w.probability is not None else None
                })
    return (" ".join(text_parts).strip(), words)


# -----------------------------
# Main processing
# -----------------------------

def stitch_per_speaker_audio(
    diarization: Annotation,
    waveform: np.ndarray,
    sr: int,
    out_dir: Path,
    min_len_sec: float = 0.0,
):
    """
    Concatenate all segments per speaker and write per-speaker WAVs.
    Also return a dict of per-speaker arrays.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    speakers = sorted(set(diarization.labels()))
    speaker_audio: Dict[str, List[np.ndarray]] = {spk: [] for spk in speakers}

    for segment, _, label in diarization.itertracks(yield_label=True):
        start_f, end_f = segment_to_frames(segment, sr)
        if (end_f - start_f) / sr < min_len_sec:
            continue
        speaker_audio[label].append(waveform[start_f:end_f].copy())

    stitched = {}
    for spk in speakers:
        if speaker_audio[spk]:
            cat = np.concatenate(speaker_audio[spk]) if len(speaker_audio[spk]) > 1 else speaker_audio[spk][0]
            stitched[spk] = cat
            write_wav(out_dir / f"{spk}.wav", cat, sr)
    return stitched


def srt_timestamp(seconds: float) -> str:
    """Format seconds into SRT timestamp."""
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - math.floor(seconds)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_outputs(
    results: List[SegmentResult],
    out_base: Path,
):
    """
    Write JSON, CSV, and SRT files of speaker-attributed transcripts.
    """
    # JSON
    data = [asdict(r) for r in results]
    (out_base.with_suffix(".json")).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # CSV
    rows = []
    for r in results:
        rows.append({
            "speaker": r.speaker,
            "start": f"{r.start:.3f}",
            "end": f"{r.end:.3f}",
            "text": r.text
        })
    df = pd.DataFrame(rows, columns=["speaker", "start", "end", "text"])
    df.to_csv(out_base.with_suffix(".csv"), index=False, encoding="utf-8")

    # SRT
    srt_lines = []
    for i, r in enumerate(results, start=1):
        srt_lines.append(str(i))
        srt_lines.append(f"{srt_timestamp(r.start)} --> {srt_timestamp(r.end)}")
        srt_lines.append(f"{r.speaker}: {r.text}".strip())
        srt_lines.append("")  # blank line
    (out_base.with_suffix(".srt")).write_text("\n".join(srt_lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Diarize large WAVs, isolate speakers, and transcribe per segment.")
    parser.add_argument("audio", type=str, help="Path to input WAV file")
    parser.add_argument("--outdir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--hf_token", type=str, default=os.getenv("HUGGINGFACE_TOKEN"), help="Hugging Face token")
    parser.add_argument("--asr_model", type=str, default="medium", help="faster-whisper model (e.g., small, medium, large-v3)")
    parser.add_argument("--asr_lang", type=str, default=None, help="ISO language hint (e.g., en)")
    parser.add_argument("--asr_beam", type=int, default=5, help="Beam size for Whisper")
    parser.add_argument("--asr_compute", type=str, default="float16", help="Whisper compute_type (float16, float32, int8, int8_float16)")
    parser.add_argument("--num_speakers", type=int, default=None, help="If known, fix number of speakers")
    parser.add_argument("--min_speakers", type=int, default=None, help="Min speakers")
    parser.add_argument("--max_speakers", type=int, default=None, help="Max speakers")
    parser.add_argument("--export_segments", action="store_true", help="Export each diarized segment as a WAV")
    parser.add_argument("--min_segment_len", type=float, default=0.0, help="Drop segments shorter than this many seconds")
    parser.add_argument("--vad_filter", action="store_true", help="Apply Whisper VAD filter during ASR")
    args = parser.parse_args()

    audio_path = Path(args.audio).expanduser().resolve()
    if not audio_path.exists():
        print(f"❌ File not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    if not args.hf_token:
        print("❌ Missing Hugging Face token. Set HUGGINGFACE_TOKEN or pass --hf_token.", file=sys.stderr)
        sys.exit(1)

    out_root = Path(args.outdir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    base_name = audio_path.stem
    artifact_dir = out_root / base_name
    artifact_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(prefer_gpu=True)
    print(f"🖥️  Using device: {device}")

    # Load audio (mono, 16k)
    print("🎧 Loading & normalizing audio...")
    waveform, sr = ensure_mono16k(audio_path)

    # ----- Diarization -----
    print("🗣️  Running speaker diarization (pyannote/speaker-diarization-3.1)...")
    diar_pipe = build_diarization_pipeline(args.hf_token, device)
    diarization = diarize_audio(
        input_wav=audio_path,
        diar_pipeline=diar_pipe,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
    )

    # Optional: save RTTM for debugging/interop
    rttm_path = artifact_dir / f"{base_name}.rttm"
    with rttm_path.open("w") as rttm:
        diarization.write_rttm(rttm)
    print(f"📝 RTTM written: {rttm_path}")

    # ----- Export per-speaker concatenated audio -----
    print("🔊 Exporting per-speaker WAVs...")
    per_spk_dir = artifact_dir / "speakers"
    stitch_per_speaker_audio(diarization, waveform, sr, per_spk_dir, min_len_sec=args.min_segment_len)
    print(f"📁 Speaker WAVs in: {per_spk_dir}")

    # ----- ASR model -----
    print(f"🔎 Loading Whisper model: {args.asr_model} (compute={args.asr_compute}) ...")
    asr = load_whisper(args.asr_model, device, args.asr_compute)

    # ----- Transcribe each diarized segment -----
    print("🧩 Transcribing diarized segments...")
    results: List[SegmentResult] = []
    seg_dir = artifact_dir / "segments"
    if args.export_segments:
        seg_dir.mkdir(parents=True, exist_ok=True)

    # Iterate tracks; overlapped speech will yield multiple labels for same time span
    # We keep each labeled segment separate to preserve who-said-what in overlaps.
    iterables = list(diarization.itertracks(yield_label=True))
    for (segment, _, label) in tqdm(iterables, total=len(iterables)):
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

        # Optionally export each segment WAV
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

    # Sort chronologically
    results.sort(key=lambda r: (r.start, r.end))

    # ----- Write combined artifacts -----
    out_base = artifact_dir / f"{base_name}_speaker_transcript"
    write_outputs(results, out_base)

    print("✅ Done.")
    print(f"📦 Outputs:\n  - JSON/CSV/SRT: {out_base.with_suffix('.json')}, {out_base.with_suffix('.csv')}, {out_base.with_suffix('.srt')}\n  - RTTM: {rttm_path}\n  - Per-speaker WAVs: {per_spk_dir}\n  - Per-segment WAVs: {'generated in ' + str(seg_dir) if args.export_segments else 'disabled'}")


if __name__ == "__main__":
    main()

