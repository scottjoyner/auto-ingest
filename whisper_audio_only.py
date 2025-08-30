#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Tuple, List

import torch
import whisper
from tqdm import tqdm


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def pick_device(user_device: Optional[str]) -> str:
    if user_device in ("cpu", "cuda"):
        return user_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def list_audio_files(root: Path) -> List[Path]:
    exts = {".wav", ".mp3"}
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in exts])


def out_paths(
    audio_path: Path, audio_root: Path, trans_root: Path, model_name: str
) -> Tuple[Path, Path]:
    """
    Mirror the YYYY/MM/DD tree into the transcriptions root and
    create outputs that match your existing naming convention:
      - <stem>_{model}_transcription.txt  (JSON dump)
      - <stem>_transcription.csv         (segments CSV)
    """
    rel = audio_path.relative_to(audio_root)
    dest_dir = trans_root / rel.parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    stem = rel.stem
    dump_path = dest_dir / f"{stem}_{model_name}_transcription.txt"  # JSON (as in your script)
    csv_path = dest_dir / f"{stem}_transcription.csv"
    return dump_path, csv_path


def already_done(dump_path: Path, csv_path: Path) -> bool:
    return dump_path.exists() and csv_path.exists()


def save_raw_result(whisper_result: dict, file_path: Path) -> None:
    # Your example writes JSON content into a .txt — we’ll do the same.
    file_path.write_text(json.dumps(whisper_result, ensure_ascii=False, indent=2), encoding="utf-8")


def save_transcription_segments_to_csv(whisper_result: dict, file_path: Path) -> None:
    segments = whisper_result.get("segments") or []
    with file_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["SegmentIndex", "StartTime", "EndTime", "Text"])
        for i, seg in enumerate(segments):
            w.writerow([i, seg.get("start"), seg.get("end"), (seg.get("text") or "").strip()])


def load_model(model_name: str, device: str):
    try:
        return whisper.load_model(model_name, device=device)
    except Exception as e:
        print(f"❌ Failed to load model '{model_name}' on '{device}': {e}", file=sys.stderr)
        sys.exit(2)


def transcribe_file(
    model,
    audio_path: Path,
    device: str,
    language: Optional[str],
    temperature: float,
    beam_size: int,
) -> dict:
    # fp16 for CUDA, fp32 on CPU
    fp16 = device == "cuda"
    result = model.transcribe(
        str(audio_path),
        language=language,            # None = auto-detect
        temperature=temperature,      # 0.0 for determinism
        beam_size=beam_size,          # beam search improves accuracy
        word_timestamps=True,         # match your example’s behavior
        condition_on_previous_text=True,
        fp16=fp16,
        verbose=False,
        # Optionals to tweak if needed:
        # no_speech_threshold=0.6,
        # logprob_threshold=-1.0,
        # compression_ratio_threshold=2.4,
    )
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe WAV/MP3 with Whisper; save a JSON dump (*.txt) and a segments CSV, mirroring YYYY/MM/DD."
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=Path("/mnt/8TB_2025/fileserver/audio"),
        help="Root directory to scan for .wav/.mp3 files.",
    )
    parser.add_argument(
        "--transcriptions-root",
        type=Path,
        default=None,
        help="Directory where outputs are written; defaults to <audio-root>/transcriptions.",
    )
    parser.add_argument(
        "--model",
        choices=["medium", "large-v3"],
        default="large-v3",
        help="Whisper model to use.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="Force device (default: auto-detect).",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Force language code (e.g., 'en'); default None = auto-detect.",
    )
    parser.add_argument(
        "--beam-size", type=int, default=5, help="Beam size for decoding (accuracy vs speed)."
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Sampling temperature (0.0 = deterministic)."
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Recreate outputs even if they already exist."
    )
    parser.add_argument(
        "--fail-fast", action="store_true", help="Stop on first error."
    )
    args = parser.parse_args()

    audio_root: Path = args.audio_root
    trans_root: Path = args.transcriptions_root or (audio_root / "transcriptions")
    device = pick_device(args.device)

    if not audio_root.exists():
        print(f"❌ Audio root not found: {audio_root}", file=sys.stderr)
        sys.exit(1)
    trans_root.mkdir(parents=True, exist_ok=True)

    files = list_audio_files(audio_root)
    if not files:
        print("ℹ️ No .wav/.mp3 files found.")
        return

    print(f"🔎 Found {len(files)} audio files under {audio_root}")
    print(f"🧠 Model: {args.model}  |  🖥️ Device: {device}")
    print(f"📁 Outputs: {trans_root}")

    model = load_model(args.model, device)

    errors = 0
    for audio_path in tqdm(files, desc="Transcribing", unit="file"):
        dump_path, csv_path = out_paths(audio_path, audio_root, trans_root, args.model)

        if not args.overwrite and already_done(dump_path, csv_path):
            continue

        try:
            result = transcribe_file(
                model=model,
                audio_path=audio_path,
                device=device,
                language=args.language,
                temperature=args.temperature,
                beam_size=args.beam_size,
            )
            # Write outputs that match your existing pattern
            save_raw_result(result, dump_path)               # JSON content → *_<model>_transcription.txt
            save_transcription_segments_to_csv(result, csv_path)
        except KeyboardInterrupt:
            print("\n⏹️ Interrupted by user.")
            sys.exit(130)
        except Exception as e:
            errors += 1
            print(f"❌ Error on {audio_path}: {e}", file=sys.stderr)
            if args.fail_fast:
                sys.exit(3)

    if errors:
        print(f"✅ Done with {errors} error(s).")
    else:
        print("✅ Done.")


if __name__ == "__main__":
    main()

