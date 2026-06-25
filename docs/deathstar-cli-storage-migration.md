# Deathstar CLI + storage migration notes

Updated: 2026-06-25

## Goal

Use `auto-ingest` as the canonical repo for dashcam/audio/transcript processing and treat `video-automation` as a deprecated migration source. The same command names should work on every fleet machine, with paths resolved by `auto_ingest_config.py` and environment overrides.

## Canonical storage layout

- Dashcam MP4 source of truth: `/media/scott/NAS3/dashcam`
- Audio/transcription hot cache: `/media/scott/SSD_4TB/audio`
- Optional slow/large processing scratch: `/mnt/8TB_2025` or `/mnt/8TBHDD/fileserver`
- Legacy/general fileserver root: `/media/scott/NAS3/fileserver`

Environment variables can override config on any machine:

```bash
export DASHCAM_ROOT=/media/scott/NAS3/dashcam
export AUDIO_ROOT=/media/scott/SSD_4TB/audio
export FILESERVER_ROOT=/media/scott/NAS3/fileserver
export SCRATCH_ROOT=/mnt/8TB_2025
export NEO4J_URI=bolt://100.64.43.123:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=replace-with-local-secret
```

## CLI aliases for ~/.bash_aliases

Use these on deathstar and adapt only the path mount points when needed:

```bash
alias fs="cd /media/scott/NAS3;ls;"
alias dashcam="cd /media/scott/NAS3/dashcam;ls;"
alias audio="cd /media/scott/SSD_4TB/audio;ls;"

alias yolo="cd ~/git/auto-ingest;./.venv/bin/python3 yolo_vehicle_detction.py;"
alias meta="cd ~/git/auto-ingest;./.venv/bin/python3 metadata_scraper_iterator.py;"
alias merge="cd ~/git/auto-ingest;./.venv/bin/python3 dashcam_merge_FR.py --base \"${DASHCAM_ROOT:-/media/scott/NAS3/dashcam}\";"
alias heatmap="cd ~/git/auto-ingest;./.venv/bin/python3 yolo_heatmap_iterator.py;"
alias transcribe="cd ~/git/auto-ingest;./.venv/bin/python3 whisper_all.py;"
alias extractaudio="cd ~/git/auto-ingest;./.venv/bin/python3 extract_audio.py;"
alias speaker="cd ~/git/auto-ingest;./.venv/bin/python3 speakers.py 2> /dev/null;"
alias embeddings="cd ~/git/auto-ingest;./.venv/bin/python3 process_text.py;"
alias music="cd ~/git/auto-ingest;./.venv/bin/python3 01_precompute_music_segments.py --push-neo4j;"
alias classifymusic="cd ~/git/auto-ingest;./.venv/bin/python3 02_classify_lyrics.py --segments-source sidecar --limit 5000;"
alias venv="source ~/git/auto-ingest/.venv/bin/activate;"
alias run="cd ~/git/auto-ingest && ./runall.sh"
alias runingest="cd ~/git/auto-ingest && ./run_ingest_all.sh"
alias vs="/home/deathstar/git/auto-ingest/vector_search.sh"
alias vecsearch="/home/deathstar/git/auto-ingest/vector_search.sh"
alias vsm="/home/deathstar/git/auto-ingest/vector_search_menu.sh"
```

Do not put Hugging Face tokens in `~/.bash_aliases`, git history, or command examples. Use:

```bash
mkdir -p ~/.config/auto-ingest
chmod 700 ~/.config/auto-ingest
cat > ~/.config/auto-ingest/hf.env <<'EOF'
export HUGGINGFACE_TOKEN=replace-with-real-token
EOF
chmod 600 ~/.config/auto-ingest/hf.env
```

Then use:

```bash
source ~/.config/auto-ingest/hf.env
```

## Main commands

Copy dashcam footage from mounted USB/SD card into canonical NAS3 layout:

```bash
cd ~/git/auto-ingest
./dashcam_copy.sh
```

Extract MP3s from NAS3 dashcam MP4s into SSD_4TB audio cache:

```bash
cd ~/git/auto-ingest
./.venv/bin/python3 extract_audio.py --dry-run --limit 20
./.venv/bin/python3 extract_audio.py
```

Run transcript/RTTM ingest against Neo4j:

```bash
cd ~/git/auto-ingest
DRY_RUN=1 ./run_ingest_all.sh
./run_ingest_all.sh
```

Run the full local pipeline:

```bash
cd ~/git/auto-ingest
./runall.sh
```

## Current video format support

The new dashcam format is not just `_F.MP4` or `_R.MP4`; it may be numeric-suffixed:

- `2026_0619_204141_000001F.MP4`
- `2026_0619_204141_000002R.MP4`

Use case-insensitive regex patterns:

- front: `_\d+F\.mp4$`
- rear: `_\d+R\.mp4$`
- either: `_\d+[FR]\.mp4$`

The new `extract_audio.py` preserves the original stem, so front/rear/merged files do not overwrite each other in the audio cache.

## Migration status from video-automation

Moved or pointed to auto-ingest:

- `run` alias now points to `~/git/auto-ingest/runall.sh`
- `extractaudio` points to `~/git/auto-ingest/extract_audio.py`
- YOLO, metadata, speaker, embeddings, music, vector-search aliases point to auto-ingest
- `run_ingest_all.sh` resolves `DASHCAM_ROOT` and `AUDIO_ROOT` from config/env
- bash copy scripts now call `auto_ingest_config.py` instead of containing invalid `get_fileserver_path(...)` shell syntax

Still to consolidate from `video-automation`:

- `shorts_builder.py`, `generate_shorts.py`, `v8.py`, `v9.py`
- phone-format front/rear/FR shorts generation
- spoken-word sentence/word highlighting with themed title at top
- MP4 compression workflow and slow-drive scratch strategy

Recommended destination layout:

```text
auto-ingest/
  extract_audio.py
  dashcam_merge_FR.py
  compress_dashcam.py
  shorts_builder.py              # migrate from video-automation once cleaned
  docs/deathstar-cli-storage-migration.md
```

## Git push-protection recovery

If GitHub rejects push because old local commits contain a token, do not bypass protection. Make a local backup branch, reset to clean `origin/main`, then replay sanitized file changes only:

```bash
cd ~/git/auto-ingest
git branch backup/deathstar-contaminated-history-$(date +%Y%m%d-%H%M%S)
git diff origin/main..HEAD > /tmp/auto-ingest-local-sanitized.patch
# inspect /tmp/auto-ingest-local-sanitized.patch before applying
# then reset/replay only clean changes
```

Rotate any token that appeared in shell output or git history.
