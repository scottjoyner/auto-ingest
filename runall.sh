#!/bin/bash
set -euo pipefail

cd ~/git/auto-ingest || { echo "❌ Failed to cd into ~/git/auto-ingest"; exit 1; }

echo "▶️ Copy: audio / dashcam / bodycam"
./audio_copy.sh        || echo "❌ Failed: audio_copy.sh"
./dashcam_copy.sh      || echo "❌ Failed: dashcam_copy.sh"
./bodycam_copy.sh      || echo "❌ Failed: bodycam_copy.sh"

echo "▶️ Whisper (chunked, merged, large-v3) over AUDIO tree"
# ./.venv/bin/python3 whisper_audio_chunked.py \
#   --audio-root FILESERVER_ROOT/audio \
#   --fast-copy \
#   --merge \
#   --model large-v3 || echo "❌ Failed: whisper_audio_chunked.py"
./.venv/bin/python3 whisper_audio_chunked.py \
  --model medium \
  --merge \
  --audio-root "${AUDIO_ROOT:-$(python3 - <<'PY'
from auto_ingest_config import get_audio_root
print(get_audio_root())
PY
)}/audio" \
  --transcriptions-root "${AUDIO_ROOT:-$(python3 - <<'PY'
from auto_ingest_config import get_audio_root
print(get_audio_root())
PY
)}/audio/transcriptions"

echo "▶️ Speaker diarization (dashcam videos + standalone audio)"
./.venv/bin/python3 speakers.py || echo "❌ Failed: speakers.py"

echo "▶️ Ingest (transcripts + RTTM → Neo4j)"
./.venv/bin/python3 ingest_transcriptions.py || echo "❌ Failed: ingest_transcriptions.py"

echo "▶️ Ingest speakers_reconcile (transcripts + RTTM → Neo4j)"
./.venv/bin/python3 speakers_reconcile.py \
  --neo4j-uri bolt://localhost:7687 \
  --neo4j-user neo4j \
  --neo4j-password livelongandprosper \
  --db neo4j \
  --batch 50 \
  --only-missing  || echo "❌ Failed: speakers_reconcile.py"

# Now the rest of your vision/metadata pipeline—these do NOT need to precede ingest
echo "▶️ YOLO vehicle detection"
./.venv/bin/python3 yolo_vehicle_detction.py || echo "❌ Failed: yolo_vehicle_detction.py"

echo "▶️ Dashcam HUD metadata"
./.venv/bin/python3 metadata_scraper_iterator.py || echo "❌ Failed: metadata_scraper_iterator.py"
# ./.venv/bin/python3 dashcam_hud_iterate.py --base "${DASHCAM_ROOT:-$(python3 - <<'PY'
from auto_ingest_config import get_dashcam_root
print(get_dashcam_root())
PY
)}" || echo "❌ Failed: dashcam_hud_iterator.py"
# ./.venv/bin/python3 dashcam_hud_iterate.py --base "${DASHCAM_ROOT:-$(python3 - <<'PY'
from auto_ingest_config import get_dashcam_root
print(get_dashcam_root())
PY
)}" || echo "❌ Failed: dashcam_hud_iterator.py"

echo "▶️ Music precompute + lyrics classifier (optional)"
./.venv/bin/python3 01_precompute_music_segments.py --push-neo4j || echo "❌ Failed: 01_precompute_music_segments.py"
./.venv/bin/python3 02_classify_lyrics.py --segments-source neo4j --limit 2000 || echo "❌ Failed: 02_classify_lyrics.py"

# === Link local Speakers to GlobalSpeaker identities ===
echo "🔗 Linking speakers globally…"

./.venv/bin/python3 link_global_speakers.py \
  --global-prefilter --global-thresh 0.78 --global-k 8 --global-index hnsw --global-m 32 --global-ef 128 \
  --skip-already-linked \
  --faiss-prefilter --faiss-k 64 --faiss-index hnsw --faiss-m 32 --faiss-ef 128 \
  --source-level auto \
  --min-seg 0.7 --max-snips 8 --max-per-file 3 --snip-len 1.6 \
  --min-proportion 0.5 --min-rms 0.005 --min-snr-db 6.0 \
  --thresh 0.72 \
  --holdout --holdout-min 0.62 --holdout-action drop-members \
  --audio-cache ./audio_path_cache.json \
  --emb-cache ./emb_cache.sqlite || echo "❌ Failed: link_global_speakers.py"

# OLD
# ./.venv/bin/python3 link_global_speakers.py \
#   --min-seg 0.7 \
#   --max-snips 8 \
#   --max-per-file 3 \
#   --snip-len 1.6 \
#   --min-proportion 0.5 \
#   --min-rms 0.005 \
#   --min-snr-db 6.0 \
#   --thresh 0.72 \
#   --holdout --holdout-min 0.62 --holdout-action drop-members \
#   --audio-cache ./audio_path_cache.json \
#   --emb-cache ./emb_cache.sqlite || echo "❌ Failed: link_global_speakers.py"

# echo "🔗 Linking speakers globally…" OLD BUT STILL MIGHT WORK NEEVER TESTED PROPERLY
# ./.venv/bin/python3 speaker_linker.py \
#   --min-utt 0.7 \
#   --max-snips 8 \
#   --pad 0.25 \
#   --thresh 0.74 2> /dev/null || echo "❌ Failed: dashcam_yolo_embeddings.py"

# echo "▶️ YOLO heatmaps"
# ./.venv/bin/python3 yolo_heatmap_iterator.py || echo "❌ Failed: yolo_heatmap_iterator.py"

echo "▶️ Dashcam YOLO embeddings"
./.venv/bin/python3 dashcam_yolo_embeddings.py \
  --resume \
  --grid 16x9 \
  --pyramid \
  --heatmap \
  --neo4j-uri bolt://localhost:7687 \
  --neo4j-user neo4j \
  --neo4j-pass livelongandprosper \
  --win-mins 10 2> /dev/null || echo "❌ Failed: dashcam_yolo_embeddings.py"

echo "▶️ Patch Missing Locations in YOLO Embeddings"
./.venv/bin/python3 patch_missing_locations.py \
  --neo4j-uri bolt://localhost:7687 \
  --neo4j-user neo4j \
  --neo4j-pass livelongandprosper \
  --key-limit 1000 \
  --win-mins 10 \
  --validate-m 50
echo "▶️ Dashcam Merger"
./.venv/bin/python3 dashcam_merge_FR.py \
  --base "${DASHCAM_ROOT:-$(python3 - <<'PY'
from auto_ingest_config import get_dashcam_root
print(get_dashcam_root())
PY
)}" \
  --base "${DASHCAM_ROOT:-$(python3 - <<'PY'
from auto_ingest_config import get_dashcam_root
print(get_dashcam_root())
PY
)}"  || echo "❌ Failed: dashcam_merge_FR.py"

# echo "▶️ Iterator" (OLD)
# ./.venv/bin/python3 iterator.py || echo "❌ Failed: iterator.py"

echo "▶️ timelapse_from_fr"
./.venv/bin/python3 timelapse_from_fr.py "${DASHCAM_ROOT:-$(python3 - <<'PY'
from auto_ingest_config import get_dashcam_root
print(get_dashcam_root())
PY
)}" --recursive || echo "❌ Failed: timelapse_from_fr.py"

echo "▶️ timelapse_from_fr"
./.venv/bin/python3 shorts_builder.py --base "${DASHCAM_ROOT:-$(python3 - <<'PY'
from auto_ingest_config import get_dashcam_root
print(get_dashcam_root())
PY
)}" --profiles clean karaoke wordgrid || echo "❌ Failed: shorts_builder.py"

# echo "▶️ timelapse_from_fr"
# ./.venv/bin/python3 shorts_builder.py \
#   --base FILESERVER_ROOT/dashcam \
#   --profiles-file profiles.json \
#   --profiles clean karaoke pop_neon cinematic minimal_lower_third meme_bold highlighter tech_hud wordgrid emoji_style || echo "❌ Failed: shorts_builder.py"

echo "✅ All scripts finished (some may have failed)."  



# === Link local Speakers to GlobalSpeaker identities ===
# More agressive clustering using pyannote 
# echo "🔗 Linking speakers globally…"
# ./.venv/bin/python3 link_global_speakers.py \
#   --backend pyannote \
#   --thresh 0.68 \
#   --holdout --holdout-min 0.62 --holdout-action drop-members
