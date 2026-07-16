#!/bin/bash
set -euo pipefail

cd ~/git/auto-ingest || { echo "❌ Failed to cd into ~/git/auto-ingest"; exit 1; }

# Load host-specific roots and service endpoints from .env. These resolve via the
# mount-aware helpers in auto_ingest_config (get_dashcam_root/get_audio_root/
# get_bodycam_root/get_fileserver_root), so no machine-specific paths are hardcoded here.
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

NEO4J_URI="${NEO4J_URI:-bolt://100.64.43.123:7687}"
NEO4J_USER="${NEO4J_USER:-neo4j}"
# Password precedence: NEO4J_PASSWORD -> NEO4J_PASS -> NEO4J_PASSWORD_DEFAULT ->
# baked historical default. Set NEO4J_PASSWORD_DEFAULT once in .env to change it.
NEO4J_PASSWORD="${NEO4J_PASSWORD:-${NEO4J_PASS:-${NEO4J_PASSWORD_DEFAULT:-knowledge_graph_2026}}}"
NEO4J_PASS="$NEO4J_PASSWORD"
# Avoid TensorFlow/Keras import path in transformers/sentence-transformers;
# this venv is torch-first and Keras 3 breaks older TF integrations.
USE_TF="${USE_TF:-0}"
TRANSFORMERS_NO_TF="${TRANSFORMERS_NO_TF:-1}"
export NEO4J_URI NEO4J_USER NEO4J_PASSWORD NEO4J_PASS USE_TF TRANSFORMERS_NO_TF

echo "▶️ Copy: audio / dashcam / bodycam"
copy_failed=0
./audio_copy.sh || { echo "❌ Failed: audio_copy.sh"; copy_failed=1; }
./dashcam_copy.sh || { echo "❌ Failed: dashcam_copy.sh"; copy_failed=1; }
./bodycam_copy.sh || { echo "❌ Failed: bodycam_copy.sh"; copy_failed=1; }
if [ "$copy_failed" -ne 0 ]; then
  echo "❌ Aborting before Whisper because one or more required copy stages failed."
  echo "   Fix the mount/path error above, then rerun."
  exit 1
fi

echo "▶️ faster-whisper (chunked, merged, medium/int8 CPU by default) over AUDIO tree"
# ./.venv/bin/python3 whisper_audio_chunked.py \
#   --audio-root FILESERVER_ROOT/audio \
#   --fast-copy \
#   --merge \
#   --model large-v3 || echo "❌ Failed: whisper_audio_chunked.py"
./.venv/bin/python3 whisper_audio_chunked.py \
  --backend faster \
  --compute-type "${WHISPER_COMPUTE_TYPE:-int8}" \
  --model "${WHISPER_MODEL:-medium}" \
  --merge \
  --audio-root "${AUDIO_ROOT:-$(python3 - <<'PY'
from auto_ingest_config import get_audio_root
print(get_audio_root())
PY
)}" \
  --transcriptions-root "${TRANSCRIPTIONS_ROOT:-$(python3 - <<'PY'
from pathlib import Path
from auto_ingest_config import get_audio_root
print(Path(get_audio_root()) / 'transcriptions')
PY
)}"

echo "▶️ Speaker diarization (dashcam videos + standalone audio)"
./.venv/bin/python3 speakers.py || echo "❌ Failed: speakers.py"

echo "▶️ Ingest (transcripts + RTTM → Neo4j)"
./.venv/bin/python3 ingest_transcriptions.py || echo "❌ Failed: ingest_transcriptions.py"

echo "▶️ Ingest speakers_reconcile (transcripts + RTTM → Neo4j)"
./.venv/bin/python3 speakers_reconcile.py \
  --neo4j-uri "$NEO4J_URI" \
  --neo4j-user "$NEO4J_USER" \
  --neo4j-password "$NEO4J_PASSWORD" \
  --db neo4j \
  --batch 50 \
  --only-missing  || echo "❌ Failed: speakers_reconcile.py"

# Now the rest of your vision/metadata pipeline—these do NOT need to precede ingest
echo "▶️ YOLO vehicle detection"
./.venv/bin/python3 yolo_vehicle_detction.py || echo "❌ Failed: yolo_vehicle_detction.py"

echo "▶️ Dashcam HUD metadata"
./.venv/bin/python3 metadata_scraper_iterator.py || echo "❌ Failed: metadata_scraper_iterator.py"
# Example if needed:
# ./.venv/bin/python3 dashcam_hud_iterate.py --base "$DASHCAM_ROOT" || echo "❌ Failed: dashcam_hud_iterator.py"

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
  --neo4j-uri "$NEO4J_URI" \
  --neo4j-user "$NEO4J_USER" \
  --neo4j-pass "$NEO4J_PASSWORD" \
  --win-mins 10 2> /dev/null || echo "❌ Failed: dashcam_yolo_embeddings.py"

echo "▶️ Patch Missing Locations in YOLO Embeddings"
if [ -f patch_missing_locations.py ]; then
  ./.venv/bin/python3 patch_missing_locations.py \
    --neo4j-uri "$NEO4J_URI" \
    --neo4j-user "$NEO4J_USER" \
    --neo4j-pass "$NEO4J_PASSWORD" \
    --key-limit 1000 \
    --win-mins 10 \
    --validate-m 50 || echo "❌ Failed: patch_missing_locations.py"
else
  echo "⚠️ Skipping patch_missing_locations.py: file not present in this checkout."
fi
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
