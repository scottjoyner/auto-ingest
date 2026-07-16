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

# ./.venv/bin/python3 whisper_audio_chunkedv1.py \ [OLD VERSION]
#   --model medium \
#   --merge \
#   --audio-root FILESERVER_ROOT/audio \
#   --transcriptions-root FILESERVER_ROOT/audio/transcriptions

# ./.venv/bin/python3 whisper_audio_chunked.py \
#   --model medium \
#   --merge \
#   --audio-root FILESERVER_ROOT/audio \
#   --transcriptions-root FILESERVER_ROOT/audio \
#   --chunks-root /tmp/chunks \
#   --pcm-wav \
#   --chunk-len 90 --stride 85 \
#   --delete-chunks-after \
#   --log-level INFO  || echo "❌ Failed: whisper_audio_chunked.py"

DO_TRANSCRIBE=1               # 0 to skip transcription
WHISPER_MODEL=medium            # tiny|base|small|medium|large
WHISPER_LANG=en
echo "▶️ Speaker diarization (dashcam videos + standalone audio)"
./.venv/bin/python3 speakers.py || echo "❌ Failed: speakers.py"

# OLD 
# echo "▶️ Ingest (transcripts + RTTM → Neo4j)"
# ./.venv/bin/python3 auto_ingest/ingest/transcripts.py || echo "❌ Failed: auto_ingest/ingest/transcripts.py"

echo "▶️ Ingest (transcripts + RTTM → Neo4j)"
# ./.venv/bin/python3 ingest_transcriptsv5.py \
#   --batch-size 100 \
#   --transcript-emb-v2 || echo "❌ Failed: ingest_transcripts.py"
./run_ingest_all.sh || echo "❌ Failed: ingest_transcripts.py"

echo "▶️ Ingest speakers_reconcile (transcripts + RTTM → Neo4j)"
./.venv/bin/python3 speakers_reconcile.py \
  --neo4j-uri bolt://localhost:7687 \
  --neo4j-user neo4j \
  --neo4j-password knowledge_graph_2026 \
  --db neo4j \
  --batch 50 \
  --only-missing \
  --allow-discovery  || echo "❌ Failed: speakers_reconcile.py"

# Now the rest of your vision/metadata pipeline—these do NOT need to precede ingest
echo "▶️ YOLO vehicle detection"
./.venv/bin/python3 yolo_vehicle_detction.py || echo "❌ Failed: yolo_vehicle_detction.py"

echo "▶️ Dashcam HUD metadata"
./.venv/bin/python3 metadata_scraper_iterator.py || echo "❌ Failed: metadata_scraper_iterator.py"
# ./.venv/bin/python3 dashcam_hud_iterate.py --base FILESERVER_ROOT/dashcam || echo "❌ Failed: dashcam_hud_iterator.py"
# ./.venv/bin/python3 dashcam_hud_iterate.py --base FILESERVER_ROOT/dashcam || echo "❌ Failed: dashcam_hud_iterator.py"

echo "▶️ Music precompute + lyrics classifier (optional)"
./.venv/bin/python3 01_precompute_music_segments.py --push-neo4j || echo "❌ Failed: 01_precompute_music_segments.py"
./.venv/bin/python3 02_classify_lyrics.py --segments-source neo4j --limit 50000 --model roberta-large-mnli --hf-token "" || echo "❌ Failed: 02_classify_lyrics.py"

# === Link local Speakers to GlobalSpeaker identities ===
echo "🔗 Linking speakers globally…"
python3 link_global_speakers.py \
  --speaker-batch 900 --items-per-speaker 32 \
  --global-prefilter --global-k 8 --global-m 20 --global-ef 64 \
  --skip-already-linked \
  --source-level auto \
  --min-seg 0.6 --max-snips 8 --max-per-file 3 --snip-len 1.6 \
  --min-proportion 0.4 --min-rms 0.004 --min-snr-db 5.0 \
  --thresh 0.68 \
  --holdout --holdout-min 0.58 --holdout-action drop-members \
  --quarantine-min 0.72 \
  --priority-name "Scott|Kipnerter|Kipnerter Scott" --priority-thresh 0.82 \
  --priority-max-attach 2000 \
  --rank-and-label \
  --audio-cache ./audio_path_cache.json --emb-cache ./emb_cache.sqlite

# ./.venv/bin/python3 link_global_speakers.py \
#   --global-prefilter \
#   --global-thresh 0.74 --global-k 8 --global-index hnsw --global-m 24 --global-ef 128 \
#   --global-include-tentative \
#   --skip-already-linked \
#   --faiss-prefilter --faiss-k 128 --faiss-index hnsw --faiss-m 48 --faiss-ef 256 \
#   --source-level auto \
#   --min-seg 0.6 --max-snips 12 --max-per-file 4 --snip-len 2.0 \
#   --min-proportion 0.4 --min-rms 0.004 --min-snr-db 5.0 \
#   --thresh 0.68 \
#   --holdout --holdout-min 0.58 --holdout-action drop-members \
#   --quarantine-min 0.72 \
#   --priority-name "Scott|Kipnerter|Kipnerter Scott" \
#   --priority-thresh 0.82 \
#   --priority-max-attach 2000 \
#   --rank-and-label \
#   --audio-cache ./audio_path_cache.json \
#   --emb-cache ./emb_cache.sqlite || echo "❌ Failed: link_global_speakers.py"

# Less agressive version
# ./.venv/bin/python3 link_global_speakers old.py \
#   --global-prefilter --global-thresh 0.78 --global-k 8 --global-index hnsw --global-m 32 --global-ef 128 \
#   --skip-already-linked \
#   --faiss-prefilter --faiss-k 64 --faiss-index hnsw --faiss-m 32 --faiss-ef 128 \
#   --source-level auto \
#   --min-seg 0.7 --max-snips 8 --max-per-file 3 --snip-len 1.6 \
#   --min-proportion 0.5 --min-rms 0.005 --min-snr-db 6.0 \
#   --thresh 0.72 \
#   --holdout --holdout-min 0.62 --holdout-action drop-members \
#   --audio-cache ./audio_path_cache.json \
#   --emb-cache ./emb_cache.sqlite || echo "❌ Failed: link_global_speakers.py"

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
#   --thresh 0.74 2> /dev/null || echo "❌ Failed: auto_ingest/dashcam/yolo_embeddings.py"

# echo "▶️ YOLO heatmaps"
# ./.venv/bin/python3 yolo_heatmap_iterator.py || echo "❌ Failed: yolo_heatmap_iterator.py"

echo "▶️ Dashcam YOLO embeddings"
./.venv/bin/python3 auto_ingest/dashcam/yolo_embeddings.py \
  --resume \
  --grid 16x9 \
  --pyramid \
  --heatmap \
  --repair-missing-moov \
  --neo4j-uri bolt://localhost:7687 \
  --neo4j-user neo4j \
  --neo4j-pass knowledge_graph_2026 \
  --win-mins 10 2> /dev/null || echo "❌ Failed: auto_ingest/dashcam/yolo_embeddings.py"

echo "▶️ Patch Missing Locations in YOLO Embeddings"
./.venv/bin/python3 patch_missing_locations.py \
  --neo4j-uri bolt://localhost:7687 \
  --neo4j-user neo4j \
  --neo4j-pass knowledge_graph_2026 \
  --key-limit 1000 \
  --win-mins 10 \
  --validate-m 50 \
  --tx-timeout 45 \
  --max-retries 5 \
  --chunk 500


# echo "▶️ Dashcam Merger" (Not working yet)
# ./.venv/bin/python3 dashcam_merge_FR.py \
#   --base FILESERVER_ROOT/dashcam \
#   --base FILESERVER_ROOT/dashcam  || echo "❌ Failed: dashcam_merge_FR.py"

echo "▶️ Iterator" # (OLD)
./.venv/bin/python3 iterator.py || echo "❌ Failed: iterator.py"

echo "▶️ ingest_sidecar_summaries.py"
./.venv/bin/python3 ingest_sidecar_summaries.py \
  --roots FILESERVER_ROOT/audio \
  --ensure-schema \
  --create-missing-transcriptions

OLLAMA_MODEL=gemma2:2b OLLAMA_USE_CLI=1 OLLAMA_TIMEOUT=900 python summarize_from_segments_v2.py --all-missing --write-notes --include-utterances

echo "▶️ timelapse_from_fr"
./.venv/bin/python3 timelapse_from_fr.py FILESERVER_ROOT/dashcam --recursive || echo "❌ Failed: timelapse_from_fr.py"

echo "▶️ timelapse_from_fr"
./.venv/bin/python3 shorts_builder.py --base FILESERVER_ROOT/dashcam --profiles clean karaoke wordgrid || echo "❌ Failed: shorts_builder.py"

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