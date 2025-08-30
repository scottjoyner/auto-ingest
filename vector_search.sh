#!/bin/bash

# Path setup
PYTHON_EXEC="/home/deathstar/git/video-automation/.venv/bin/python"
SCRIPT_PATH="/home/deathstar/git/video-automation/ingest_transcriptions.py"

# Prompt for search query
read -p "🔍 Enter search query (default: none): " SEARCH
SEARCH_ARG=""
if [ -n "$SEARCH" ]; then
    SEARCH_ARG="--search \"$SEARCH\""
fi

# Prompt for target
read -p "🎯 Enter target [utterance | segment | transcription] (default: transcription): " TARGET
TARGET=${TARGET:-transcription}

# Prompt for top-k results
read -p "🔢 Top-K results (default: 10): " TOPK
TOPK=${TOPK:-10}

# Optional flags
read -p "📦 Include node embeddings in output? (y/N): " INCLUDE_EMB
INCLUDE_EMB_FLAG=""
if [[ "$INCLUDE_EMB" =~ ^[Yy]$ ]]; then
    INCLUDE_EMB_FLAG="--include-emb"
fi

# Optional: Add --win-mins input (advanced)
read -p "⏱ Window minutes to snap to nearest PhoneLog (default: 10): " WIN_MINS
WIN_MINS=${WIN_MINS:-10}

# Execute the command, hiding stderr (logs/errors)
eval $PYTHON_EXEC "$SCRIPT_PATH" \
    $SEARCH_ARG \
    --target "$TARGET" \
    --topk "$TOPK" \
    --win-mins "$WIN_MINS" \
    $INCLUDE_EMB_FLAG \
    2>/dev/null
