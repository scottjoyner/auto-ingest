#!/bin/bash
export NEO4J_PASSWORD='***'
cd ~/git/auto-ingest
python3 -u link_global_speakers_2.py --max-speakers 50 --dry-run --global-prefilter --faiss-prefilter
