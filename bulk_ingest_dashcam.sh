#!/usr/bin/env bash
set -euo pipefail

# Bulk Ingest Script for Dashcam Data
# Processes all dashcam data from NAS2 into Neo4j

LOG_DIR="${LOG_DIR:-./logs}"
mkdir -p "$LOG_DIR"
STAMP="$(date +'%Y%m%d_%H%M%S')"
LOG_FILE="$LOG_DIR/bulk_ingest_$STAMP.log"

echo "=== Bulk Dashcam Ingest Starting @ $(date) ===" | tee -a "$LOG_FILE"
echo "Logging to: $LOG_FILE" | tee -a "$LOG_FILE"

# Configuration
DASHCAM_ROOT="${DASHCAM_ROOT:-/media/scott/NAS2/fileserver/dashcam}"
NEO4J_URI="${NEO4J_URI:-bolt://host.docker.internal:7687}"
NEO4J_USER="${NEO4J_USER:-neo4j}"
NEO4J_PASS="${NEO4J_PASS:-knowledge_graph_2026}"

# Years and months to process (can be overridden)
YEARS_TO_PROCESS="${YEARS_TO_PROCESS:-2023 2024 2025 2026}"

echo "Processing years: $YEARS_TO_PROCESS" | tee -a "$LOG_FILE"
echo "Dashcam root: $DASHCAM_ROOT" | tee -a "$LOG_FILE"

# Function to process a single day
process_day() {
    local year=$1
    local month=$2
    local day=$3
    local base_path="${DASHCAM_ROOT}/${year}/${month}/${day}"
    
    if [[ ! -d "$base_path" ]]; then
        echo "[SKIP] Directory not found: $base_path" | tee -a "$LOG_FILE"
        return 0
    fi
    
    # Count metadata files
    local clip_count=$(find "$base_path" -name '*_metadata.csv' 2>/dev/null | wc -l)
    if [[ $clip_count -eq 0 ]]; then
        echo "[SKIP] No clips found in: $base_path" | tee -a "$LOG_FILE"
        return 0
    fi
    
    echo "[RUN] Processing ${year}/${month}/${day} (${clip_count} clips)" | tee -a "$LOG_FILE"
    
    # Run the ingest script for this day
    docker compose run --rm --no-deps -v /media/scott/NAS2/fileserver/dashcam:/dashcam ingest-service \
        bash -lc "python /app/dashcam_yolo_embeddings.py \\
            --bases /dashcam/${year}/${month}/${day} \\
            --neo4j-uri ${NEO4J_URI} \\
            --neo4j-user ${NEO4J_USER} \\
            --neo4j-pass ${NEO4J_PASS} \\
            --resume" 2>&1 | tee -a "$LOG_FILE"
    
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        echo "[ERROR] Failed to process ${year}/${month}/${day} (exit code: $exit_code)" | tee -a "$LOG_FILE"
        return $exit_code
    fi
    
    echo "[DONE] Completed ${year}/${month}/${day}" | tee -a "$LOG_FILE"
    return 0
}

# Main processing loop
total_days=0
processed_days=0
failed_days=0

for year in $YEARS_TO_PROCESS; do
    year_path="${DASHCAM_ROOT}/${year}"
    if [[ ! -d "$year_path" ]]; then
        echo "[SKIP] Year directory not found: $year_path" | tee -a "$LOG_FILE"
        continue
    fi
    
    for month in $(ls -1 "$year_path" 2>/dev/null | sort); do
        month_path="${year_path}/${month}"
        if [[ ! -d "$month_path" ]]; then
            continue
        fi
        
        for day in $(ls -1 "$month_path" 2>/dev/null | sort); do
            ((total_days++)) || true
            
            process_day "$year" "$month" "$day"
            if [[ $? -eq 0 ]]; then
                ((processed_days++)) || true
            else
                ((failed_days++)) || true
            fi
        done
    done
done

echo "" | tee -a "$LOG_FILE"
echo "=== Bulk Ingest Summary ===" | tee -a "$LOG_FILE"
echo "Total days found: $total_days" | tee -a "$LOG_FILE"
echo "Successfully processed: $processed_days" | tee -a "$LOG_FILE"
echo "Failed: $failed_days" | tee -a "$LOG_FILE"
echo "=== Bulk Ingest Finished @ $(date) ===" | tee -a "$LOG_FILE"

exit 0
