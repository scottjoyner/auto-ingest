#!/usr/bin/env bash
set -euo pipefail

DROP_ROOT="${DROP_ROOT:-/nas/drop}"
JOB_KIND="${1:-all}"
STAMP="$(date +%Y%m%d_%H%M%S)"
JOB_FILE="$DROP_ROOT/${STAMP}_${JOB_KIND}.job"

mkdir -p "$DROP_ROOT"

case "$JOB_KIND" in
  audio)
    cmd='cd /app && SCAN_ROOTS="${AUDIO_ROOT:-/nas/S/audio},${TRANSCRIPT_ROOT:-/nas/fileserver/audio/transcriptions}" /usr/bin/env bash run_ingest_all.sh'
    ;;
  dashcam)
    cmd='cd /app && SCAN_ROOTS="${DASHCAM_ROOT:-/nas/fileserver/dashcam},${DASHCAM_ROOT:-/nas/fileserver/dashcam}/transcriptions" DASHCAM_ROOT="${DASHCAM_ROOT:-/nas/fileserver/dashcam}" /usr/bin/env bash run_ingest_all.sh'
    ;;
  bodycam)
    cmd='cd /app && SCAN_ROOTS="${BODYCAM_ROOT:-/nas/fileserver/bodycam}" /usr/bin/env bash run_ingest_all.sh'
    ;;
  all)
    cmd='cd /app && /usr/bin/env bash run_ingest_all.sh'
    ;;
  *)
    echo "Usage: $0 {audio|dashcam|bodycam|all}" >&2
    exit 2
    ;;
esac

cat > "$JOB_FILE" <<JOB
#!/usr/bin/env bash
set -euo pipefail
$cmd
JOB
chmod +x "$JOB_FILE"
echo "Created job: $JOB_FILE"

# Best-effort: also create the queryable IngestJob manifest node so the claim
# protocol (G4) has something to claim. A down Neo4j must not block job creation.
KEY="$(basename "$JOB_FILE" .job)"
"$(dirname "$0")/../scripts/claim_job.py" create "$KEY" \
  || echo "manifest create skipped (neo4j unreachable)" >&2
