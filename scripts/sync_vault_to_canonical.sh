#!/bin/bash
# sync_vault_to_canonical.sh — THE single writer to the canonical vault (G2).
#
# CANONICAL MODEL (see docs/VAULT_SYNC.md):
#   * `canonical_vault_path` in config.yaml (currently /media/scott/NAS5/
#     shared-knowledge) is the ONE AND ONLY writer of the shared vault.
#   * All other hosts READ from the canonical mirror; they never SSH-push
#     their local clones to each other (that was the old, conflict-prone
#     4-hardcoded-IP design).
#
# This script:
#   1. Takes the LOCAL clone (already merged by knowledge_sync_all.sh) and
#      rsyncs it into the canonical mirror. That is the only write to mirror.
#   2. Records a sync marker so peers can detect freshness.
#
# It does NOT pull from peers, does NOT push to peers. Those jobs belong to
# knowledge_sync_all.sh (which calls this after merging). Keep a single
# writer to avoid divergent mirrors.

set -u

CONFIG_FILE="${CONFIG_FILE:-config.yaml}"
LOG_FILE="${LOG_FILE:-/home/scott/logs/knowledge_sync.log}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

_canonical() {
    if command -v yq >/dev/null 2>&1; then
        yq eval "['knowledge_map']['canonical_vault_path']" "$CONFIG_FILE" 2>/dev/null
    else
        python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG_FILE')); print(c['knowledge_map']['canonical_vault_path'])" 2>/dev/null
    fi
}
_rsync_opts() {
    if command -v yq >/dev/null 2>&1; then
        yq eval "['knowledge_map']['vault_sync']['rsync_opts']" "$CONFIG_FILE" 2>/dev/null
    else
        python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG_FILE')); print(c['knowledge_map']['vault_sync']['rsync_opts'])" 2>/dev/null
    fi
}

LOCAL_CLONE="${LOCAL_CLONE:-/home/scott/nas-knowledge}"
CANONICAL_PATH="$(_canonical)"
RSYNC_OPTS="${RSYNC_OPTS:-$(_rsync_opts)}"
RSYNC_OPTS="${RSYNC_OPTS:--a --delete}"

if [ -z "$CANONICAL_PATH" ]; then
    log "ERROR: canonical_vault_path not found in $CONFIG_FILE"
    exit 1
fi

log "=== sync_vault_to_canonical: local clone -> canonical mirror ==="
log "Local clone : $LOCAL_CLONE"
log "Canonical   : $CANONICAL_PATH"

if [ ! -d "$LOCAL_CLONE" ]; then
    log "ERROR: local clone $LOCAL_CLONE missing; nothing to write."
    exit 1
fi
if [ ! -d "$CANONICAL_PATH" ]; then
    log "ERROR: canonical mirror $CANONICAL_PATH not mounted; refusing to write."
    exit 1
fi

# The ONLY write to the canonical mirror.
# shellcheck disable=SC2086
rsync $RSYNC_OPTS "$LOCAL_CLONE/" "$CANONICAL_PATH/"
rc=$?

if [ $rc -eq 0 ]; then
    date '+%Y-%m-%dT%H:%M:%S' > "$CANONICAL_PATH/.last-canonical-sync"
    log "✓ Canonical mirror written (single writer)."
else
    log "✗ rsync to canonical mirror failed (rc=$rc)."
    exit $rc
fi
