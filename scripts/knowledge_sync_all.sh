#!/bin/bash
# Single canonical vault sync (G2 consolidation, DEEP_DIVE §3.7).
#
# MODEL (see docs/VAULT_SYNC.md):
#   * The mirror vault (config.yaml `canonical_vault_path`, currently
#     /media/scott/NAS5/shared-knowledge) is THE ONLY WRITER.
#   * Every peer listed in config.yaml `knowledge_map.vault_peers` keeps a
#     local clone and PULLS from the canonical mirror. No host SSH-pushes
#     its local vault to other peers.
#   * Peers are discovered by Tailscale hostname only — never by raw IP.
#   * A peer that is unreachable (ssh/ping timeout) is SKIPPED so one
#     down machine never breaks the whole sync run.
#
# This script runs on the canonical-writer host (e.g. x1-370) and:
#   1. Pulls each reachable peer's clone into the local working copy.
#   2. Merges.
#   3. Pushes the merged result to the canonical mirror (single writer).
#   4. Pulls the canonical mirror back down to each reachable peer.

set -u

CONFIG_FILE="${CONFIG_FILE:-config.yaml}"
LOG_FILE="${LOG_FILE:-/home/scott/logs/knowledge_sync.log}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# --- config-driven discovery (no hardcoded IPs) ---------------------------
# Prefer yq; fall back to python3 (system python has pyyaml).
_cfg_get() {
    # $1 = dotted-or-list path we parse loosely; we just extract the values.
    local key="$1"
    if command -v yq >/dev/null 2>&1; then
        yq eval "$key" "$CONFIG_FILE" 2>/dev/null
    else
        python3 -c "import yaml,sys; c=yaml.safe_load(open('$CONFIG_FILE')); print('\n'.join(c$key) if isinstance(c$key,list) else c$key)" 2>/dev/null
    fi
}

# Read peers as a newline-separated list from the config.
mapfile -t MACHINES < <(_cfg_get "['knowledge_map']['vault_peers']")
SSH_USER="$(_cfg_get "['knowledge_map']['vault_sync']['ssh_user']")"
GIT_BRANCH="$(_cfg_get "['knowledge_map']['vault_sync']['git_branch']")"
RSYNC_OPTS="$(_cfg_get "['knowledge_map']['vault_sync']['rsync_opts']")"
PEER_TIMEOUT="$(_cfg_get "['knowledge_map']['vault_sync']['peer_timeout_sec']")"
CANONICAL_PATH="$(_cfg_get "['knowledge_map']['canonical_vault_path']")"
PEER_VAULT_PATH="$(_cfg_get "['knowledge_map']['peer_vault_path']")"

# Hostname of the machine running this sync; never push to ourselves.
SELF_HOST="$(hostname)"

# Defaults if config keys are somehow missing.
SSH_USER="${SSH_USER:-scott}"
GIT_BRANCH="${GIT_BRANCH:-main}"
RSYNC_OPTS="${RSYNC_OPTS:--a --delete}"
PEER_TIMEOUT="${PEER_TIMEOUT:-8}"
PEER_VAULT_PATH="${PEER_VAULT_PATH:-/home/scott/nas-knowledge}"

log "=== Starting Canonical Vault Sync ==="
log "Canonical (single writer): ${CANONICAL_PATH}"
log "Peers: ${MACHINES[*]:-<none>}"

# --- reachability probe ---------------------------------------------------
peer_reachable() {
    local host="$1"
    # Use a cheap, bounded ssh probe (Tailscale magic DNS resolves hostname).
    timeout "$PEER_TIMEOUT" ssh -o BatchMode=yes -o ConnectTimeout="$PEER_TIMEOUT" \
        -o StrictHostKeyChecking=no "$SSH_USER@$host" "true" >/dev/null 2>&1
}

# --- Step 1: pull each reachable peer's clone into our working copy -------
cd "$PEER_VAULT_PATH" || { log "ERROR: local vault clone $PEER_VAULT_PATH missing"; exit 1; }

log "Step 1: Fetching from reachable peers..."
for name in "${MACHINES[@]:-}"; do
    [ -z "$name" ] && continue
    if [ "$name" = "self" ] || [ "$name" = "$SELF_HOST" ]; then
        log "Skipping self ($name)."
        continue
    fi
    if ! peer_reachable "$name"; then
        log "SKIP $name (unreachable within ${PEER_TIMEOUT}s) — continuing."
        continue
    fi
    log "Fetching from $name..."
    timeout $((PEER_TIMEOUT * 10)) ssh "$SSH_USER@$name" \
        "cd '$PEER_VAULT_PATH' && git fetch origin $GIT_BRANCH 2>/dev/null" || \
        log "WARN: fetch from $name failed; skipping merge for it."
done

# --- Step 2: merge -------------------------------------------------------
log "Step 2: Merging..."
git pull --rebase "origin" "$GIT_BRANCH" || log "WARN: rebase/pull had issues; inspect before push."

if git status --porcelain | grep -q '^UU'; then
    log "CONFLICTS DETECTED! Manual resolution required."
    CONFLICT_FILES=$(git diff --name-only --diff-filter=U)
    cat > "$PEER_VAULT_PATH/.sync-conflicts-$(date +%Y%m%d-%H%M%S).md" << EOF
# Knowledge Sync Conflicts
Generated: $(date)

Conflicted files requiring manual resolution:
$CONFLICT_FILES

To resolve, run on this host:
  cd $PEER_VAULT_PATH
  git mergetool
  git add . && git commit -m "Resolved sync conflicts"
EOF
    log "Conflict report saved to: $PEER_VAULT_PATH/.sync-conflicts-*.md"
else
    log "No conflicts detected."
fi

# --- Step 3: push merged working copy to the CANONICAL mirror ------------
# The mirror is the only writer. Delegate the single authoritative write to
# sync_vault_to_canonical.sh so there is exactly one code path that writes
# the mirror (see docs/VAULT_SYNC.md).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/sync_vault_to_canonical.sh" ]; then
    CONFIG_FILE="$CONFIG_FILE" LOCAL_CLONE="$PEER_VAULT_PATH" \
        bash "$SCRIPT_DIR/sync_vault_to_canonical.sh"
else
    log "WARN: sync_vault_to_canonical.sh missing; falling back to inline rsync."
    log "Step 3: Pushing merged vault to canonical mirror ($CANONICAL_PATH)..."
    if [ -d "$CANONICAL_PATH" ]; then
        # shellcheck disable=SC2086
        rsync $RSYNC_OPTS "$PEER_VAULT_PATH/" "$CANONICAL_PATH/" && \
            log "✓ Canonical mirror updated (single writer)." || \
            log "✗ Failed to update canonical mirror."
    else
        log "WARN: canonical path $CANONICAL_PATH not mounted; skipping push."
    fi
fi

# --- Step 4: pull canonical back down to each reachable peer -------------
log "Step 4: Pulling canonical back down to reachable peers..."
for name in "${MACHINES[@]:-}"; do
    [ -z "$name" ] && continue
    if [ "$name" = "self" ] || [ "$name" = "$SELF_HOST" ]; then
        continue
    fi
    if ! peer_reachable "$name"; then
        log "SKIP $name (unreachable) — will catch up next run."
        continue
    fi
    log "Updating $name from canonical mirror..."
    timeout $((PEER_TIMEOUT * 10)) ssh "$SSH_USER@$name" \
        "cd '$PEER_VAULT_PATH' && git pull origin $GIT_BRANCH" 2>&1 | \
        tee -a "$LOG_FILE" || \
        log "WARN: pull on $name failed."
done

log "=== Canonical Vault Sync Complete ==="
