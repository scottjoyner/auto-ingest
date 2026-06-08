#!/bin/bash
# Multi-Machine Knowledge Sync Script
# Runs on x1-370 (master source) to sync knowledge across all machines

set -e

MASTER_PATH="/media/scott/S/shared-knowledge"
LOG_FILE="/home/scott/logs/knowledge_sync.log"
MACHINES=(
    "deathstar@100.78.106.121:~/nas-knowledge"
    "demo-1@100.65.68.58:/home/falcon/nas-knowledge"
    "destroyer@100.81.57.77:/home/scott/nas-knowledge"
    "optiplex@100.69.158.114:/home/scott/nas-knowledge"
)

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Step 1: Pull all machine branches into master
log "=== Starting Knowledge Sync ==="
cd "$MASTER_PATH" || exit 1

log "Step 1: Fetching from all machines..."
for machine in "${MACHINES[@]}"; do
    name=$(echo $machine | cut -d'@' -f1)
    ip=$(echo $machine | cut -d'@' -f2 | cut -d':' -f1)
    
    log "Fetching from $name ($ip)..."
    ssh "$name" "cd ~/knowledge/nas-knowledge && git fetch origin main 2>/dev/null || true"
done

# Step 2: Merge all branches and resolve conflicts
log "Step 2: Merging branches..."
git pull --rebase origin main

# Check for conflicts
if git status | grep -q "unmerged"; then
    log "CONFLICTS DETECTED! Manual resolution required."
    
    # List conflicted files
    CONFLICT_FILES=$(git diff --name-only --diff-filter=U)
    log "Conflicted files:"
    echo "$CONFLICT_FILES" | while read file; do
        log "  - $file"
    done
    
    # Create conflict report
    cat > "$MASTER_PATH/.sync-conflicts-$(date +%Y%m%d-%H%M%S).md" << EOF
# Knowledge Sync Conflicts
Generated: $(date)

Conflicted files requiring manual resolution:
$CONFLICT_FILES

To resolve, run on x1-370:
  cd $MASTER_PATH
  git mergetool
  # Review each conflict, then:
  git add . && git commit -m "Resolved sync conflicts"
EOF
    
    log "Conflict report saved to: $MASTER_PATH/.sync-conflicts-*.md"
else
    log "No conflicts detected. Continuing..."
fi

# Step 3: Run Neo4j sync via knowledge_map
log "Step 3: Running Neo4j sync..."
cd /home/scott/git/auto-ingest || exit 1

if [ -f config.yaml ]; then
    python3 -m knowledge_map sync_vault_to_neo4j --config config.yaml 2>&1 | tee -a "$LOG_FILE"
else
    log "WARNING: config.yaml not found, skipping Neo4j sync"
fi

# Step 4: Push to all machines
log "Step 4: Pushing updates to all machines..."
for machine in "${MACHINES[@]}"; do
    name=$(echo $machine | cut -d'@' -f1)
    
    log "Pushing to $name..."
    ssh "$name" << EOF
cd ~/knowledge/nas-knowledge || exit 1
git pull origin main
EOF
    
    if [ $? -eq 0 ]; then
        log "✓ Successfully synced to $name"
    else
        log "✗ Failed to sync to $name"
    fi
done

log "=== Knowledge Sync Complete ==="
