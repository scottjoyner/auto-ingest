#!/bin/bash
# Sync shared .hermes data across all Tailscale machines
# Runs on x1-370 (master) via cron or manually

set -e

SHARED_HERMES="/media/scott/S/shared-hermes"
LOG_FILE="/home/scott/logs/hermes_sync.log"
MACHINES=(
    "deathstar@100.78.106.121:~/nas-knowledge"
    "demo-1@100.65.68.58:/home/falcon/nas-knowledge"
    "destroyer@100.81.57.77:/home/scott/nas-knowledge"
    "optiplex@100.69.158.114:/home/scott/nas-knowledge"
)

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Step 1: Copy shared data to each machine's knowledge base
log "=== Starting .hermes Data Sync ==="

for machine in "${MACHINES[@]}"; do
    name=$(echo $machine | cut -d'@' -f1)
    path=$(echo $machine | cut -d':' -f2)
    
    log "Syncing to $name..."
    
    # Copy shared .hermes to machine's knowledge base
    ssh "$name" << EOF
cd ${path} || exit 1

# Create symlinked .hermes directory
mkdir -p ~/.hermes

# Remove existing symlinks if they exist
rm -f ~/.hermes/chats ~/.hermes/skills ~/.hermes/config

# Create new symlinks to shared location
ln -sf /media/scott/S/shared-hermes/chats ~/.hermes/chats
ln -sf /media/scott/S/shared-hermes/skills ~/.hermes/skills
ln -sf /media/scott/S/shared-hermes/config ~/.hermes/config

# Verify symlinks are valid
ls -la ~/.hermes/ | grep shared-hermes || echo "Symlink setup complete"
EOF
    
    if [ $? -eq 0 ]; then
        log "✓ Successfully synced to $name"
    else
        log "✗ Failed to sync to $name"
    fi
done

# Step 2: Copy per-machine configs from local ~/.hermes/
log "Syncing per-machine configurations..."

for machine in "${MACHINES[@]}"; do
    name=$(echo $machine | cut -d'@' -f1)
    
    # Check if machine has local config to sync
    ssh "$name" << EOF 2>/dev/null || true
if [ -f ~/.hermes/config-local/config.yaml ]; then
    cp ~/.hermes/config-local/config.yaml /media/scott/S/shared-hermes/config/${name}-config.yaml
fi
EOF
    
done

log "=== .hermes Data Sync Complete ==="
