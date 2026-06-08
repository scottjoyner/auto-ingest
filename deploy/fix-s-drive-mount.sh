#!/bin/bash
# Fix S drive mount and prepare for knowledge vault migration
# Location: /home/scott/git/auto-ingest/deploy/fix-s-drive-mount.sh

set -euo pipefail

echo "=== S Drive Mount Fix ==="

# Step 1: Check if S partition exists
S_PARTITION="/dev/nvme1n1p3"
if [[ ! -b "$S_PARTITION" ]]; then
    echo "ERROR: $S_PARTITION not found"
    exit 1
fi

echo "[OK] Found S partition: $S_PARTITION"

# Step 2: Create mount point if needed
MOUNT_POINT="/media/scott/S"
if [[ ! -d "$MOUNT_POINT" ]]; then
    echo "Creating mount point: $MOUNT_POINT"
    sudo mkdir -p "$MOUNT_POINT"
fi

echo "[OK] Mount point exists: $MOUNT_POINT"

# Step 3: Check current fstab entry
if grep -q "$MOUNT_POINT" /etc/fstab; then
    echo "Found existing fstab entry for S drive"
    grep "$MOUNT_POINT" /etc/fstab
else
    echo "Adding new fstab entry..."
    
    # Get UUID of S partition
    UUID=$(sudo blkid -s UUID -o value $S_PARTITION)
    echo "UUID: $UUID"
    
    # Add to fstab (NTFS with proper permissions for scott user)
    sudo bash -c "echo '$UUID  $MOUNT_POINT  ntfs-3g  uid=1000,gid=1000,umask=022,nofail,windows_names  0  0' >> /etc/fstab"
    
    echo "[OK] Added fstab entry:"
    grep "$MOUNT_POINT" /etc/fstab
fi

# Step 4: Mount the S drive
echo "Mounting S drive..."
sudo mount $MOUNT_POINT

# Step 5: Verify it's mounted and writeable
if mountpoint -q $MOUNT_POINT; then
    echo "[OK] S drive is now mounted at $MOUNT_POINT"
    
    # Test write permission
    TEST_FILE="$MOUNT_POINT/.write-test-$$"
    if touch "$TEST_FILE" 2>/dev/null; then
        echo "[OK] Write permissions verified"
        sudo rm -f "$TEST_FILE"
    else
        echo "WARNING: May need to adjust permissions"
        ls -la $MOUNT_POINT
    fi
else
    echo "ERROR: S drive failed to mount"
    exit 1
fi

# Step 6: Create shared-knowledge directory structure
echo "Creating knowledge vault structure..."
sudo mkdir -p "$MOUNT_POINT/shared-knowledge"/{00-Inbox,10-Infrastructure,20-Projects,30-Skills,40-Personal,50-Research,60-Mappings,70-Templates,80-Archives,90-References}

# Step 7: Initialize Git repo if not exists
if [[ ! -d "$MOUNT_POINT/shared-knowledge/.git" ]]; then
    echo "Initializing Git repository..."
    cd $MOUNT_POINT/shared-knowledge
    sudo git init
    sudo git config user.name "Hermes Sync Service"
    sudo git config user.email "hermes@scottjoyner.local"
else
    echo "Git repo already exists at $MOUNT_POINT/shared-knowledge/.git"
fi

# Step 8: Copy existing ~/knowledge to S drive (if not already done)
if [[ -d "/home/scott/knowledge" ]]; then
    if [[ ! -f "$MOUNT_POINT/shared-knowledge/.migrated" ]]; then
        echo "Migrating ~/knowledge to S drive..."
        sudo rsync -av /home/scott/knowledge/ "$MOUNT_POINT/shared-knowledge/"
        touch "$MOUNT_POINT/shared-knowledge/.migrated"
        
        cd $MOUNT_POINT/shared-knowledge
        sudo git add .
        sudo git commit -m "Initial migration from ~/knowledge to S drive" || true
    else
        echo "~/knowledge already migrated (found .migrated marker)"
    fi
fi

echo ""
echo "=== S Drive Setup Complete ==="
echo "S drive mounted at: $MOUNT_POINT"
echo "Knowledge vault location: $MOUNT_POINT/shared-knowledge/"
echo ""
echo "Next steps:"
echo "1. Update config.yaml to use /media/scott/S/shared-knowledge/ as central_vault_path"
echo "2. Run docker compose up -d knowledge-sync to start sync service"
echo "3. Verify with: ls -la $MOUNT_POINT/shared-knowledge/"
