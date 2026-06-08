#!/bin/bash
# Set up daily knowledge sync cron jobs

LOG_FILE="/home/scott/logs/knowledge_sync.log"
SYNC_SCRIPT="/home/scott/git/auto-ingest/scripts/knowledge_sync_all.sh"

# Ensure log directory exists
mkdir -p "$(dirname $LOG_FILE)"

# Check if script is executable
if [ ! -x "$SYNC_SCRIPT" ]; then
    chmod +x "$SYNC_SCRIPT"
fi

# Create crontab entry for daily sync at 2 AM
CRON_ENTRY="0 2 * * * $SYNC_SCRIPT >> $LOG_FILE 2>&1"

echo "Adding cron job: $CRON_ENTRY"

# Add to crontab (append if exists)
(crontab -l 2>/dev/null | grep -v "$SYNC_SCRIPT"; echo "$CRON_ENTRY") | crontab -

echo "✓ Cron job added successfully"
echo ""
echo "To verify:"
crontab -l | grep knowledge_sync
echo ""
echo "To view logs:"
tail -f $LOG_FILE
