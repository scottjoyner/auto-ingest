#!/bin/bash
# install-signal-bridge-cron.sh — run as scott on x1-370.
# Installs the Signal->KG bridge as a user cron job.
# The script itself lives on the shared SSD_4TB mount, so it's already deployed.
set -e
REPO=/media/scott/SSD_4TB/hermes-home/auto-ingest
LOGDIR=$REPO/logs
mkdir -p "$LOGDIR"

CRONLINE="*/20 * * * * cd $REPO && $REPO/.venv/bin/python scripts/signal_kg_bridge.py >> $LOGDIR/signal_kg.log 2>&1"

# merge: drop any prior signal_kg_bridge line, append the new one
( crontab -l 2>/dev/null | grep -v signal_kg_bridge; echo "$CRONLINE" ) | crontab -

echo "=== installed ==="
crontab -l | grep signal_kg

# ensure signal-cli is available + show account hint
if command -v signal-cli >/dev/null 2>&1; then
  echo "signal-cli found: $(command -v signal-cli)"
else
  echo "WARN: signal-cli not on PATH — bridge will skip until installed/linked"
fi
echo "Set SIGNAL_ACCOUNT in the cron env or ~/.profile if not the default device."
