#!/usr/bin/env bash
set -euo pipefail

CRON_FILE="${CRON_FILE:-/app/deploy/cron/ingest.crontab}"

if [[ ! -f "$CRON_FILE" ]]; then
  echo "Cron file not found: $CRON_FILE" >&2
  exit 1
fi

cp "$CRON_FILE" /etc/cron.d/app-cron
chmod 0644 /etc/cron.d/app-cron
crontab /etc/cron.d/app-cron

mkdir -p /var/log
touch /var/log/cron.log

exec cron -f
