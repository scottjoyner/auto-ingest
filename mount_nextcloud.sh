#!/usr/bin/env bash
# mount_nextcloud.sh — reach the real Nextcloud filestore.
#
# Two ways to use the data:
#   1) Live FUSE mount (optional). If `rclone` (or `davfs2`) is available this
#      mounts the Nextcloud WebDAV share to MOUNT (default /media/scott/SSD_4TB/nextcloud)
#      so the rest of the pipeline can treat it like any local dir.
#   2) WebDAV pull (always works, no root mounts). ingest_media.py can pull
#      files directly from Nextcloud over WebDAV — just run:
#        auto-ingest ingest --nextcloud-url "$NEXTCLOUD_URL" \
#            --nextcloud-user "$NEXTCLOUD_USER" --nextcloud-pass "$NEXTCLOUD_PASS" --kind all
#
# Credentials: set NEXTCLOUD_URL / NEXTCLOUD_USER / NEXTCLOUD_PASS in the env
# (or pass as args). The URL is the WebDAV base, e.g.
#   https://cloud.example.com/remote.php/dav/files/ME/Photos
set -euo pipefail

MOUNT="${NEXTCLOUD_MOUNT:-/media/scott/SSD_4TB/nextcloud}"
URL="${NEXTCLOUD_URL:-${1:-}}"
USER="${NEXTCLOUD_USER:-${2:-}}"
PASS="${NEXTCLOUD_PASS:-${3:-}}"

if [ -z "$URL" ]; then
  echo "Usage: NEXTCLOUD_URL=... NEXTCLOUD_USER=... NEXTCLOUD_PASS=... $0 [mount]" >&2
  echo "  (or: $0 <url> <user> <pass>)" >&2
  exit 2
fi

mkdir -p "$MOUNT"

# Prefer rclone (handles Nextcloud/WebDAV natively, no root needed).
if command -v rclone >/dev/null 2>&1; then
  echo "[mount_nextcloud] rclone present -> mounting $URL to $MOUNT"
  rclone mount ":webdav:$URL" "$MOUNT" \
    --user "$USER" --password "$PASS" \
    --daemon --vfs-cache-mode minimal
  echo "[mount_nextcloud] mounted at $MOUNT"
  exit 0
fi

# Fall back to davfs2 if installed.
if command -v mount.davfs >/dev/null 2>&1; then
  echo "[mount_nextcloud] davfs2 present -> mounting $URL to $MOUNT"
  echo "$USER:$PASS" > ~/.nextcloud_davfs_credentials
  chmod 600 ~/.nextcloud_davfs_credentials
  mount -t davfs "$URL" "$MOUNT" -o noexec,user,username="$USER" \
    && echo "[mount_nextcloud] mounted at $MOUNT" && exit 0
fi

echo "[mount_nextcloud] No FUSE mounter (rclone/davfs2) available." >&2
echo "  -> Use the WebDAV-pull path instead (no mount needed):" >&2
echo "     auto-ingest ingest --nextcloud-url '$URL' \\" >&2
echo "         --nextcloud-user '$USER' --nextcloud-pass '***' --kind all" >&2
exit 0
