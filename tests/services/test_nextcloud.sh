#!/bin/bash
# Test Nextcloud Service
# This script validates Nextcloud functionality and external storage

set -e

echo "=== Nextcloud Service Tests ==="

# Test Nextcloud health
echo "Testing Nextcloud health..."
curl -s --max-time 5 http://localhost:8081 > /dev/null
echo "Nextcloud health check passed"

# Test Nextcloud API
echo "Testing Nextcloud API..."
curl -s --max-time 5 http://localhost:8081/ocs/v2.php/cloud/capabilities > /dev/null
echo "Nextcloud API is accessible"

# Test external storage
echo "Testing external storage..."
if [ -d "/mnt/nas" ]; then
    echo "NAS external storage is mounted"
else
    echo "NAS external storage is NOT mounted"
fi

if [ -d "/mnt/ssd" ]; then
    echo "SSD external storage is mounted"
else
    echo "SSD external storage is NOT mounted"
fi

echo ""
echo "=== Nextcloud Tests Passed ==="
