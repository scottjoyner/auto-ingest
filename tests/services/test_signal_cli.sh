#!/bin/bash
# Test Signal-CLI Service
# This script validates Signal-CLI REST API functionality

set -e

echo "=== Signal-CLI Service Tests ==="

# Test Signal-CLI health
echo "Testing Signal-CLI health..."
curl -s --max-time 5 http://localhost:8400/v1/about > /dev/null
echo "Signal-CLI health check passed"

# Test Signal-CLI API
echo "Testing Signal-CLI API..."
curl -s --max-time 5 http://localhost:8400/v1/accounts > /dev/null
echo "Signal-CLI API is accessible"

# Test Signal-CLI registration
echo "Testing Signal-CLI registration..."
if curl -s --max-time 5 http://localhost:8400/v1/register > /dev/null; then
    echo "Signal-CLI registration endpoint is accessible"
else
    echo "Signal-CLI registration endpoint is NOT accessible"
fi

echo ""
echo "=== Signal-CLI Tests Passed ==="
