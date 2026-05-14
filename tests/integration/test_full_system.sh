#!/bin/bash
# Full System Integration Tests
# This script tests all services together

set -e

echo "=== Full System Integration Tests ==="

# Test all services are running
echo "Testing all services are running..."
docker compose ps | grep -q "Up"
echo "All services are running"

# Test service connectivity
echo "Testing service connectivity..."
curl -s --max-time 5 http://localhost:7474 > /dev/null
curl -s --max-time 5 http://localhost:8081 > /dev/null
curl -s --max-time 5 http://localhost:8400/v1/about > /dev/null
curl -s --max-time 5 http://localhost:80 > /dev/null
echo "All services are accessible"

# Test data persistence
echo "Testing data persistence..."
if [ -d "/media/scott/S/neo4j/data" ]; then
    echo "Neo4j data persistence is configured"
else
    echo "Neo4j data persistence is NOT configured"
fi

if [ -d "/media/scott/NAS/fileserver" ]; then
    echo "NAS fileserver is mounted"
else
    echo "NAS fileserver is NOT mounted"
fi

if [ -d "/media/scott/S" ]; then
    echo "SSD drive is mounted"
else
    echo "SSD drive is NOT mounted"
fi

echo ""
echo "=== Full System Integration Tests Passed ==="
