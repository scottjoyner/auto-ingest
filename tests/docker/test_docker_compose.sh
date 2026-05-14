#!/bin/bash
# Test Docker Compose Configuration
# This script validates the docker-compose.yml file and tests service connectivity

set -e

echo "=== Docker Compose Validation ==="

# Validate docker-compose.yml
echo "Validating docker-compose.yml..."
docker compose config > /dev/null
echo "docker-compose.yml is valid"

# Test service connectivity
echo ""
echo "=== Service Connectivity Tests ==="

# Test Neo4j
echo "Testing Neo4j..."
if curl -s --max-time 5 http://localhost:7474 > /dev/null; then
    echo "Neo4j is accessible"
else
    echo "Neo4j is NOT accessible"
    exit 1
fi

# Test Nextcloud
echo "Testing Nextcloud..."
if curl -s --max-time 5 http://localhost:8081 > /dev/null; then
    echo "Nextcloud is accessible"
else
    echo "Nextcloud is NOT accessible"
    exit 1
fi

# Test Signal-CLI
echo "Testing Signal-CLI..."
if curl -s --max-time 5 http://localhost:8400/v1/about > /dev/null; then
    echo "Signal-CLI is accessible"
else
    echo "Signal-CLI is NOT accessible"
    exit 1
fi

# Test Nginx
echo "Testing Nginx..."
if curl -s --max-time 5 http://localhost:80 > /dev/null; then
    echo "Nginx is accessible"
else
    echo "Nginx is NOT accessible"
    exit 1
fi

echo ""
echo "=== All Service Tests Passed ==="
