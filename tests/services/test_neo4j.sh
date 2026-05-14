#!/bin/bash
# Test Neo4j Service
# This script validates Neo4j functionality and database connectivity

set -e

echo "=== Neo4j Service Tests ==="

# Test Neo4j health
echo "Testing Neo4j health..."
curl -s --max-time 5 http://localhost:7474 > /dev/null
echo "Neo4j health check passed"

# Test Neo4j Bolt connector
echo "Testing Neo4j Bolt connector..."
if command -v cypher-shell > /dev/null 2>&1; then
    cypher-shell -u neo4j -p knowledge_graph_2026 -a bolt://localhost:7687 "RETURN 1 AS test" > /dev/null
    echo "Neo4j Bolt connector is working"
else
    echo "cypher-shell not available, skipping Bolt test"
fi

# Test Neo4j databases
echo "Testing Neo4j databases..."
curl -s --max-time 5 http://localhost:7474/db/manage/database > /dev/null
echo "Neo4j database management is accessible"

echo ""
echo "=== Neo4j Tests Passed ==="
