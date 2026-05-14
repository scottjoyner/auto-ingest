# Containerized System Testing

This directory contains tests for the containerized home infrastructure system.

## Services

- **Neo4j**: Graph database with multi-database support (auto_ingest + knowledge_graph)
- **Nextcloud**: File storage and collaboration with external storage (NAS + SSD)
- **Signal-CLI**: Signal messaging REST API
- **Nginx**: Reverse proxy for all services
- **MySQL**: Database for Nextcloud
- **Redis**: Caching for Nextcloud

## Testing Infrastructure

### Test Types

1. **Docker Tests** (`tests/docker/`)
   - Validate docker-compose.yml configuration
   - Test service connectivity
   - Verify port mappings

2. **Service Tests** (`tests/services/`)
   - Neo4j health and database tests
   - Nextcloud API and external storage tests
   - Signal-CLI API tests

3. **Integration Tests** (`tests/integration/`)
   - Full system connectivity tests
   - Data persistence tests
   - Cross-service communication tests

4. **Unit Tests** (`tests/unit/`)
   - Configuration validation
   - Data model tests
   - Utility function tests

### Running Tests

```bash
# Run all tests
./tests/docker/test_docker_compose.sh
./tests/services/test_neo4j.sh
./tests/services/test_nextcloud.sh
./tests/services/test_signal_cli.sh
./tests/integration/test_full_system.sh

# Run Python unit tests
python3 -m pytest tests/unit/
```

## Test Configuration

Test configuration is stored in `tests/test_config.ini`. Update this file with your specific settings.

## External Storage

- **NAS**: `/media/scott/NAS/fileserver` mounted as `/mnt/nas` in Nextcloud
- **SSD**: `/media/scott/S` mounted as `/mnt/ssd` in Nextcloud

## Neo4j Configuration

- **Port**: 7474 (HTTP), 7687 (Bolt)
- **Database**: neo4j (default)
- **Password**: knowledge_graph_2026
- **Data Directory**: `/media/scott/S/neo4j/data`

## Nextcloud Configuration

- **Port**: 8081
- **Admin User**: admin
- **Admin Password**: admin_password_2026
- **Database**: MySQL (mariadb:10.6)
- **Cache**: Redis

## Signal-CLI Configuration

- **Port**: 8400
- **API Version**: v1
- **Data Directory**: `/home/.local/share/signal-cli`

## Nginx Configuration

- **Ports**: 80 (HTTP), 443 (HTTPS)
- **Proxy**: Routes requests to appropriate services
- **SSL**: Configured for HTTPS

## Troubleshooting

### Service Not Accessible

1. Check if the service is running: `docker compose ps`
2. Check service logs: `docker compose logs <service_name>`
3. Verify port mappings: `docker compose ps`

### External Storage Not Mounted

1. Verify mount points exist: `ls -la /mnt/nas /mnt/ssd`
2. Check Nextcloud external storage configuration: `docker exec nextcloud /var/www/html/occ files_external:list`

### Neo4j Connection Issues

1. Verify Neo4j is running: `curl -s http://localhost:7474`
2. Check Bolt connector: `cypher-shell -u neo4j -p knowledge_graph_2026 -a bolt://localhost:7687 "RETURN 1"`

## Future Testing

- Load testing for Neo4j
- Performance testing for Nextcloud
- Security scanning for all services
- Backup and restore testing
