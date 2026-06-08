# Knowledge Map System — Low-Level Design (LLD)

> **Version:** 1.0  
> **Last Updated:** 2026-05-30  
> **Author:** Scott Joyner / Hermes Agent  
> **Status:** Draft for review

---

## 1. Implementation Overview

This document specifies the concrete implementation details for the Knowledge Map System, including:
- Sync service architecture and scripts
- Neo4j query specifications
- Markdown template definitions
- Configuration parameters
- Deployment steps

---

## 2. Directory Structure

### 2.1 Central Library Location

**Primary Path:** `/media/scott/NAS1/shared-knowledge/` (S drive on x1-370)

**Current Status:** Mount exists but not writeable — requires config fix

**Fallback Path:** `~/knowledge` (local to each agent, Git-tracked)

### 2.2 Project Layout

```
/home/scott/git/auto-ingest/
├── HLD.md                          # High-level design (this doc's partner)
├── LLD.md                          # Low-level design (this file)
├── config.yaml                     # Main configuration
├── docker-compose.yml              # Service definitions
│
├── deploy/
│   ├── sync_service.py             # Bidirectional sync engine
│   ├── neo4j_to_vault.py           # Graph → Markdown extraction
│   ├── vault_to_neo4j.py           # Markdown → Graph ingestion
│   └── templates/                  # Jinja2 markdown templates
│       ├── device.md.j2
│       ├── service.md.j2
│       ├── project.md.j2
│       ├── infrastructure.md.j2
│       └── mapping.md.j2
│
├── sync/
│   ├── cron/                       # Cron job definitions
│   │   ├── neo4j-to-vault.cron     # Every 30 minutes
│   │   └── vault-to-neo4j.cron     # Every 1 hour
│   ├── logs/                       # Sync operation logs
│   └── state/                      # Sync state tracking
│       ├── last_sync_neo4j.json    # Last neo4j→vault sync timestamp
│       └── last_vault_sync.json    # Last vault→neo4j sync timestamp
│
├── queries/                        # Neo4j Cypher query definitions
│   ├── extract_devices.cypher
│   ├── extract_services.cypher
│   ├── extract_projects.cypher
│   └── search_semantic.cypher
│
└── tests/                          # Unit/integration tests
    ├── test_sync_service.py
    ├── test_neo4j_queries.py
    └── test_markdown_templates.py
```

---

## 3. Configuration Specification (Updated)

### 3.1 config.yaml Structure (Event-Driven Sync)

```yaml
# Knowledge Map System Configuration
# Location: /home/scott/git/auto-ingest/config.yaml

knowledge_map:
  # Central library location (S drive on x1-370)
  central_vault_path: "/media/scott/NAS1/shared-knowledge/"
  
  # Local fallback path
  local_vault_path: "~/knowledge"
  
  # Git configuration for versioning
  git:
    enabled: true
    repo_url: "git@github.com:scottjoyner/knowledge-vault.git"
    branch: "main"
    auto_commit: true
    commit_message_template: "Sync {direction} at {timestamp}"
  
  # Neo4j connection
  neo4j:
    uri: "bolt://100.64.43.123:7687"  # x1-370 Tailscale IP
    user: "neo4j"
    password: "${NEO4J_PASSWORD}"
    database: "neo4j"  # Main memory graph
  
  # Event-driven sync settings
  sync:
    # vault → neo4j direction (file change triggered)
    vault_to_neo4j:
      enabled: true
      trigger: "file_change"  # watchdog event
      debounce_seconds: 5     # Wait for batch of changes
      scan_modified_only: true
      
    # neo4j → vault direction (on write event or poll)
    neo4j_to_vault:
      enabled: true
      trigger: "write_event"  # Neo4j transaction hook
      fallback_poll_interval: "5m"  # Poll if no hooks available
      node_types:
        - Device
        - Service
        - Project
        - Infrastructure
        - Mapping
      confidence_threshold: 0.7
      max_nodes_per_sync: 100
  
  # Multi-agent write coordination
  agents:
    x1-370:
      role: "primary"
      write_vault: true
      read_vault: true
      neo4j_write: true
      lock_priority: 1
      
    deathstar-XPS-8920:
      role: "compute"
      write_vault: true
      read_vault: true
      neo4j_write: false
      lock_priority: 2
      
    scotts-macbook-air:
      role: "mobile"
      write_vault: true
      read_vault: true
      neo4j_write: false
      lock_priority: 3
      
    # Add other agents as needed
  
  # Template configuration
  templates:
    device: "templates/device.md.j2"
    service: "templates/service.md.j2"
    project: "templates/project.md.j2"
    infrastructure: "templates/infrastructure.md.j2"
    mapping: "templates/mapping.md.j2"
  
  # Logging
  logging:
    level: "INFO"
    file: "/var/log/knowledge-map/sync.log"
    max_size_mb: 100
    backup_count: 5
  
  # Sync queue settings
  sync_queue:
    max_size: 100
    worker_threads: 4
    retry_attempts: 3
```

### 3.2 Environment Variables (Updated)

```bash
# .env file for auto-ingest container
KNOLEDGE_MAP_ENABLED=true
KNOLEDGE_MAP_CENTRAL_PATH=/media/scott/NAS1/shared-knowledge/
KNOLEDGE_MAP_LOCAL_PATH=/home/scott/knowledge
NEO4J_URI=bolt://host.docker.internal:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=${NEO4J_PASSWORD}

# Event-driven sync settings
SYNC_TRIGGER=vault_change  # file_change, write_event, or both
SYNC_DEBOUNCE_SECONDS=5
SYNC_QUEUE_MAX_SIZE=100
```

---

## 4. Sync Service Architecture (Event-Driven)

### 4.1 Main Sync Engine (`sync_service.py`)

```python
#!/usr/bin/env python3
"""
Bidirectional sync engine for Knowledge Map System with event-driven triggers.
Location: /home/scott/git/auto-ingest/deploy/sync_service.py
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

from neo4j import GraphDatabase
from jinja2 import Environment, FileSystemLoader
import git
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logger = logging.getLogger(__name__)


class KnowledgeMapSyncService:
    """Bidirectional sync between Neo4j and markdown vault with event-driven triggers."""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.neo4j_driver = GraphDatabase.driver(
            self.config['knowledge_map']['neo4j']['uri'],
            auth=(
                self.config['knowledge_map']['neo4j']['user'],
                self.config['knowledge_map']['neo4j']['password']
            )
        )
        
        # Git repo for versioning
        if self.config['knowledge_map']['git']['enabled']:
            self.git_repo = git.Repo(
                self.config['knowledge_map']['central_vault_path']
            )
        
        # Sync queue for async processing
        self.sync_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
    
    async def run_sync(self, direction: str) -> dict:
        """Run sync in specified direction."""
        start_time = datetime.utcnow()
        
        if direction == "neo4j-to-vault":
            result = await self._sync_neo4j_to_vault()
        elif direction == "vault-to-neo4j":
            result = await self._sync_vault_to_neo4j()
        else:
            raise ValueError(f"Unknown direction: {direction}")
        
        result['duration_seconds'] = (datetime.utcnow() - start_time).total_seconds()
        result['timestamp'] = start_time.isoformat()
        
        # Log to assistx database
        await self._log_sync_result(result)
        
        return result
    
    async def _sync_neo4j_to_vault(self, node_ids: list = None) -> dict:
        """Extract nodes from Neo4j and generate markdown files."""
        sync_stats = {
            'direction': 'neo4j-to-vault',
            'nodes_processed': 0,
            'files_written': 0,
            'errors': []
        }
        
        # Get node types to extract (or specific nodes if provided)
        if node_ids:
            # Sync specific nodes only
            cypher_query = """
            MATCH (n) WHERE n.id IN $node_ids
            RETURN n
            """
            results = await self._execute_cypher(cypher_query, {'node_ids': node_ids})
        else:
            # Get node types to extract
            node_types = self.config['knowledge_map']['sync'][
                'neo4j_to_vault'
            ]['node_types']
            
            for node_type in node_types:
                cypher_query = self._get_cypher_query(node_type)
                results = await self._execute_cypher(cypher_query)
                
                # Generate markdown for each node
                template_path = self.config['knowledge_map']['templates'].get(
                    node_type.lower(), f"templates/{node_type.lower()}.md.j2"
                )
                jinja_env = Environment(
                    loader=FileSystemLoader('deploy/templates')
                )
                template = jinja_env.get_template(template_path)
                
                for record in results:
                    try:
                        markdown_content = template.render(node=record)
                        
                        # Write to central vault
                        file_path = self._generate_file_path(node_type, record)
                        await self._write_markdown(file_path, markdown_content)
                        
                        sync_stats['files_written'] += 1
                        
                    except Exception as e:
                        sync_stats['errors'].append({
                            'node_id': record.id,
                            'error': str(e)
                        })
                
                sync_stats['nodes_processed'] += len(results)
        
        # Commit changes to Git
        if self.config['knowledge_map']['git']['enabled']:
            await self._commit_to_git(sync_stats)
        
        return sync_stats
    
    async def _sync_vault_to_neo4j(self, file_path: str = None) -> dict:
        """Parse markdown files and update Neo4j."""
        sync_stats = {
            'direction': 'vault-to-neo4j',
            'files_scanned': 0,
            'nodes_updated': 0,
            'nodes_created': 0,
            'conflicts_resolved': 0,
            'errors': []
        }
        
        # Get modified files since last sync (or specific file)
        if file_path:
            modified_files = [file_path]
        else:
            modified_files = await self._get_modified_vault_files()
        
        for path in modified_files:
            try:
                # Parse markdown frontmatter + content
                node_data = self._parse_markdown(path)
                
                # Check for conflicts
                existing_node = await self._find_existing_node(node_data['node_id'])
                
                if existing_node:
                    # Resolve conflict based on timestamp and agent priority
                    if await self._should_overwrite(
                        existing_node, node_data
                    ):
                        await self._update_neo4j_node(existing_node, node_data)
                        sync_stats['nodes_updated'] += 1
                        
                        # Check if this was a conflict
                        if existing_node.last_modified != node_data['last_updated']:
                            sync_stats['conflicts_resolved'] += 1
                else:
                    # Create new node
                    await self._create_neo4j_node(node_data)
                    sync_stats['nodes_created'] += 1
                
            except Exception as e:
                sync_stats['errors'].append({
                    'file_path': path,
                    'error': str(e)
                })
            
            sync_stats['files_scanned'] += 1
        
        return sync_stats
    
    async def _log_sync_result(self, result: dict):
        """Log sync result to assistx database."""
        query = """
        MERGE (sync:Sync {timestamp: $timestamp})
        ON CREATE SET sync.direction = $direction, 
                      sync.status = 'completed'
        ON MATCH SET sync.nodes_processed = $nodes_processed,
                     sync.files_written = $files_written,
                     sync.duration_seconds = $duration
        RETURN sync
        """
        
        with self.neo4j_driver.session() as session:
            session.run(query, **result)
    
    async def _commit_to_git(self, stats: dict):
        """Commit changes to Git repository."""
        if not self.config['knowledge_map']['git']['enabled']:
            return
        
        repo = self.git_repo
        
        # Add all changed files
        repo.index.add([self.config['knowledge_map']['central_vault_path']])
        
        # Commit with auto-generated message
        timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
        message = f"Sync neo4j-to-vault at {timestamp}"
        
        repo.index.commit(message)
        
        # Push to remote (if configured)
        if self.config['knowledge_map']['git'].get('repo_url'):
            try:
                repo.remotes.origin.push()
            except git.GitCommandError as e:
                logger.warning(f"Git push failed: {e}")


class KnowledgeVaultWatcher(FileSystemEventHandler):
    """Watchdog handler for file changes in knowledge vault."""
    
    def __init__(self, sync_service: KnowledgeMapSyncService, debounce_seconds: int = 5):
        super().__init__()
        self.sync_service = sync_service
        self.debounce_seconds = debounce_seconds
        self._pending_files: Dict[str, datetime] = {}
    
    async def on_modified(self, event):
        if event.src_path.endswith('.md'):
            await self._queue_sync('vault-to-neo4j', event.src_path)
    
    async def on_created(self, event):
        if event.src_path.endswith('.md'):
            await self._queue_sync('vault-to-neo4j', event.src_path)
    
    async def _queue_sync(self, direction: str, file_path: str):
        """Queue sync with debounce to batch multiple changes."""
        now = datetime.utcnow()
        
        # Debounce: if same file changed recently, wait for more changes
        if file_path in self._pending_files:
            last_change = self._pending_files[file_path]
            if (now - last_change).total_seconds() < self.debounce_seconds:
                logger.debug(f"Debounce: {file_path}, waiting...")
                return
        
        self._pending_files[file_path] = now
        await self.sync_service.sync_queue.put({
            'direction': direction,
            'path': file_path
        })


async def process_sync_queue(sync_service: KnowledgeMapSyncService):
    """Process sync queue with worker threads."""
    while True:
        event = await sync_service.sync_queue.get()
        
        try:
            result = await sync_service.run_sync(
                event['direction'],
                event.get('path')
            )
            logger.info(f"Sync completed: {result}")
        except Exception as e:
            logger.error(f"Sync failed: {e}")


async def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Knowledge Map Sync Service (Event-Driven)'
    )
    parser.add_argument(
        'direction',
        choices=['neo4j-to-vault', 'vault-to-neo4j', 'both']
    )
    parser.add_argument(
        '--config', default='/home/scott/git/auto-ingest/config.yaml'
    )
    
    args = parser.parse_args()
    
    service = KnowledgeMapSyncService(args.config)
    
    if args.direction == 'both':
        result1 = await service.run_sync('neo4j-to-vault')
        result2 = await service.run_sync('vault-to-neo4j')
        print(f"Neo4j→Vault: {result1}")
        print(f"Vault→Neo4j: {result2}")
    else:
        result = await service.run_sync(args.direction)
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    asyncio.run(main())
```

### 4.2 Watchdog Service (`deploy/watchdog.py`)

```python
#!/usr/bin/env python3
"""
File system watchdog for knowledge vault changes.
Location: /home/scott/git/auto-ingest/deploy/watchdog.py
"""

import asyncio
import logging
from pathlib import Path

from deploy.sync_service import (
    KnowledgeMapSyncService, 
    KnowledgeVaultWatcher,
    process_sync_queue
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def run_watchdog():
    """Run file system watchdog for knowledge vault."""
    
    # Load config
    from deploy.sync_service import load_config
    config = load_config('/home/scott/git/auto-ingest/config.yaml')
    
    central_path = Path(config['knowledge_map']['central_vault_path'])
    debounce_seconds = config['sync'].get('vault_to_neo4j', {}).get(
        'debounce_seconds', 5
    )
    
    # Initialize sync service and watchdog
    sync_service = KnowledgeMapSyncService('/home/scott/git/auto-ingest/config.yaml')
    watcher = KnowledgeVaultWatcher(sync_service, debounce_seconds)
    
    # Start observer
    observer = Observer()
    observer.schedule(watcher, str(central_path), recursive=True)
    observer.start()
    
    logger.info(f"Watchdog started for {central_path}")
    
    try:
        # Process sync queue in background
        queue_task = asyncio.create_task(process_sync_queue(sync_service))
        
        # Keep running
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Stopping watchdog...")
        observer.stop()
        observer.join()


if __name__ == '__main__':
    asyncio.run(run_watchdog())
```

### 4.3 Sync Queue Manager (`deploy/sync_queue.py`)

```python
#!/usr/bin/env python3
"""
Async sync queue manager with worker threads.
Location: /home/scott/git/auto-ingest/deploy/sync_queue.py
"""

import asyncio
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class SyncQueueManager:
    """Manages async sync queue with configurable workers."""
    
    def __init__(self, max_size: int = 100, worker_threads: int = 4):
        self.queue = asyncio.Queue(maxsize=max_size)
        self.worker_threads = worker_threads
        self._workers = []
    
    async def enqueue(self, direction: str, path: str = None) -> bool:
        """Add sync event to queue."""
        try:
            await self.queue.put({
                'direction': direction,
                'path': path
            })
            return True
        except asyncio.QueueFull:
            logger.warning(f"Sync queue full, dropping {direction}")
            return False
    
    async def start_workers(self, sync_func):
        """Start worker tasks to process queue."""
        self._workers = [
            asyncio.create_task(self._worker(sync_func))
            for _ in range(self.worker_threads)
        ]
    
    async def _worker(self, sync_func):
        """Worker coroutine that processes queue items."""
        while True:
            try:
                event = await self.queue.get()
                
                try:
                    if event['path']:
                        await sync_func(event['direction'], event['path'])
                    else:
                        await sync_func(event['direction'])
                    
                    self.queue.task_done()
                except Exception as e:
                    logger.error(f"Worker error processing {event}: {e}")
            
            except asyncio.CancelledError:
                break
    
    async def stop(self):
        """Stop all workers."""
        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)


# Singleton instance
_sync_queue_manager = None


def get_sync_queue_manager() -> SyncQueueManager:
    global _sync_queue_manager
    if _sync_queue_manager is None:
        from deploy.sync_service import load_config
        config = load_config('/home/scott/git/auto-ingest/config.yaml')
        sync_cfg = config['knowledge_map'].get('sync_queue', {})
        _sync_queue_manager = SyncQueueManager(
            max_size=sync_cfg.get('max_size', 100),
            worker_threads=sync_cfg.get('worker_threads', 4)
        )
    return _sync_queue_manager


async def enqueue_sync(direction: str, path: str = None):
    """Convenience function to enqueue sync."""
    manager = get_sync_queue_manager()
    await manager.enqueue(direction, path)
```

---

## 5. Neo4j Query Specifications (Updated for Event-Driven)

### 5.1 Extract Devices with Timestamp (`queries/extract_devices.cypher`)

```cypher
// Extract all Device nodes with full properties and last_updated timestamp
MATCH (d:Device)
WHERE d.confidence >= $confidence_threshold
RETURN 
  d.id AS node_id,
  d.name AS name,
  d.role AS role,
  d.tailscale_ip AS tailscale_ip,
  d.ssh_status AS ssh_status,
  d.docker_enabled AS docker,
  d.neo4j_enabled AS neo4j,
  d.lmstudio_enabled AS lmstudio,
  d.ollama_enabled AS ollama,
  d.specs AS specs,
  d.last_updated AS last_updated,
  {
    services: [(d)-[:RUNS_SERVICE]->(s:Service) | s.name],
    projects: [(d)-[:USED_IN_PROJECT]->(p:Project) | p.name]
  } AS relationships
ORDER BY d.last_updated DESC
LIMIT $max_nodes
```

### 5.2 Extract Services with Timestamp (`queries/extract_services.cypher`)

```cypher
// Extract all Service nodes with host mappings and last_updated timestamp
MATCH (s:Service)-[:HOSTED_ON]->(d:Device)
WHERE s.confidence >= $confidence_threshold
RETURN 
  s.id AS node_id,
  s.name AS name,
  s.type AS service_type,
  s.port_http AS port_http,
  s.port_bolt AS port_bolt,
  s.version AS version,
  s.auth_user AS auth_user,
  s.description AS description,
  d.name AS host_device,
  d.tailscale_ip AS host_ip,
  s.last_updated AS last_updated,
  [(s)-[:PART_OF_PROJECT]->(p:Project) | p.name] AS projects
ORDER BY s.last_updated DESC
LIMIT $max_nodes
```

### 5.3 Extract Projects with Timestamp (`queries/extract_projects.cypher`)

```cypher
// Extract all Project nodes with status and links
MATCH (p:Project)
WHERE p.confidence >= $confidence_threshold
RETURN 
  p.id AS node_id,
  p.name AS name,
  p.status AS status,
  p.description AS description,
  p.github_repo AS github_repo,
  p.local_path AS local_path,
  p.start_date AS start_date,
  p.last_updated AS last_updated,
  [(p)-[:HAS_COMPONENT]->(c) | c.name] AS components,
  [(p)-[:USES_SERVICE]->(s:Service) | s.name] AS services_used,
  [(p)-[:RELATED_TO]->(r:Project) | r.name] AS related_projects
ORDER BY p.last_updated DESC
LIMIT $max_nodes
```

### 5.4 Semantic Search (`queries/search_semantic.cypher`)

```cypher
// Vector search for semantic similarity with timestamp filtering
CALL db.index.vector.queryNodes(
  'node_embeddings', 
  $top_k, 
  $query_vector
)
YIELD node, score
WHERE node.confidence >= $confidence_threshold
  AND node.last_updated > datetime($since_timestamp)
RETURN 
  node.id AS node_id,
  node.name AS name,
  labels(node)[0] AS node_type,
  score AS similarity_score,
  {
    content: CASE 
      WHEN 'Transcription' IN labels(node) THEN node.transcript
      WHEN 'Summary' IN labels(node) THEN node.summary
      ELSE node.description
    END
  } AS context_snippet,
  node.last_updated AS last_updated
ORDER BY score DESC
LIMIT $top_k
```

---

## 6. Markdown Templates (Updated with Agent ID)

### 6.1 Device Template (`templates/device.md.j2`) - Updated

```jinja2
# {{ node.name }}

> **Last Updated:** {{ node.last_updated }}  
> **Node ID:** `{{ node.node_id }}`  
> **Synced By:** {{ agent_id | default('unknown') }}

## Overview

{{ node.description | default('No description available.') }}

## Specifications

| Property | Value |
|----------|-------|
| Role | {{ node.role }} |
| Tailscale IP | {{ node.tailscale_ip }} |
| SSH Status | {% if node.ssh_status %}{{ node.ssh_status }}{% else %}N/A{% endif %} |
| Docker | {% if node.docker_enabled %}YES{% else %}NO{% endif %} |
| Neo4j | {% if node.neo4j_enabled %}YES{% else %}NO{% endif %} |
| LMStudio | {% if node.lmstudio_enabled %}YES{% else %}NO{% endif %} |
| Ollama | {% if node.ollama_enabled %}YES{% else %}NO{% endif %} |

## Specs

```yaml
{{ node.specs }}
```

## Associated Services

{% if node.services %}
- {{ node.services | join(', ') }}
{% else %}
No services hosted on this device.
{% endif %}

## Projects

{% if node.projects %}
- {{ node.projects | join(', ') }}
{% else %}
No projects using this device.
{% endif %}

---

**Source:** Neo4j (`{{ node.node_id }}`)  
**Synced by agent:** `{{ agent_id | default('unknown') }}` at `{{ now() }}`  
**Related:** [[10-Infrastructure/Overview]]
```

### 6.2 Service Template (`templates/service.md.j2`) - Updated

```jinja2
# {{ node.name }}

> **Last Updated:** {{ node.last_updated }}  
> **Node ID:** `{{ node.node_id }}`  
> **Synced By:** {{ agent_id | default('unknown') }}

## Overview

{{ node.description | default('No description available.') }}

## Configuration

| Property | Value |
|----------|-------|
| Type | {{ node.service_type }} |
| HTTP Port | {% if node.port_http %}{{ node.port_http }}{% else %}N/A{% endif %} |
| Bolt Port | {% if node.port_bolt %}{{ node.port_bolt }}{% else %}N/A{% endif %} |
| Version | {{ node.version }} |
| Auth User | {{ node.auth_user }} |

## Hosted On

- **Device:** [[10-Infrastructure/devices/{{ node.host_device | slugify }}.md]]
- **Tailscale IP:** `{{ node.host_ip }}`

## Projects

{% if node.projects %}
- {{ node.projects | join(', ') }}
{% else %}
No projects using this service.
{% endif %}

---

**Source:** Neo4j (`{{ node.node_id }}`)  
**Synced by agent:** `{{ agent_id | default('unknown') }}` at `{{ now() }}`  
**Related:** [[10-Infrastructure/services/Overview]]
```

### 6.3 Project Template (`templates/project.md.j2`) - Updated

```jinja2
# {{ node.name }}

> **Last Updated:** {{ node.last_updated }}  
> **Node ID:** `{{ node.node_id }}`  
> **Status:** {{ node.status }}  
> **Synced By:** {{ agent_id | default('unknown') }}

## Overview

{{ node.description | default('No description available.') }}

## Repository

{% if node.github_repo %}
- **GitHub:** [{{ node.github_repo }}]({{ node.github_repo }})
{% endif %}

{% if node.local_path %}
- **Local Path:** `{{ node.local_path }}`
{% endif %}

## Components

{% if node.components %}
| Component | Description |
|-----------|-------------|
{% for component in node.components %}
| {{ component }} | See project docs |
{% endfor %}
{% else %}
No components defined.
{% endif %}

## Services Used

{% if node.services_used %}
- {{ node.services_used | join(', ') }}
{% else %}
No services used.
{% endif %}

## Related Projects

{% if node.related_projects %}
- {{ node.related_projects | join(', ') }}
{% else %}
No related projects.
{% endif %}

---

**Source:** Neo4j (`{{ node.node_id }}`)  
**Synced by agent:** `{{ agent_id | default('unknown') }}` at `{{ now() }}`  
**Related:** [[20-Projects/Overview]]
```

---

## 7. Cron Job Definitions (Updated for Event-Driven)

### 7.1 Fallback Polling Cron (`deploy/cron/ne4j-to-vault-poll.cron`)

```bash
# Fallback: poll Neo4j every 5 minutes if no write events detected
*/5 * * * * /usr/bin/env bash -c "cd /app && python deploy/sync_service.py neo4j-to-vault >> /var/log/knowledge-map/neo4j-to-vault-poll.log 2>&1"
```

### 7.2 Manual Full Reconciliation Cron (`deploy/cron/full-reconcile.cron`)

```bash
# Run daily at 2 AM for complete sync (fallback)
0 2 * * * /usr/bin/env bash -c "cd /app && python deploy/sync_service.py both >> /var/log/knowledge-map/full-reconcile.log 2>&1"
```

---

## 8. S Drive Mount Configuration Fix (Updated)

### 8.3 docker-compose.yml Update (Multi-Agent Write Support)

```yaml
version: '3.8'

services:
  knowledge-sync:
    build:
      context: .
      dockerfile: Dockerfile.sync
    container_name: knowledge-sync
    restart: unless-stopped
    
    environment:
      - NEO4J_URI=bolt://host.docker.internal:7687
      - NEO4J_USER=neo4j
      - NEO4J_PASSWORD=${NEO4J_PASSWORD}
      - CENTRAL_VAULT_PATH=/nas/fileserver/shared-knowledge/
      - LOCAL_VAULT_PATH=/home/scott/knowledge
      - SYNC_TRIGGER=vault_change  # Event-driven trigger
      - SYNC_DEBOUNCE_SECONDS=5
      - AGENT_ID=x1-370  # Unique agent identifier
      
    volumes:
      # Mount central vault (S drive) with write access
      - /media/scott/NAS1/shared-knowledge:/nas/fileserver/shared-knowledge:rw
      # Mount local vault as fallback
      - ~/knowledge:/home/scott/knowledge:rw
      # Mount config
      - ./config.yaml:/app/config.yaml:ro
      
    extra_hosts:
      - "host.docker.internal:host-gateway"  # For Neo4j access
    
    user: "${UID}:${GID}"  # Pass host UID/GID for write permissions
    
    deploy:
      resources:
        limits:
          cpus: '1'
          memory: 512M
      
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
    
    # Health check for watchdog service
    healthcheck:
      test: ["CMD", "python", "-c", "import asyncio; from deploy.watchdog import run_watchdog; asyncio.run(run_watchdog())"]
      interval: 30s
      timeout: 10s
      retries: 3
```

---

## 9. Deployment Steps (Updated for Event-Driven)

### 9.2 Deployment Commands (Event-Driven)

```bash
# Step 1: Navigate to auto-ingest repo
cd /home/scott/git/auto-ingest

# Step 2: Create shared-knowledge directory on S drive (if not exists)
mkdir -p /media/scott/NAS1/shared-knowledge/{00-Inbox,10-Infrastructure,20-Projects,30-Skills,40-Personal,50-Research,60-Mappings,70-Templates,80-Archives,90-References}

# Step 3: Initialize Git repo (if not exists)
cd /media/scott/NAS1/shared-knowledge
git init
git config user.name "Hermes Sync Service"
git config user.email "hermes@scottjoyner.local"

# Step 4: Copy existing ~/knowledge to S drive (one-time migration)
rsync -av ~/knowledge/ /media/scott/NAS1/shared-knowledge/
cd /media/scott/NAS1/shared-knowledge
git add .
git commit -m "Initial migration from ~/knowledge"

# Step 5: Update config.yaml with event-driven settings
vi config.yaml  # Edit sync.trigger to "file_change"

# Step 6: Install watchdog dependency
pip install watchdog

# Step 7: Build and start sync service
docker compose up -d --build knowledge-sync

# Step 8: Verify watchdog is running
docker logs -f knowledge-sync | grep -i "watchdog\|sync"

# Step 9: Trigger first manual sync to establish baseline
docker exec knowledge-sync python deploy/sync_service.py neo4j-to-vault

# Step 10: Test event-driven sync by modifying a file
echo "# Test" >> /media/scott/NAS1/shared-knowledge/60-Mappings/test.md
# Should trigger vault→neo4j sync within ~5 seconds
```

---

## 10. Monitoring & Logging (Updated)

### 10.1 Log File Locations (Event-Driven)

| Service | Log Path | Rotation |
|---------|----------|----------|
| watchdog events | `/var/log/knowledge-map/watchdog.log` | 10MB, 5 backups |
| vault→neo4j sync | `/var/log/knowledge-map/vault-to-neo4j.log` | 10MB, 5 backups |
| neo4j→vault poll | `/var/log/knowledge-map/neo4j-to-vault-poll.log` | 10MB, 5 backups |
| full reconcile | `/var/log/knowledge-map/full-reconcile.log` | 10MB, 5 backups |

### 10.2 Sync State Tracking (Updated)

```json
// sync/state/last_sync_neo4j.json
{
  "last_sync": "2026-05-30T15:30:00+00:00",
  "direction": "neo4j-to-vault",
  "trigger": "write_event|poll_fallback",
  "nodes_processed": 9,
  "files_written": 7,
  "agent_id": "x1-370",
  "errors": [],
  "duration_seconds": 2.34
}

// sync/state/last_sync_vault.json
{
  "last_sync": "2026-05-30T16:00:00+00:00",
  "direction": "vault-to-neo4j",
  "trigger": "file_change|manual",
  "files_scanned": 12,
  "nodes_updated": 3,
  "nodes_created": 1,
  "conflicts_resolved": 0,
  "agent_id": "x1-370",
  "errors": []
}

// sync/state/sync_queue.json (new)
{
  "queue_size": 5,
  "pending_events": [
    {"direction": "vault-to-neo4j", "path": "/media/scott/S/shared-knowledge/10-Infrastructure/devices/test.md", "timestamp": "2026-05-30T16:05:00Z"}
  ],
  "processed_last_hour": 47,
  "failed_last_hour": 2
}
```

---

## 11. Testing Strategy (Updated)

### 11.1 Unit Tests (`tests/test_sync_service.py`) - Updated

```python
import pytest
from deploy.sync_service import KnowledgeMapSyncService


class TestKnowledgeMapSync:
    
    @pytest.fixture
    def sync_service(self):
        return KnowledgeMapSyncService('config.yaml')
    
    async def test_neo4j_to_vault_extraction(self, sync_service):
        """Test that Neo4j nodes are correctly extracted to markdown."""
        result = await sync_service._sync_neo4j_to_vault()
        
        assert 'nodes_processed' in result
        assert 'files_written' in result
        assert isinstance(result['nodes_processed'], int)
    
    async def test_vault_to_neo4j_ingestion(self, sync_service):
        """Test that markdown files are correctly ingested to Neo4j."""
        result = await sync_service._sync_vault_to_neo4j()
        
        assert 'files_scanned' in result
        assert 'nodes_updated' in result
        assert 'nodes_created' in result
    
    async def test_event_driven_sync(self, sync_service):
        """Test that file changes trigger async sync."""
        from deploy.watchdog import KnowledgeVaultWatcher
        
        watcher = KnowledgeVaultWatcher(sync_service, debounce_seconds=1)
        
        # Simulate file change event
        await watcher.on_modified(type('Event', (), {'src_path': '/tmp/test.md'})())
        
        # Verify event was queued
        assert not sync_service.sync_queue.empty()
    
    async def test_conflict_resolution_multi_agent(self, sync_service):
        """Test that timestamp-based conflict resolution works with agent priority."""
        # Create conflicting markdown files with same timestamp from different agents
        # Verify x1-370 version is preferred over other agents
        pass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

---

## 12. Rollback Strategy (Updated)

### 12.1 Git-Based Rollback (Multi-Agent)

```bash
# View sync history with agent IDs
cd /media/scott/S/shared-knowledge
git log --oneline --format="%h %an %ad" | head -20

# Revert to previous version
git revert HEAD~1  # Revert last commit
git push origin main

# Or restore specific commit from specific agent
git checkout <commit-hash> -- path/to/file.md
```

### 12.2 Neo4j Rollback (Updated)

```cypher
// Restore node from backup (if using neo4j-admin)
CALL dbms.dumpBackup('neo4j-backup-2026-05-30')

// Or delete problematic nodes with agent tracking
MATCH (n:Device {name: 'problematic-device', synced_by: 'agent-x1-370'})
DETACH DELETE n

// View sync history in Neo4j
MATCH (s:Sync) RETURN s.timestamp, s.direction, s.nodes_processed ORDER BY s.timestamp DESC LIMIT 20
```

---

## Appendix C: Requirements (Updated)

```txt
# requirements.txt for sync service
neo4j>=5.24.0
jinja2>=3.1.0
gitpython>=3.1.0
pydantic>=2.0.0
asyncio>=3.4.3
aiohttp>=3.9.0
watchdog>=3.0.0  # File system watcher
```

## Appendix D: Error Codes (Updated)

| Code | Meaning | Resolution |
|------|---------|------------|
| SYNC_001 | Neo4j connection failed | Check URI, credentials, network |
| SYNC_002 | Vault path not writeable | Fix mount permissions, UID/GID |
| SYNC_003 | Git commit failed | Check repo state, remote URL |
| SYNC_004 | Template render error | Validate Jinja2 template syntax |
| SYNC_005 | Conflict resolution timeout | Manual review required in assistx dashboard |
| SYNC_006 | Watchdog queue full | Increase max_size or add more workers |
| SYNC_007 | Distributed lock held by other agent | Wait for lock release, then retry |

---

## 5. Neo4j Query Specifications

### 5.1 Extract Devices (`queries/extract_devices.cypher`)

```cypher
// Extract all Device nodes with full properties
MATCH (d:Device)
WHERE d.confidence >= $confidence_threshold
RETURN 
  d.id AS node_id,
  d.name AS name,
  d.role AS role,
  d.tailscale_ip AS tailscale_ip,
  d.ssh_status AS ssh_status,
  d.docker_enabled AS docker,
  d.neo4j_enabled AS neo4j,
  d.lmstudio_enabled AS lmstudio,
  d.ollama_enabled AS ollama,
  d.specs AS specs,
  d.last_seen AS last_updated,
  {
    services: [(d)-[:RUNS_SERVICE]->(s:Service) | s.name],
    projects: [(d)-[:USED_IN_PROJECT]->(p:Project) | p.name]
  } AS relationships
ORDER BY d.name
LIMIT $max_nodes
```

### 5.2 Extract Services (`queries/extract_services.cypher`)

```cypher
// Extract all Service nodes with host mappings
MATCH (s:Service)-[:HOSTED_ON]->(d:Device)
WHERE s.confidence >= $confidence_threshold
RETURN 
  s.id AS node_id,
  s.name AS name,
  s.type AS service_type,
  s.port_http AS port_http,
  s.port_bolt AS port_bolt,
  s.version AS version,
  s.auth_user AS auth_user,
  s.description AS description,
  d.name AS host_device,
  d.tailscale_ip AS host_ip,
  s.last_updated AS last_updated,
  [(s)-[:PART_OF_PROJECT]->(p:Project) | p.name] AS projects
ORDER BY s.name
LIMIT $max_nodes
```

### 5.3 Extract Projects (`queries/extract_projects.cypher`)

```cypher
// Extract all Project nodes with status and links
MATCH (p:Project)
WHERE p.confidence >= $confidence_threshold
RETURN 
  p.id AS node_id,
  p.name AS name,
  p.status AS status,
  p.description AS description,
  p.github_repo AS github_repo,
  p.local_path AS local_path,
  p.start_date AS start_date,
  p.last_updated AS last_updated,
  [(p)-[:HAS_COMPONENT]->(c) | c.name] AS components,
  [(p)-[:USES_SERVICE]->(s:Service) | s.name] AS services_used,
  [(p)-[:RELATED_TO]->(r:Project) | r.name] AS related_projects
ORDER BY p.name
LIMIT $max_nodes
```

### 5.4 Semantic Search (`queries/search_semantic.cypher`)

```cypher
// Vector search for semantic similarity
CALL db.index.vector.queryNodes(
  'node_embeddings', 
  $top_k, 
  $query_vector
)
YIELD node, score
WHERE node.confidence >= $confidence_threshold
RETURN 
  node.id AS node_id,
  node.name AS name,
  labels(node)[0] AS node_type,
  score AS similarity_score,
  {
    content: CASE 
      WHEN 'Transcription' IN labels(node) THEN node.transcript
      WHEN 'Summary' IN labels(node) THEN node.summary
      ELSE node.description
    END
  } AS context_snippet
ORDER BY score DESC
LIMIT $top_k
```

---

## 6. Markdown Templates

### 6.1 Device Template (`templates/device.md.j2`)

```jinja2
# {{ node.name }}

> **Last Updated:** {{ node.last_updated }}  
> **Node ID:** `{{ node.node_id }}`

## Overview

{{ node.description | default('No description available.') }}

## Specifications

| Property | Value |
|----------|-------|
| Role | {{ node.role }} |
| Tailscale IP | {{ node.tailscale_ip }} |
| SSH Status | {{ node.ssh_status }} |
| Docker | {% if node.docker_enabled %}YES{% else %}NO{% endif %} |
| Neo4j | {% if node.neo4j_enabled %}YES{% else %}NO{% endif %} |
| LMStudio | {% if node.lmstudio_enabled %}YES{% else %}NO{% endif %} |
| Ollama | {% if node.ollama_enabled %}YES{% else %}NO{% endif %} |

## Specs

```yaml
{{ node.specs }}
```

## Associated Services

{% if node.services %}
- {{ node.services | join(', ') }}
{% else %}
No services hosted on this device.
{% endif %}

## Projects

{% if node.projects %}
- {{ node.projects | join(', ') }}
{% else %}
No projects using this device.
{% endif %}

---

**Source:** Neo4j (`{{ node.node_id }}`)  
**Related:** [[10-Infrastructure/Overview]]
```

### 6.2 Service Template (`templates/service.md.j2`)

```jinja2
# {{ node.name }}

> **Last Updated:** {{ node.last_updated }}  
> **Node ID:** `{{ node.node_id }}`

## Overview

{{ node.description | default('No description available.') }}

## Configuration

| Property | Value |
|----------|-------|
| Type | {{ node.service_type }} |
| HTTP Port | {% if node.port_http %}{{ node.port_http }}{% else %}N/A{% endif %} |
| Bolt Port | {% if node.port_bolt %}{{ node.port_bolt }}{% else %}N/A{% endif %} |
| Version | {{ node.version }} |
| Auth User | {{ node.auth_user }} |

## Hosted On

- **Device:** [[10-Infrastructure/devices/{{ node.host_device | slugify }}.md]]
- **Tailscale IP:** `{{ node.host_ip }}`

## Projects

{% if node.projects %}
- {{ node.projects | join(', ') }}
{% else %}
No projects using this service.
{% endif %}

---

**Source:** Neo4j (`{{ node.node_id }}`)  
**Related:** [[10-Infrastructure/services/Overview]]
```

### 6.3 Project Template (`templates/project.md.j2`)

```jinja2
# {{ node.name }}

> **Last Updated:** {{ node.last_updated }}  
> **Node ID:** `{{ node.node_id }}`  
> **Status:** {{ node.status }}

## Overview

{{ node.description | default('No description available.') }}

## Repository

{% if node.github_repo %}
- **GitHub:** [{{ node.github_repo }}]({{ node.github_repo }})
{% endif %}

{% if node.local_path %}
- **Local Path:** `{{ node.local_path }}`
{% endif %}

## Components

{% if node.components %}
| Component | Description |
|-----------|-------------|
{% for component in node.components %}
| {{ component }} | See project docs |
{% endfor %}
{% else %}
No components defined.
{% endif %}

## Services Used

{% if node.services_used %}
- {{ node.services_used | join(', ') }}
{% else %}
No services used.
{% endif %}

## Related Projects

{% if node.related_projects %}
- {{ node.related_projects | join(', ') }}
{% else %}
No related projects.
{% endif %}

---

**Source:** Neo4j (`{{ node.node_id }}`)  
**Related:** [[20-Projects/Overview]]
```

---

## 7. Cron Job Definitions

### 7.1 neo4j-to-vault Cron (`deploy/cron/neo4j-to-vault.cron`)

```bash
# Run every 30 minutes
*/30 * * * * /usr/bin/env bash -c "cd /app && python deploy/sync_service.py neo4j-to-vault >> /var/log/knowledge-map/neo4j-to-vault.log 2>&1"
```

### 7.2 vault-to-neo4j Cron (`deploy/cron/vault-to-neo4j.cron`)

```bash
# Run every hour
0 * * * * /usr/bin/env bash -c "cd /app && python deploy/sync_service.py vault-to-neo4j >> /var/log/knowledge-map/vault-to-neo4j.log 2>&1"
```

### 7.3 Full Reconciliation Cron (`deploy/cron/full-reconcile.cron`)

```bash
# Run daily at 2 AM for complete sync
0 2 * * * /usr/bin/env bash -c "cd /app && python deploy/sync_service.py both >> /var/log/knowledge-map/full-reconcile.log 2>&1"
```

---

## 8. S Drive Mount Configuration Fix

### 8.1 Current Issue

The S drive (`/media/scott/NAS1`) exists but is not writeable from the auto-ingest container.

### 8.2 Root Cause

- exfat filesystem mounted with read-only flags
- Container lacks write permissions to mount point
- User/group ID mismatch between host and container

### 8.3 Fix Steps

```bash
# Step 1: Check current mount options
mount | grep NAS1

# Expected output (problematic):
# /dev/sdX on /media/scott/NAS1 type exfat (ro,nosuid,owner,rw)

# Step 2: Remount with write permissions
sudo mount -o remount,rw /media/scott/NAS1

# Step 3: Verify write access
touch /media/scott/NAS1/.write-test && rm /media/scott/NAS1/.write-test

# Step 4: Update docker-compose.yml to pass correct UID/GID
# In docker-compose.yml, add:
services:
  sync-service:
    user: "${UID}:${GID}"  # Add this line
    environment:
      - NAS_PATH=/media/scott/NAS1/shared-knowledge/

# Step 5: Make mount persistent (optional)
# Edit /etc/fstab to ensure proper mount options on boot
```

### 8.4 docker-compose.yml Update

```yaml
version: '3.8'

services:
  knowledge-sync:
    build:
      context: .
      dockerfile: Dockerfile.sync
    container_name: knowledge-sync
    restart: unless-stopped
    
    environment:
      - NEO4J_URI=bolt://host.docker.internal:7687
      - NEO4J_USER=neo4j
      - NEO4J_PASSWORD=${NEO4J_PASSWORD}
      - CENTRAL_VAULT_PATH=/nas/fileserver/shared-knowledge/
      - LOCAL_VAULT_PATH=/home/scott/knowledge
      
    volumes:
      # Mount central vault (S drive)
      - /media/scott/NAS1/shared-knowledge:/nas/fileserver/shared-knowledge:rw
      # Mount local vault as fallback
      - ~/knowledge:/home/scott/knowledge:rw
      # Mount config
      - ./config.yaml:/app/config.yaml:ro
      
    extra_hosts:
      - "host.docker.internal:host-gateway"  # For Neo4j access
    
    user: "${UID}:${GID}"  # Pass host UID/GID for write permissions
    
    deploy:
      resources:
        limits:
          cpus: '1'
          memory: 512M
      
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

---

## 9. Deployment Steps

### 9.1 Pre-deployment Checklist

- [ ] Verify S drive is writeable (`/media/scott/NAS1`)
- [ ] Create shared-knowledge directory structure
- [ ] Initialize Git repository (if not exists)
- [ ] Set up Neo4j connection credentials
- [ ] Install Python dependencies: `pip install -r requirements.txt`

### 9.2 Deployment Commands

```bash
# Step 1: Navigate to auto-ingest repo
cd /home/scott/git/auto-ingest

# Step 2: Create shared-knowledge directory on S drive
mkdir -p /media/scott/NAS1/shared-knowledge/{00-Inbox,10-Infrastructure,20-Projects,30-Skills,40-Personal,50-Research,60-Mappings,70-Templates,80-Archives,90-References}

# Step 3: Initialize Git repo (if not exists)
cd /media/scott/NAS1/shared-knowledge
git init
git config user.name "Hermes Sync Service"
git config user.email "hermes@scottjoyner.local"

# Step 4: Copy existing ~/knowledge to S drive (one-time migration)
rsync -av ~/knowledge/ /media/scott/NAS1/shared-knowledge/
cd /media/scott/NAS1/shared-knowledge
git add .
git commit -m "Initial migration from ~/knowledge"

# Step 5: Update config.yaml with correct paths
vi config.yaml  # Edit central_vault_path to /media/scott/NAS1/shared-knowledge/

# Step 6: Build and start sync service
docker compose up -d --build knowledge-sync

# Step 7: Verify sync is running
docker logs -f knowledge-sync

# Step 8: Trigger first sync manually
docker exec knowledge-sync python deploy/sync_service.py neo4j-to-vault
```

---

## 10. Monitoring & Logging

### 10.1 Log File Locations

| Service | Log Path | Rotation |
|---------|----------|----------|
| neo4j→vault sync | `/var/log/knowledge-map/neo4j-to-vault.log` | 10MB, 5 backups |
| vault→neo4j sync | `/var/log/knowledge-map/vault-to-neo4j.log` | 10MB, 5 backups |
| full reconcile | `/var/log/knowledge-map/full-reconcile.log` | 10MB, 5 backups |

### 10.2 Sync State Tracking

```json
// sync/state/last_sync_neo4j.json
{
  "last_sync": "2026-05-30T15:30:00+00:00",
  "direction": "neo4j-to-vault",
  "nodes_processed": 9,
  "files_written": 7,
  "errors": [],
  "duration_seconds": 2.34
}

// sync/state/last_sync_vault.json
{
  "last_sync": "2026-05-30T16:00:00+00:00",
  "direction": "vault-to-neo4j",
  "files_scanned": 12,
  "nodes_updated": 3,
  "nodes_created": 1,
  "conflicts_resolved": 0,
  "errors": []
}
```

### 10.3 Health Check Endpoint

```python
# Add to sync_service.py for Docker health check
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "last_neo4j_to_vault_sync": last_sync_time,
        "last_vault_to_neo4j_sync": last_vault_sync_time,
        "neo4j_connected": neo4j_driver.verify_connection()
    }
```

---

## 11. Testing Strategy

### 11.1 Unit Tests (`tests/test_sync_service.py`)

```python
import pytest
from deploy.sync_service import KnowledgeMapSyncService


class TestKnowledgeMapSync:
    
    @pytest.fixture
    def sync_service(self):
        return KnowledgeMapSyncService('config.yaml')
    
    async def test_neo4j_to_vault_extraction(self, sync_service):
        """Test that Neo4j nodes are correctly extracted to markdown."""
        result = await sync_service._sync_neo4j_to_vault()
        
        assert 'nodes_processed' in result
        assert 'files_written' in result
        assert isinstance(result['nodes_processed'], int)
    
    async def test_vault_to_neo4j_ingestion(self, sync_service):
        """Test that markdown files are correctly ingested to Neo4j."""
        result = await sync_service._sync_vault_to_neo4j()
        
        assert 'files_scanned' in result
        assert 'nodes_updated' in result
        assert 'nodes_created' in result
    
    async def test_conflict_resolution(self, sync_service):
        """Test that timestamp-based conflict resolution works."""
        # Create conflicting markdown files with same timestamp
        # Verify x1-370 version is preferred
        pass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

---

## 12. Rollback Strategy

### 12.1 Git-Based Rollback

```bash
# View sync history
cd /media/scott/NAS1/shared-knowledge
git log --oneline | head -20

# Revert to previous version
git revert HEAD~1  # Revert last commit
git push origin main

# Or restore specific commit
git checkout <commit-hash> -- path/to/file.md
```

### 12.2 Neo4j Rollback

```cypher
// Restore node from backup (if using neo4j-admin)
CALL dbms.dumpBackup('neo4j-backup-2026-05-30')

// Or delete problematic nodes
MATCH (n:Device {name: 'problematic-device'})
DETACH DELETE n
```

---

## Appendix A: Requirements

```txt
# requirements.txt for sync service
neo4j>=5.24.0
jinja2>=3.1.0
gitpython>=3.1.0
pydantic>=2.0.0
asyncio>=3.4.3
aiohttp>=3.9.0
```

## Appendix B: Error Codes

| Code | Meaning | Resolution |
|------|---------|------------|
| SYNC_001 | Neo4j connection failed | Check URI, credentials, network |
| SYNC_002 | Vault path not writeable | Fix mount permissions, UID/GID |
| SYNC_003 | Git commit failed | Check repo state, remote URL |
| SYNC_004 | Template render error | Validate Jinja2 template syntax |
| SYNC_005 | Conflict resolution timeout | Manual review required in assistx dashboard |
