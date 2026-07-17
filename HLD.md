# Knowledge Map System — High-Level Design (HLD)

> **Version:** 1.0  
> **Last Updated:** 2026-05-30  
> **Author:** Scott Joyner / Hermes Agent  
> **Status:** Draft for review

---

## 1. Overview

The Knowledge Map System connects three layers:

1. **Neo4j Graph Database** — Primary source of structured knowledge (20M+ nodes, 50M+ relationships)
2. **Markdown Knowledge Vault** — Human-readable documentation organized by topic and project
3. **Distributed Agent Network** — Multiple Hermes agents across the Tailscale swarm that read/write to both layers

The system enables agents to:
- Query Neo4j first for context-aware answers
- Fall back to markdown files for detailed documentation
- Sync changes bidirectionally between graph and vault
- Maintain a central shared library accessible from all nodes

---

## 2. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TAILSCALE SWARM NETWORK                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐       │
│  │   x1-370     │     │ deathstar    │     │ Other Nodes  │       │
│  │ (Primary)    │◄───►│ (Compute)    │◄───►│ (Agents)     │       │
│  └──────┬───────┘     └──────────────┘     └──────────────┘       │
│         │                          Tailscale mesh                  │
│         │ bolt://7687              (100.x.y.z IPs)                 │
│         ▼                                                          │
│  ┌─────────────────────────────────────────────────────────┐      │
│  │                    NEO4J GRAPH DB                       │      │
│  │  neo4j: ${NEO4J_PASSWORD}                            │      │
│  │  - PhoneLog (20M)                                      │      │
│  │  - DashcamEmbedding (4.2M)                             │      │
│  │  - Utterance, Speaker, Frame, YOLODetection            │      │
│  │  - assistx: orchestration state                        │      │
│  └───────────────┬─────────────────────────────────────────┘      │
│                  │ neo4j driver / cypher queries                   │
│                  ▼                                                  │
│  ┌─────────────────────────────────────────────────────────┐      │
│  │           SHARED MARKDOWN LIBRARY                       │      │
│  │  Location: /nas/fileserver/shared-knowledge/            │      │
│  │  Structure: Same as ~/knowledge vault (10 sections)     │      │
│  │  Git-tracked for versioning and conflict resolution     │      │
│  └───────────────┬─────────────────────────────────────────┘      │
│                  │ sync mechanism                                  │
│                  ▼                                                  │
│  ┌─────────────────────────────────────────────────────────┐      │
│  │              SYNC SERVICE                               │      │
│  │  - neo4j → vault: Extract relevant nodes to markdown    │      │
│  │  - vault → neo4j: Ingest markdown changes back to graph │      │
│  │  - Conflict resolution via timestamps + Git             │      │
│  └─────────────────────────────────────────────────────────┘      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Core Components

### 3.1 Neo4j Graph Database

**Purpose:** Primary source of structured, queryable knowledge about Scott's world.

**Databases:**
- `neo4j` (main): Historical/personal memory graph (20M+ nodes)
- `assistx`: Orchestration/control-plane state (tasks, agent runs, tool calls)
- `memory` (transitional): Sophia voice/auth data (being migrated to main)

**Key Node Types:**
| Label | Count | Purpose |
|-------|-------|---------|
| PhoneLog | 20M | Call/SMS records |
| DashcamEmbedding | 4.2M | Video embeddings |
| YOLODetection | 4.1M | Object detections |
| Frame | 3.7M | Video frames |
| Utterance | 420K | Speech utterances |
| Speaker | 233K | Voice identities |
| Transcription | 64K | Transcript records |
| Summary | 179K | Content summaries |

### 3.2 Markdown Knowledge Vault

**Purpose:** Human-readable documentation that complements the graph with narrative context, project plans, and reference material.

**Structure (10 numbered sections):**
```
~/knowledge/
├── 00-Inbox/          # Quick captures, unsorted notes
├── 10-Infrastructure/ # Hardware, networking, services
├── 20-Projects/       # Active project documentation
├── 30-Skills/         # Hermes agent skills catalog
├── 40-Personal/       # User profiles, preferences
├── 50-Research/       # Emerging research topics
├── 60-Mappings/       # Cross-reference maps, dashboards
├── 70-Templates/      # Note templates
├── 80-Archives/       # Completed/obsolete content
└── 90-References/     # Static reference docs
```

**Central Location:** `/nas/fileserver/shared-knowledge/` (mounted on all agents)

### 3.3 Sync Service

**Purpose:** Bidirectional synchronization between Neo4j and markdown vault.

**Sync Directions:**
1. **neo4j → vault**: Extract relevant nodes, generate/update markdown files
2. **vault → neo4j**: Parse markdown changes, create/update graph nodes

**Trigger Mechanisms:**
- Scheduled cron (every 30 minutes)
- Manual trigger via API
- Event-driven on file changes (future enhancement)

### 3.4 Agent Network

**Roles:**
| Node | Role | Write to Central Vault? | Read from Central Vault? |
|------|------|------------------------|-------------------------|
| x1-370 | Primary control plane | YES | YES |
| deathstar-XPS-8920 | Compute node | YES | YES |
| scotts-macbook-air | Mobile agent | YES | YES |
| Other agents | Execution workers | YES | YES |

**Query Flow:**
```
Agent receives request
    │
    ▼
1. Query Neo4j first for context-aware answers
    │
    ▼
2. If context found → Answer from graph data
    │
    ▼
3. If detailed docs needed → Read markdown files from vault
    │
    ▼
4. Synthesize answer from both sources
```

**Write Coordination:**
- All agents can write to central vault
- Distributed lock prevents concurrent writes to same file
- Git-based conflict resolution via merge strategy

---

## 4. Data Flow

### 4.1 Ingestion Flow (Historical Memory)

```
Source Files (dashcam/audio/bodycam)
        │
        ▼
auto-ingest pipeline (weekly/monthly batch)
        │
        ├─► Normalize & classify content
        ├─► Extract entities, opinions, preferences
        └─► Promote to Neo4j graph
                │
                ▼
        Vector embeddings generated
                │
                ▼
        neo4j → vault sync (30 min interval)
                │
                ▼
        Markdown files updated in central library
```

### 4.2 Query Flow (Real-Time)

```
User request to agent
        │
        ├─► Sophia voice auth (if applicable)
        └─► AssistX task creation
                │
                ▼
        Neo4j query:
          - Cypher for structured data
          - Vector search for semantic similarity
                │
                ▼
        If context incomplete → Read markdown vault
                │
                ▼
        LLM synthesis (local model endpoint)
                │
                ▼
        Response to user + log to Neo4j
```

### 4.3 Sync Flow (Bidirectional)

```
Scheduled sync trigger (cron every 30 min)
        │
        ├─► neo4j → vault:
        │     - Query nodes by type/category
        │     - Generate markdown templates
        │     - Write to central library
        │     - Commit to Git with diff tracking
        │
        └─► vault → neo4j:
              - Scan for modified/added markdown files
              - Parse frontmatter + content
              - Create/update graph nodes
              - Log changes to assistx database
```

---

## 5. Central Library Design

### 5.1 Location & Access

**Primary Path:** `/nas/fileserver/shared-knowledge/`

**Mount Strategy:**
- x1-370: Native mount (exfat/NAS)
- Other nodes: NFS/Samba mount via Tailscale
- Fallback: Clone repo if network unavailable

### 5.2 Git Integration

**Repository:** `scottjoyner/knowledge-vault` (private GitHub)

**Workflow:**
1. All sync writes commit to local Git
2. Daily push to remote for backup
3. Conflict resolution via merge strategy
4. Branch per agent for parallel edits (future)

### 5.3 File Naming Convention

```
{section}-{category}-{topic}.md
Examples:
- 10-infrastructure-services-neo4j.md
- 20-projects-auto-ingest-architecture.md
- 60-mappings-infra-map.md
```

**Frontmatter Template:**
```yaml
---
title: "Neo4j Service Details"
section: 10-Infrastructure
category: services
last_updated: "2026-05-30"
source: neo4j
node_ids: ["neo4j_instance_1", "neo4j_config_1"]
tags: [neo4j, database, service]
related: [[10-infrastructure-overview], [60-mappings-infra-map]]
---
```

---

## 6. Agent Query Strategy

### 6.1 Context Priority

When answering a query, agents follow this priority:

1. **Neo4j structured data** (highest confidence)
   - Cypher queries for exact matches
   - Vector search for semantic similarity
   
2. **Markdown vault** (narrative context)
   - Read relevant markdown files from central library
   - Extract specific sections via headers/anchors
   
3. **LLM synthesis** (combine both sources)
   - Local model endpoint on nearest agent
   - Prompt includes graph data + markdown excerpts

### 6.2 Fallback Behavior

```python
def answer_query(query, context=None):
    # Step 1: Neo4j search
    graph_results = neo4j_search(query, limit=50)
    
    if graph_results.confidence > THRESHOLD:
        return synthesize_from_graph(graph_results)
    
    # Step 2: Markdown fallback
    markdown_files = find_relevant_vault_files(query)
    vault_content = read_markdown_files(markdown_files)
    
    if vault_content:
        return synthesize_from_both(graph_results, vault_content)
    
    # Step 3: Ask for clarification
    return "Context unclear. Please provide more details."
```

---

## 7. Sync Mechanism Design (Event-Driven)

### 7.1 Event-Based Triggers

Syncs are now triggered by file changes rather than scheduled intervals:

| Trigger | Direction | Frequency | Data Scope |
|---------|-----------|-----------|------------|
| File change detected | vault → neo4j | Real-time (~5 sec delay) | Modified files only |
| Neo4j node update | neo4j → vault | On write event | Changed nodes only |
| Manual trigger | Both directions | On-demand | Full reconciliation |

### 7.2 Watchdog Mechanism

**File Change Detection:**
```python
# deploy/watchdog.py - File system watcher
import watchdog.observers
import watchdog.events

class KnowledgeVaultWatcher(watchdog.observers.Observer):
    def on_modified(self, event):
        if event.src_path.endswith('.md'):
            # Queue for vault → neo4j sync
            queue_sync_event('vault-to-neo4j', event.src_path)
    
    def on_created(self, event):
        if event.src_path.endswith('.md'):
            queue_sync_event('vault-to-neo4j', event.src_path)
```

**Neo4j Change Detection:**
- Triggered via Neo4j transaction hooks (future enhancement)
- Or periodic poll every 5 minutes for write events

### 7.3 Sync Queue

```python
# deploy/sync_queue.py - Async sync queue
from asyncio import Queue, create_task

sync_queue = Queue(maxsize=100)

async def enqueue_sync(direction: str, source_path: str):
    await sync_queue.put({'direction': direction, 'path': source_path})

async def process_sync_queue():
    while True:
        event = await sync_queue.get()
        create_task(run_sync(event['direction'], event['path']))
```

### 7.4 Conflict Resolution (Multi-Agent Writes)

**Strategy:** Git merge with automatic resolution where possible

1. **No conflict:** Files don't overlap → Auto-merge
2. **Same file, different sections:** Line-based merge via Git
3. **Same file, same section:** Timestamp + agent priority:
   - x1-370 > deathstar > other agents
   - If same timestamp → manual review required

**Conflict Resolution Flow:**
```
Agent A writes to file.md
    │
    ▼
Check distributed lock (Redis/Neo4j)
    │
    ├─► Lock available → Acquire lock, write file
    │       │
    │       ▼
    │   Commit to Git with agent ID
    │       │
    │       ▼
    │   Trigger vault→neo4j sync
    │
    └─► Lock held by Agent B → Queue write request
            │
            ▼
        Wait for lock release, then retry
```

---

## 8. Hierarchy & Ownership (Updated)

### 8.1 Data Ownership

| Layer | Owner | Write Access | Read Access |
|-------|-------|--------------|-------------|
| Neo4j `neo4j` | All agents | Ingest workers + authorized agents | All agents |
| Neo4j `assistx` | AssistX runtime | Orchestrator + all agents | All agents |
| Markdown vault | All agents | All agents (with lock) | All agents |

### 8.2 Section Ownership (Updated for Multi-Agent Write)

| Section | Primary Owner | Secondary Owners | Lock Priority |
|---------|---------------|------------------|---------------|
| 10-Infrastructure | x1-370 admin | All agents | x1-370 > deathstar > others |
| 20-Projects | Project leads | Contributors | Author > x1-370 > others |
| 30-Skills | Skills curator | All agents | Curator > x1-370 > others |
| 40-Personal | Scott (user) | AssistX runtime | Scott > x1-370 > others |
| 50-Research | Research team | All agents | Author > x1-370 > others |
| 60-Mappings | Mapping maintainers | All agents | Maintainer > x1-370 > others |

---

## 9. Future Enhancements (Updated)

### Phase 2 (Q3 2026)
- [ ] Event-driven sync on file changes (watchdog) ✅ *In design*
- [ ] Agent-specific branches for parallel editing
- [ ] Vector embeddings in markdown files for semantic search
- [ ] Conflict resolution UI via AssistX dashboard

### Phase 3 (Q4 2026)
- [ ] FalkorDB cache layer for fast graph queries
- [ ] Multi-agent write coordination with distributed locks ✅ *In design*
- [ ] Automated knowledge graph visualization from vault
- [ ] Voice-to-graph direct ingestion (Sophia → Neo4j) |

---

## 8. Hierarchy & Ownership

### 8.1 Data Ownership

| Layer | Owner | Write Access | Read Access |
|-------|-------|--------------|-------------|
| Neo4j `neo4j` | auto-ingest pipeline | Ingest workers | All agents |
| Neo4j `assistx` | AssistX runtime | Orchestrator | All agents |
| Markdown vault | Central library | x1-370 + authorized agents | All agents |

### 8.2 Section Ownership

| Section | Primary Owner | Secondary Owners |
|---------|---------------|------------------|
| 10-Infrastructure | x1-370 admin | All agents |
| 20-Projects | Project leads | Contributors |
| 30-Skills | Skills curator | All agents |
| 40-Personal | Scott (user) | AssistX runtime |
| 50-Research | Research team | All agents |
| 60-Mappings | Mapping maintainers | All agents |

---

## 9. Future Enhancements

### Phase 2 (Q3 2026)
- [ ] Event-driven sync on file changes (watchdog)
- [ ] Agent-specific branches for parallel editing
- [ ] Vector embeddings in markdown files for semantic search
- [ ] Conflict resolution UI via AssistX dashboard

### Phase 3 (Q4 2026)
- [ ] FalkorDB cache layer for fast graph queries
- [ ] Multi-agent write coordination with distributed locks
- [ ] Automated knowledge graph visualization from vault
- [ ] Voice-to-graph direct ingestion (Sophia → Neo4j)

---

## 10. Open Questions

1. **Central location:** Should `/nas/fileserver/shared-knowledge/` be NFS-mounted or Git-cloned on each agent?
2. **Write permissions:** Which agents can write to the central vault vs. read-only?
3. **Sync filtering:** What criteria determine which Neo4j nodes get synced to markdown?
4. **Conflict strategy:** Should we use Git branches per agent or merge-based approach?
5. **Performance:** How do we handle sync of 20M+ nodes without blocking queries?

---

## Appendix A: References

- [UNIFICATION.md](docs/UNIFICATION.md) — Cross-repo alignment plan
- [system_design.md](../auto-ingest/docs/system_design.md) — Auto-ingest architecture
- [architecture.md](../auto-ingest/docs/architecture.md) — Content OS workflow
- [Hermes Agent Skills](https://hermes-agent.nousresearch.com/docs)
