---
name: knowledge-sync
category: data-science
description: Multi-machine knowledge base sync with git merge and Neo4j integration
version: 1.0.0
author: scottjoyner
created: 2026-06-03
---

# Knowledge Sync Skill

Bidirectional sync system between multiple machines, git-based conflict resolution, and Neo4j knowledge graph indexing.

## Overview

This skill manages synchronization of the knowledge vault across all Tailscale machines:
- **x1-370** (master source): `/media/scott/S/shared-knowledge/`
- **deathstar**: `~/knowledge/nas-knowledge/`
- **demo-1**: `~/nas-knowledge/`
- **destroyer**: `~/nas-knowledge/`
- **optiplex**: `~/nas-knowledge/`

## Slash Commands

### `/knowledge-sync [machine]`

Trigger a full knowledge sync from x1-370 to all machines.

**Arguments:**
- `[machine]` (optional): Specific machine to sync from/to
  - If omitted: Sync from x1-370 master to all nodes
  - If specified: Push changes from that machine to master

**Examples:**
```bash
/knowledge-sync              # Full sync from x1-370 to all machines
/knowledge-sync deathstar    # Push deathstar changes to master
/knowledge-sync --force      # Force overwrite, skip conflicts
```

### `/knowledge-neo4j`

Run Neo4j indexing on the local vault using knowledge_map.

**Examples:**
```bash
/knowledge-neo4j             # Index current vault to Neo4j
/knowledge-neo4j --dry-run   # Preview changes without applying
```

### `/knowledge-conflicts`

List and report any git merge conflicts in the knowledge base.

**Examples:**
```bash
/knowledge-conflicts         # Show conflicted files
/knowledge-conflicts resolve # Attempt auto-resolution
```

## Sync Workflow

### 1. Git-Based Multi-Machine Sync

Each machine maintains its own git repository with a `main` branch:

```bash
# On each non-master machine (cron job or manual):
cd ~/knowledge/nas-knowledge/
git remote add master ssh://scott@x1-370:/media/scott/S/shared-knowledge/.git 2>/dev/null || true
git fetch master
git merge master --no-edit

# If conflicts exist:
git mergetool  # Resolve manually
git add . && git commit -m "Sync from $(hostname)"
git push master main
```

### 2. Master Aggregation (x1-370)

On x1-370, the master sync script aggregates all changes:

```bash
# Run daily via cron or manually:
/home/scott/git/auto-ingest/scripts/knowledge_sync_all.sh
```

This script:
1. Fetches from all machines
2. Merges branches with conflict detection
3. Runs Neo4j indexing via knowledge_map
4. Pushes updates to all nodes

### 3. Conflict Resolution

When files differ between machines, git will flag conflicts:

**Detected conflicts are logged to:**
```
/media/scott/S/shared-knowledge/.sync-conflicts-YYYYMMDD-HHMMSS.md
```

**Resolution steps:**
1. Review conflicted files in the conflict report
2. Run `git mergetool` on x1-370
3. Choose appropriate version or merge manually
4. Commit resolved changes: `git add . && git commit -m "Resolved sync conflicts"`

## Cron Job Setup

### Daily Sync (Recommended)

Add to x1-370 crontab (`crontab -e`):

```bash
# Run knowledge sync daily at 2 AM
0 2 * * * /home/scott/git/auto-ingest/scripts/knowledge_sync_all.sh >> /home/scott/logs/knowledge_sync.log 2>&1
```

### Per-Machine Push (Optional)

Add to each non-master machine crontab:

```bash
# Push changes every hour
0 * * * * cd ~/knowledge/nas-knowledge && git add . && git commit -m "Auto sync" && git push master main 2>/dev/null || true
```

## Neo4j Integration

The skill integrates with the existing `knowledge_map` module:

**Location:** `/home/scott/git/auto-ingest/`

**Key scripts:**
- `sync_engine.py`: Core sync logic (neo4j-to-vault, vault-to-neo4j)
- `config.yaml`: Neo4j connection settings
- Templates per node type in `src/knowledge_map/templates/`

**Usage:**
```bash
# Index vault to Neo4j:
python3 -m knowledge_map sync_vault_to_neo4j --config config.yaml

# Export Neo4j to vault:
python3 -m knowledge_map sync_neo4j_to_vault --config config.yaml
```

## Manual Trigger from Agents

Agents can trigger syncs via slash commands during their workflow:

**Example agent workflow:**
1. Agent updates knowledge file on local machine
2. Agent runs `/knowledge-sync` to push changes
3. System handles git merge and Neo4j indexing automatically
4. Changes propagate to all machines

## Files Created/Modified

| File | Purpose |
|------|---------|
| `scripts/knowledge_sync_all.sh` | Master sync script (x1-370) |
| `skills/knowledge-sync/SKILL.md` | This documentation |
| `~/.hermes/skills/knowledge-sync/` | Installed skill location |

## Troubleshooting

### Common Issues

**"Permission denied" on SSH:**
```bash
# Ensure SSH keys are deployed:
ssh-copy-id scott@<machine-ip>
```

**Git merge conflicts persist:**
```bash
# Check conflicted files:
git status --porcelain | grep "^??"

# Resolve with mergetool:
git mergetool
```

**Neo4j sync fails:**
```bash
# Verify connection:
python3 -c "from neo4j import GraphDatabase; driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', '${NEO4J_PASSWORD}')); print(driver.verify_connectivity())"

# Check config.yaml exists and has correct settings
```

**Sync not running on schedule:**
```bash
# Check cron jobs:
crontab -l | grep knowledge_sync

# View logs:
tail -f /home/scott/logs/knowledge_sync.log
```

## Version History

- **v1.0.0 (2026-06-03)**: Initial implementation with git-based multi-machine sync and Neo4j integration
