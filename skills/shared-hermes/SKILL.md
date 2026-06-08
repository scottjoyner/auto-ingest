---
name: shared-hermes
category: configuration
description: Shared .hermes data across all Tailscale machines
version: 1.0.0
author: scottjoyner
created: 2026-06-03
---

# Shared Hermes Data Skill

Centralized storage for hermes-agent data (chats, skills, configs) accessible from all Tailscale machines.

## Overview

This skill manages a shared `.hermes` directory structure on the NAS that all agents can access:

**Shared Location:** `/media/scott/S/shared-hermes/`

**Directory Structure:**
```
/media/scott/S/shared-hermes/
├── chats/          # Chat history (SQLite databases)
├── skills/         # Shared skills repository
├── config/         # Configuration files (config.yaml, .env)
└── logs/           # Centralized agent logs
```

## Machine-Specific vs Shared Data

| File/Folder | Location | Reason |
|-------------|----------|--------|
| `chats/*.db` | Shared | Chat history accessible from all machines |
| `skills/` | Shared | All agents use same skills repository |
| `config/config.yaml` | Per-machine | Machine-specific settings (model, API keys) |
| `config/.env` | Per-machine | Sensitive credentials per machine |
| `logs/agent.log` | Local | Performance - local log files |

## Symlink Setup

Each machine creates symlinks from its home `.hermes/` to the shared location:

```bash
# On each machine (run once):
mkdir -p ~/.hermes
ln -sf /media/scott/S/shared-hermes/chats ~/.hermes/chats
ln -sf /media/scott/S/shared-hermes/skills ~/.hermes/skills
ln -sf /media/scott/S/shared-hermes/config ~/.hermes/config

# Per-machine config (optional):
mkdir -p ~/.hermes/config-local
cp ~/.hermes/config/config.yaml ~/.hermes/config-local/
```

## Slash Commands

### `/shared-hermes setup`

Set up symlinks on the current machine.

**Examples:**
```bash
/shared-hermes setup           # Create all symlinks
/shared-hermes setup --force   # Overwrite existing symlinks
```

### `/shared-hermes status`

Show which data is shared vs local on this machine.

**Examples:**
```bash
/shared-hermes status          # Show current symlink status
/shared-hermes status --detailed  # Show file sizes and counts
```

### `/shared-hermes migrate`

Migrate existing `~/.hermes/` data to shared location.

**Examples:**
```bash
/shared-hermes migrate         # Move all data to shared storage
/shared-hermes migrate chats   # Migrate only chat history
```

## Configuration Files

### Per-Machine Config (`~/.hermes/config-local/config.yaml`)

Machine-specific settings that override shared config:

```yaml
# Machine identifier
machine_name: deathstar-XPS-8920
tailscale_ip: 100.78.106.121

# Model routing (can differ per machine)
default_model: qwen3.5-35b-a3b-uncensored-hauhaucs-aggressive
provider: lmstudio

# Local overrides
display:
  theme: dark
  spinner: kawaii
```

### Shared Config (`/media/scott/S/shared-hermes/config/config.yaml`)

Common settings for all machines:

```yaml
# Global settings
version: 21

# Default model (can be overridden per-machine)
default_model: qwen3.5-35b-a3b-uncensored-hauhaucs-aggressive
provider: lmstudio

# Toolsets enabled by default
enabled_toolsets:
  - terminal
  - file
  - web
  - browser
  - vision
```

## Sync Workflow

### When Agent Updates Data:

1. **Skills**: Changes to `~/.hermes/skills/` are shared immediately
2. **Chats**: SQLite databases are shared (may need locking)
3. **Config**: Per-machine config overrides shared defaults

### Daily Sync via Cron:

```bash
# On x1-370, sync shared .hermes to all machines
0 3 * * * /home/scott/git/auto-ingest/scripts/sync_hermes_data.sh >> /home/scott/logs/hermes_sync.log 2>&1
```

## Files Created/Modified

| File | Purpose |
|------|---------|
| `skills/shared-hermes/SKILL.md` | This documentation |
| `scripts/sync_hermes_data.sh` | Sync script for all machines |
| `/media/scott/S/shared-hermes/` | Shared data location on NAS |

## Troubleshooting

### "Permission denied" when accessing shared .hermes:

```bash
# Fix permissions on NAS:
chmod -R 755 /media/scott/S/shared-hermes
chown -R scott:scott /media/scott/S/shared-hermes
```

### SQLite database locked during chat sync:

```bash
# Use WAL mode for better concurrency:
sqlite3 ~/.hermes/chats/session.db "PRAGMA journal_mode=WAL;"
```

### Config not loading correctly:

```bash
# Check symlink is valid:
ls -la ~/.hermes/config

# Verify config.yaml exists:
cat ~/.hermes/config/config.yaml
```

## Version History

- **v1.0.0 (2026-06-03)**: Initial implementation with shared storage and per-machine overrides
