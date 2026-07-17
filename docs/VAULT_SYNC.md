# Vault Sync — Canonical Model (G2)

> Implements gap **G2** from `docs/DEEP_DIVE_2026-07-14.md` §3.7 / `docs/GAP_ANALYSIS_2026-07-17.md`.
> Replaces the old 4-hardcoded-IP SSH-push design in `scripts/knowledge_sync_all.sh`.

## The model

There is **one** canonical vault: the mirror declared in `config.yaml` as
`knowledge_map.canonical_vault_path` (currently `/media/scott/NAS5/shared-knowledge`).

* **One writer.** Only `scripts/sync_vault_to_canonical.sh` writes to the
  canonical mirror. It rsyncs a *single, already-merged* local clone into the
  mirror and stamps `.last-canonical-sync`.
* **Many readers.** Every other host keeps a local clone
  (`knowledge_map.peer_vault_path`, e.g. `/home/scott/nas-knowledge`) and
  **pulls** from the canonical mirror. No host SSH-pushes its clone to another
  host — that was the old, conflict-prone behavior.
* **Hosts resolved by Tailscale hostname, never IP.** The peer list lives in
  `config.yaml` as `knowledge_map.vault_peers` (a list of hostnames). No raw
  IPs anywhere in the sync scripts.
* **Down machines don't break sync.** Before touching a peer the script runs a
  bounded `ssh` reachability probe (`knowledge_map.vault_sync.peer_timeout_sec`,
  default 8s). An unreachable peer is skipped and logged; the run continues.

## Peer list location

`config.yaml` → `knowledge_map.vault_peers` (list of Tailscale hostnames):

```yaml
knowledge_map:
  canonical_vault_path: /media/scott/NAS5/shared-knowledge
  peer_vault_path: /home/scott/nas-knowledge
  vault_peers:
    - deathstar
    - demo-1
    - destroyer
    - optiplex
  vault_sync:
    ssh_user: scott
    git_branch: main
    rsync_opts: "-a --delete"
    peer_timeout_sec: 8
    push_to_canonical: true
```

## How to add a peer

1. Add the Tailscale hostname to `knowledge_map.vault_peers` in `config.yaml`.
2. Ensure the host has a local clone at `knowledge_map.peer_vault_path` and is
   reachable over Tailscale SSH as `knowledge_map.vault_sync.ssh_user`.
3. That's it — the next sync run discovers it automatically (config-driven).

## How to remove a peer

Delete its hostname from `knowledge_map.vault_peers`. No script edits needed.

## Running

From the canonical-writer host (e.g. `x1-370`):

```bash
bash scripts/knowledge_sync_all.sh
```

What it does, in order:
1. Fetch each reachable peer's clone into the local working copy.
2. Merge (`git pull --rebase`).
3. Delegate the single authoritative write to
   `scripts/sync_vault_to_canonical.sh` (local clone → canonical mirror).
4. Pull the canonical mirror back down to each reachable peer.

`python3 -m knowledge_map sync_vault_to_neo4j` (the KG sync wrapper, a
separate task / K-N7) is intentionally **not** invoked here so this script
never depends on a missing module.

## Safety

* No hardcoded IPs (guarded by `tests/test_vault_sync_config.py` and a grep in CI).
* Unreachable peers are skipped, not fatal.
* Only one code path (`sync_vault_to_canonical.sh`) writes the mirror, preventing
  divergent mirrors.
