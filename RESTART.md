# Fleet Restart / Recovery Checkpoint

Updated: 2026-08-19 — post-restart recovery, reboot-safe hardening, NAS5→beelink.

## Hosts / topology
- **deathstar** (this box; Tailscale `100.78.106.121`): hosts the **bodycam KG**
  neo4j. Binds `0.0.0.0:7687` → reachable on `127.0.0.1:7687` (local) AND
  `bolt://100.78.106.121:7687` (Tailscale). LAN blocked by ufw `tailscale0` allow-rule.
  Manual `sudo neo4j start` after reboot. Data at `/var/lib/neo4j/data`.
- **x1-370** (Tailscale `100.64.43.123`; LAN `192.168.1.237`): hosts the **research KG**
  neo4j `bolt://127.0.0.1:7687` / `bolt://100.64.43.123:7687`. Runs the embed watcher.
  Also runs **LM Studio** at `http://192.168.1.237:1234` (local LAN model endpoint,
  for vision/RAG — not yet wired into the pipeline).
- **beelink** (Tailscale `100.85.72.121`; `beelink-ryzen-7-mini-pc`): **NAS5 SMB host.**
  Exports `//100.85.72.121/fileserver`. Both deathstar and x1-370 mount
  `/media/scott/NAS5` from it.
- **xwing**: GPU box; Tailscale SSH approval required for manual access (not in core pipeline).

## Current state (all GREEN)
- **Bodycam KG (deathstar):** Segment / Utterance / Transcription / Summary / Entity
  100% embedded with `emb_e5_large` (1024-d, multilingual-e5-large). ~256 nodes
  in-flight, caught up every 15 min by the watcher (normal incremental lag during ingest).
- **Research KG (x1-370):** Chunk + Entity 100% embedded (Chunk catch-up completed).
- **Cross-KG links:** 74,960 in `crosskg_links.sqlite` (bodycam Entity <-> research
  Entity, e5 cos sim >= 0.85). Saved in this repo (committed + pushed to `origin/main`)
  and `/home/deathstar/backups/crosskg/crosskg_links.sqlite`.
- **NAS5:** deathstar `/media/scott/NAS5` mounts `//100.85.72.121/fileserver` (beelink);
  `bodycam/`, `audio/`, `shared-knowledge/` all reachable. Credential file
  `/home/deathstar/.smbcredentials-x1-370` uses beelink password (was stale `scott`,
  fixed to `ghuhlaf8`).
- **Git:** clean tree, on `main`, pushed to `origin`.

## What auto-resumes after reboot (NO Tailscale SSH approval needed)
- **deathstar neo4j:** start manually — `sudo neo4j start`.
- **x1-370 watcher** (`/home/scott/embed_x1/run_watch_x1.sh`, `cron @reboot`):
  embeds new/missed nodes for BOTH KGs every 15 min.
  - Bodycam via **direct Tailscale TCP** `bolt://100.78.106.121:7687` (no SSH tunnel).
  - Research locally `bolt://100.64.43.123:7687`.
  - CPU-only (`CUDA_VISIBLE_DEVICES=-1`); model = multilingual-e5-large.
- (Removed) the deathstar `@reboot` reverse-tunnel cron — obsolete after the
  direct-connect hardening; the watcher no longer needs it.

## Manual steps after reboot
1. `sudo neo4j start` on deathstar (data persists; no restore needed).
2. Verify:
   - watcher alive: `ssh scott@100.64.43.123 'pgrep -af run_watch_x1.sh'`
   - bodycam reachable over Tailscale:
     `ssh scott@100.64.43.123 'timeout 8 bash -c "echo > /dev/tcp/100.78.106.121/7687" && echo bodycam-reachable'`
   - NAS5 mounted: `ls /media/scott/NAS5/bodycam` (auto-mounts on access; if dead,
     `sudo systemctl restart media-scott-NAS5.mount`).

## Known constraints (documented, not blockers)
- **x1-370 GPU:** `AutoModel.to("cuda")` deadlocks for e5-large (ROCm bug); all
  embedding runs CPU via `CUDA_VISIBLE_DEVICES=-1`. llama-server (Ornith-35B) owns
  the GPU and is unaffected.
- **Neo4j dump NOT taken:** store is 116G, only 83G free — a full
  `neo4j-admin database dump` will not fit on disk. The on-disk store at
  `/var/lib/neo4j/data` IS the backup and survives reboot (neo4j recovers via
  transaction logs). Take a dump later only with freed space / external drive:
  `sudo neo4j-admin database dump neo4j --to-path=<external>`.
- **xwing:** reachable only when Tailscale SSH approved; not required for the
  core pipeline (research-Chunk finished on x1-370 KG).
- **Tailscale SSH to x1-370/xwing** requires interactive browser approval
  (per-session check) — needed only for *manual* command launch; sustain no longer
  depends on it (direct Tailscale TCP).
- **LM Studio (x1-370):** `http://192.168.1.237:1234` is a local LAN model endpoint;
  integrate into vision/RAG when needed (not yet wired).

## Regenerate cross-KG links if ever needed
```
cd /home/deathstar/git/auto-ingest
python3 link_entities_crosskg.py \
  --kg-a-uri bolt://127.0.0.1:7687 --kg-b-uri bolt://100.64.43.123:7687 \
  --user neo4j --password knowledge_graph_2026 \
  --prop emb_e5_large --threshold 0.85 --topk 5 \
  --db crosskg_links.sqlite
```
(Needs `Entity_emb_e5_large_index` ONLINE in both KGs — it is.)
