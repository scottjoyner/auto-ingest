# Fleet Restart / Recovery Checkpoint

Generated: 2026-08-18 — state at "good stopping place" before fleet restart.

## Current state (all GREEN)
- **Bodycam KG (deathstar, bolt://127.0.0.1:7687):** Segment, Utterance, Entity,
  Transcription, Summary all 100% embedded with `emb_e5_large` (1024-d, multilingual-e5-large).
- **Research KG (x1-370, bolt://100.64.43.123:7687 over Tailscale / 127.0.0.1:7687 on x1-370):**
  Chunk + Entity 100% embedded.
- **Cross-KG entity links:** 74,960 links in `crosskg_links.sqlite`
  (bodycam Entity <-> research Entity, e5 cos sim >= 0.85). Saved in:
  - this repo (committed + pushed to `origin/main`)
  - local copy: `/home/deathstar/backups/crosskg/crosskg_links.sqlite`
- **Git:** all fixes on `main`, pushed to `origin`. Clean tree.

## What auto-resumes after restart
- **x1-370 watcher** (`/home/scott/embed_x1/run_watch_x1.sh`, CPU-only):
  relaunches via `cron @reboot` on x1-370. It sustains BOTH KGs
  (bodycam via the tunnel below; research locally) and embeds any new/missed nodes.
- **deathstar reverse tunnel:** `cron @reboot` on deathstar runs
  `ssh -N -R 17687:127.0.0.1:7687 scott@100.64.43.123` so x1-370's watcher can
  reach deathstar's neo4j (bodycam) post-reboot. (deathstar neo4j binds 127.0.0.1 only.)

## Manual steps after reboot
1. **Start deathstar neo4j** (manual, needs sudo): `sudo neo4j start`
   Data persists at `/var/lib/neo4j/data` — no restore needed.
2. x1-370 + deathstar cron @reboot handle the rest. Verify with:
   `ssh scott@x1-370 'pgrep -af run_watch_x1.sh'` and
   `ssh scott@x1-370 '(echo > /dev/tcp/127.0.0.1/17687) && echo tunnel-up'`.

## Known constraints (documented, not blockers)
- **x1-370 GPU:** `AutoModel.to("cuda")` deadlocks for e5-large (ROCm bug); all
  embedding runs CPU via `CUDA_VISIBLE_DEVICES=-1`. llama-server (Ornith-35B)
  owns the GPU and is unaffected.
- **Neo4j dump NOT taken:** store is 116G, only 83G free — a full
  `neo4j-admin database dump` will not fit on disk. The on-disk store at
  `/var/lib/neo4j/data` IS the backup and survives reboot (neo4j recovers via
  transaction logs). Take a dump later only with freed space / external drive:
  `sudo neo4j-admin database dump neo4j --to-path=<external>`.
- **xwing:** reachable only when Tailscale SSH approved; not required for the
  above (research-Chunk finished on x1-370 KG).
- **Tailscale SSH to x1-370/xwing** requires interactive browser approval
  (per-session check) — needed only for manual command launch.

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
