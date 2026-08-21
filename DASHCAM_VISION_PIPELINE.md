# Dashcam Vision RAG Pipeline

Batch-describes dashcam clips with a vision LLM, stores per-minute frames as
`DashcamFrame` nodes in the **bodycam** Neo4j KG, embeds the descriptions into
`emb_e5_large` (multilingual-e5-large), and makes them semantically retrievable.

## Data sources (IMPORTANT — two archives)
Clips live on **two** mounts, not one:
- `/mnt/8TB_2025/fileserver/dashcam` — ~26,966 files (2025×60, 2026×26,906)
- `/mnt/8TBHDD/fileserver/dashcam` — ~52,208 files (older archive)

The orchestrator walks **both** (see `run_dashcam_vision.sh`). `clips_on_disk` in
`PipelineHealth` counts both. Total source ≈ **79k clips**. Process this as one
pipeline; do not point the orchestrator at only one mount or you will silently
skip ~2/3 of the footage.

## Architecture
- **Frame extraction**: `ffmpeg` (robust on corrupt clips; `cv2` segfaults/hangs on a
  meaningful share of source files, so `cv2` is only used for resize/encode).
- **Vision**: MacBook Air Tailscale node `100.85.64.117:1234`, LM Studio model
  `qwen3.5-0.8b-mlx` (~14–25s/frame). This is the only vision model in the fleet;
  keep the MacBook Air **awake with the model loaded** (disable LM Studio idle-unload).
- **Storage**: `dashcam_frame_vision.py` → `DashcamFrame {key,minute,view,description,
  t_sec,timestamp,model,created}` with `DashcamClip -[:HAS_FRAME]-> DashcamFrame`.
- **Embedding**: `reembed.py DashcamFrame --model intfloat/multilingual-e5-large --prop
  emb_e5_large` (run loop on deathstar).
- **Retrieval**: vector index `DashcamFrame_emb_e5_large_index` (cosine). Embed queries
  with the same e5-large model (no `query:`/`passage:` prefix — matches ingest).

## Run
The pipeline is managed by systemd and **auto-starts on boot**, waits for neo4j and the
MacBook Air endpoint, then processes both mounts continuously:

    sudo systemctl status dashcam-vision dashcam-monitor
    sudo journalctl -u dashcam-vision -f

Manual one-shot (single mount):
```
python3 dashcam_frame_vision.py --base /mnt/8TB_2025/fileserver/dashcam \
    --orchestrate --max-minutes 5 --view both --sleep 0.1 \
    --timeout 120 --clip-timeout 500 --vision-retries 4 --workers 2
```
`--workers N` runs N clip subprocesses in parallel (default 1). Already-done clips are
skipped; a clip that crashes/hangs is marked `failed` and skipped on re-runs.

## Data-quality tracking (DB side)
Source clips are of **mixed integrity** — a meaningful share are decode-corrupt
(`ffmpeg` yields 0 frames). This is tracked, not silently dropped:

- `DashcamClip.failed = true` with `failure_reason`:
  - `decode_failed` — ffmpeg could not extract any frame (corrupt/truncated source).
  - `timeout_or_hang` — clip subprocess exceeded `clip_timeout`.
  - `unrecoverable` — salvage re-mux/re-encode also failed (permanently skipped).
  - (legacy clips may have `failed=true` with `failure_reason` NULL — pre-dates tracking).
- `DashcamClip.processed = true` — clip successfully described at least once.
- `DashcamClip.recovered = true` — clip was rescued by the salvage pass.
- `DashcamClip.path` — absolute source file path (set on every clip).
- `PipelineHealth {name:'dashcam_vision'}` — live rollup written every 30s by
  `dashcam_monitor.py`: `total_frames`, `embedded`, `processed_clips`, `failed_clips`,
  `clip_nodes`, `clips_on_disk`, `clips_on_disk_breakdown`, `updated`.

### Useful queries
```cypher
// live health
MATCH (h:PipelineHealth {name:'dashcam_vision'}) RETURN h

// integrity tally
MATCH (c:DashcamClip {failed:true}) RETURN c.failure_reason AS reason, count(c) AS n

// sample corrupt clips for manual review
MATCH (c:DashcamClip {failed:true}) RETURN c.key, c.failure_reason, c.path LIMIT 50

// coverage
MATCH (c:DashcamClip)
RETURN count(c) AS clip_nodes,
       count(c {processed:true}) AS described,
       count(c {failed:true})   AS corrupt
```

## Corrupt-clip salvage (`dashcam_salvage.py`)
Many "decode_failed" clips are re-muxable. The salvage pass re-muxes (and, as a
fallback, re-encodes) each failed clip to a clean container and retries extraction +
description. Recovered clips are stored and their `failed` flag is cleared
(`recovered=true`). Clips that still fail are marked `unrecoverable` so they are not
retried forever. The wrapper runs this automatically after each orchestration pass.

    python3 dashcam_salvage.py --base /mnt/8TB_2025/fileserver/dashcam --limit 10

## KG linking (`dashcam_link.py`)
Each clip key encodes its timestamp (`2026_0709_203536_R` → 2026-07-09 20:35:36).
`DashcamClip` is linked `(DashcamClip)-[:ON_DAY]->(DashcamDay {date,year,month,day})`,
and `DashcamClip.timestamp` / `DashcamFrame.timestamp` are set, enabling time-based
queries ("clips from the night of July 9 2026"). The orchestrator links new clips as
they are stored; `dashcam_link.py --only-unlinked` backfills the rest (idempotent).

## Semantic search (`dashcam_ask.py`)
```
python3 dashcam_ask.py "police car pulled over on the shoulder at night" --top-k 5
python3 dashcam_ask.py "rain on the windshield" --view R --json
```
Embeds the query with e5-large and runs a vector search against
`DashcamFrame_emb_e5_large_index`, returning key, view, minute, timestamp, source path,
and description.

## Research KG backup → NAS5 (FIXED)
The research Neo4j (voice KG: `KgNode`, `VoiceIdentity`, `VoiceprintGroup`, …) runs in
Docker on `x1-370` (`100.64.43.123`, image `neo4j:5-enterprise`, DBs: `neo4j`,
`assistx`, `system`). Online backups are taken **inside** the container via its backup
listener (`localhost:6362`):
```
docker exec neo4j bash -c "mkdir -p /tmp/neo4j-bk && \
  for db in neo4j assistx system; do \
    neo4j-admin database backup --from=localhost:6362 --to-path=/tmp/neo4j-bk \$db >/dev/null; \
  done"
sudo docker cp neo4j:/tmp/neo4j-bk /home/scott/research_bk   # on x1-370 host
```
Pull to deathstar and drop on **NAS5** (beelink SMB share `//100.85.72.121/fileserver`,
mounted at `/media/scott/NAS5`):
```
rsync -av --partial /home/scott/research_bk/ /media/scott/NAS5/backups/neo4j/x1-370/
```
**Location of the research backup: `/media/scott/NAS5/backups/neo4j/x1-370/`**
(`neo4j`, `assistx`, `system` full snapshots). A repeatable script
`/home/scott/backup_research_kg.sh` + nightly cron (`17 3 * * *`) exists on x1-370.
A redundant copy also lives on NAS3 (`/media/scott/NAS3/backups/neo4j/x1-370/`).

**NAS5 was previously broken** — beelink's 81 TB `exFAT` volume (`/dev/sda2`) had been
left uncleanly unmounted during the NAS migration (`dmesg: exFAT-fs … Volume was not
properly unmounted … corrupt … run fsck`). This made writes crawl at ~1–2 MB/s while reads
were fine, causing SMB clients to time out with I/O errors. **Fixed by running
`fsck.exfat -y /dev/sda2` on beelink** (after stopping smbd + unmounting); post-repair
local writes are ~600 MB/s and SMB writes from deathstar ~100 MB/s. If NAS5 writes ever
stall again, run `fsck.exfat` on beelink (`sudo systemctl stop smbd; sudo umount
/media/scott/NAS5; sudo fsck.exfat -y /dev/sda2; sudo mount /media/scott/NAS5; sudo
systemctl start smbd`). Mount NAS5 with `soft` so a future stall fails fast instead of
hanging the shell. This also unblocked the bodycam KG mirror to NAS5 (see two-tier backup
below).

## Bodycam KG backup (two-tier → NAS5)
The bodycam Neo4j (`neo4j` DB, backup connector `localhost:6362`) is backed up in two
independent tiers so a network/CIFS failure can never fill the root SSD or wedge the
pipeline:
- **Tier 1 — primary (cron `27 2 * * *`)**: `backup_bodycam_kg.sh` runs
  `neo4j-admin database backup --from=localhost:6362` to the **LOCAL HDD**
  `/mnt/8TB_2025/backups/neo4j/deathstar/bodycam-<ts>/` (~2.9 TB free). It guards for
  ~130 GB free, takes a `flock`, and aborts cleanly if the HDD is low. Latest snapshot
  ~31 GB. No root-disk or NAS5 staging — ever.
- **Tier 2 — cold mirror (cron `27 3 * * *`)**: `mirror_bodycam_to_nas5.sh` rsyncs the
  latest `bodycam-<ts>/` from the HDD to **NAS5**
  (`/media/scott/NAS5/backups/neo4j/deathstar/`). It is independent of tier 1 and **skips
  gracefully** if NAS5 is unmounted or has <~200 GB free, so a stale CIFS mount can never
  break the primary tier. Retention: keeps the last 3 snapshots on NAS5.

A historical full snapshot is also kept: `neo4j-2026-08-19T11-34-00.backup` on NAS5.
Verified current state: `bodycam-2026-08-21T10-48-44/neo4j-2026-08-21T15-00-30.backup`
(31.1 GB) present on **both** the HDD and NAS5.

### NAS5 mount hardening (fstab)
CIFS to beelink over Tailscale goes **stale** — clients then show phantom/empty dirs and
`No such device`, and a cached `ls` can make a backup look "missing" when it actually
landed. The NAS5 fstab entry is hardened to fail fast and self-heal:
`soft,retrans=10,echo_interval=60` + `_netdev` (mount at boot). `x-systemd.automount` was
removed because the autofs trigger wedge-cycles on this box; use the plain `.mount` unit.
After any suspected staleness, remount to get a truthful view **before** concluding a
backup did/didn't land:
```
sudo systemctl stop media-scott-NAS5.automount 2>/dev/null
sudo umount -l /media/scott/NAS5; sudo systemctl daemon-reload
sudo systemctl start media-scott-NAS5.mount
ls /media/scott/NAS5/backups/neo4j/deathstar/   # re-list on the fresh mount
```

## Caveats
- **Legacy false-failures**: an earlier cv2-based extractor marked a large batch of clips
  `decode_failed` even though `ffmpeg` extracts them fine. These stale flags were bulk-cleared
  (`failure_reason='decode_failed'` → reprocessed) and the running pipeline is recovering
  them via the ffmpeg path. After clearing, `failed_clips` should sit at 0.
- **MacBook Air availability** is the throughput bottleneck. If it sleeps/unloads, vision
  calls transiently fail (auto-retried); keep it awake + model loaded for sustained runs.
- Decode-corrupt clips are permanently skipped (`failed`/`unrecoverable`); re-run does not
  reprocess them unless `failed` is cleared.
- Per-clip frame count is capped at `--max-minutes 5` (first frame of each minute, ≤6
  frames/clip) for batch sustainability.
- The research KG CIFS write to NAS5 is slow; budget minutes for the copy step.
