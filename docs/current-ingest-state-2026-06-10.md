# Auto-ingest / Neo4j current state — 2026-06-10 11:01 EDT

This document is a guardrail for future agents. Do not assume the auto-ingest stack is correctly scanning all NAS media just because containers are up.

## Executive summary

Neo4j has been recovered to the large historical graph, but auto-ingest is currently pointed at the wrong mounted NAS tree for the major media corpus.

Current running auto-ingest containers are alive, but they are mostly reprocessing a tiny NAS1 subset, not the canonical NAS2 dashcam/audio corpus.

Critical fact:
- Docker Compose mounts `NAS_ROOT=/media/scott/NAS1` as `/nas`.
- The running scan config uses paths like `/nas/fileserver/dashcam` and `/nas/fileserver/audio`.
- On the host, `/media/scott/NAS1/fileserver/audio` exists but is small.
- On the host, `/media/scott/NAS1/fileserver/dashcam` does not exist.
- The large canonical corpus is on `/media/scott/NAS2/fileserver/...`, especially `/media/scott/NAS2/fileserver/dashcam` and `/media/scott/NAS2/fileserver/audio`.

Therefore: current auto-ingest is NOT ingesting the big dashcam/audio NAS corpus correctly.

## Running services checked

Repo: `/home/scott/git/auto-ingest`

Docker Compose services were up:
- `auto-ingest-service`
- `auto-ingest-worker`
- `auto-sync-service`
- `auto-ingest-job-api`
- `content-generation-service`

Relevant compose/env state:
- `.env`: `NAS_ROOT=/media/scott/NAS1`
- Container mount: `${NAS_ROOT}:/nas`
- `SCAN_ROOTS=/nas/S/audio,/nas/fileserver/audio,/nas/fileserver/audio/transcriptions,/nas/fileserver/bodycam,/nas/fileserver/dashcam,/nas/fileserver/dashcam/transcriptions`
- `DASHCAM_ROOT=/nas/fileserver/dashcam`
- `AUDIO_ROOT=/nas/S/audio`
- `NEO4J_URI=bolt://host.docker.internal:7687`
- `NEO4J_DB=neo4j`

`auto-ingest-service` command:
- `while true; do /app/run_ingest_all.sh; sleep 300; done`

`auto-ingest-worker` status:
- repeatedly logs `No job files found in /nas/drop`.

Recent ingest logs:
- `/home/scott/git/auto-ingest/logs/ingest_20260610_145614.log`
- `/home/scott/git/auto-ingest/logs/ingest_20260610_145055.log`
- `/home/scott/git/auto-ingest/logs/ingest_20260610_144534.log`

Recent run behavior:
- Found only 311 keys.
- Finished in about 14-18 seconds per loop.
- Example final line: `ALL DONE | elapsed=13.60s | processed=18 skipped=122 total=311`.
- This is far too small for the NAS2 corpus and confirms the service is not scanning the large dashcam/audio tree.

## Filesystem inventory checked

Host paths actually present:

### NAS1, currently mounted into auto-ingest container as `/nas`

`/media/scott/NAS1/fileserver/audio`:
- exists
- 375 files
- about 22.3 GB
- 293 `.wav`
- 20 `.webm`
- 24 `.csv`
- 24 `.txt`
- 14 `.json`
- newest mtime around 2026-06-10 10:45:53

`/media/scott/NAS1/fileserver/bodycam`:
- exists as directories only
- 133 dirs
- 0 files

`/media/scott/NAS1/fileserver/dashcam`:
- does NOT exist

`/media/scott/NAS1/S/audio`:
- does NOT exist

`/media/scott/NAS1/fileserver/incoming/deathstar`:
- does NOT exist

### NAS2, apparent canonical corpus

`/media/scott/NAS2/fileserver/audio`:
- exists
- 301,614 files
- about 188 GB
- 38,604 `.mp3`
- 1,655 `.wav`
- 15 `.webm`
- 102,002 `.json`
- 59,476 `.txt`
- 52,255 `.csv`
- 39,733 `.rttm`
- newest mtime around 2026-06-09 07:42:44

`/media/scott/NAS2/fileserver/dashcam`:
- exists
- 299,633 files
- about 15.6 TB
- 111,387 `.mp4`
- 104,275 `.csv`
- 78,889 `.png`
- 4,285 `.txt`
- 747 `.json`
- 34 `.avi`
- newest mtime around 2026-06-10 01:33:42

## Neo4j current coverage checked

Live Neo4j password used for checks: `knowledge_graph_2026`.

Relevant label counts:
- `PhoneLog`: 20,272,010
- `Transcription`: 64,108
- `Segment`: 361,734
- `Utterance`: 419,672
- `Speaker`: 233,184
- `Entity`: 14,761
- `DashcamClip`: 71,590
- `DashcamEmbedding`: 4,266,109
- `Frame`: 3,722,710
- `Audio`: 1
- `Transcript`: 1
- `SophiaCapture`: 1
- `Device`: 2

Transcription path-kind counts:
- dashcam paths: 26,619
- audio paths: 37,233
- other/blank: 256

PhoneLog range after recovery:
- min: 2024-08-20T20:26:39Z
- max: 2026-06-10T14:24:34Z

Known PhoneLog gap still present:
- 2026-05-20T14:25:15Z to 2026-06-06T23:59:03Z has no PhoneLog rows in the recovered DB.
- This was not recovered from the selected Neo4j backups and likely needs raw-source/auto-ingest backfill.

## Dashcam coverage comparison

Compared DB `DashcamClip.key` against host files in `/media/scott/NAS2/fileserver/dashcam`.

DB dashcam clips:
- 71,590 `DashcamClip` nodes

NAS2 dashcam videos:
- 111,421 video files counted by extension (`.mp4`, `.avi`, `.mov`, `.mkv`)
- 111,203 unique video stems/keys

Exact key comparison:
- FS unique keys: 111,203
- DB keys: 71,590
- FS keys missing in DB: 44,869
- DB keys not found on FS by exact key: 5,256

Normalized base comparison with trailing `_F` / `_R` / `_L` / `_I` view suffix removed:
- FS base keys: 76,227
- DB base keys: 36,740
- FS base keys missing in DB: 42,115
- DB base keys not found on FS: 2,628

Recent year/month comparison:

DB `DashcamClip` by key month:
- 2026-01: 1,245
- 2026-02: 1,149
- 2026-03: 2,547
- 2026-04: 1,581
- 2026-05: 0
- 2026-06: 0

NAS2 dashcam videos by key month:
- 2026-01: 1,867
- 2026-02: 1,209
- 2026-03: 3,227
- 2026-04: 3,128
- 2026-05: 4,075
- 2026-06: 790

Conclusion: dashcam DB ingestion appears to stop at April 2026. May/June 2026 dashcam videos are present on NAS2 but not represented as `DashcamClip` nodes.

## Audio/transcript coverage comparison

NAS2 audio media:
- 40,277 media files total
- 38,604 `.mp3`
- 1,655 `.wav`
- 15 `.webm`
- 2 `.mp4`
- 1 `.m4a`

NAS2 audio transcript artifacts:
- 127,858 transcript/RTTM/VTT/SRT/TSV-ish artifacts counted under `/media/scott/NAS2/fileserver/audio`

NAS1 audio media currently visible to running auto-ingest:
- 313 media files total
- 293 `.wav`
- 20 `.webm`

NAS1 audio transcript artifacts currently visible to running auto-ingest:
- 48 transcript-ish artifacts under `/media/scott/NAS1/fileserver/audio`

Conclusion: the running auto-ingest loop is pointed at a small NAS1 subset, not the large NAS2 audio/transcription corpus.

## What auto-ingest is poised to do right now

If left as-is, `auto-ingest-service` will continue to run every 5 minutes and process about 311 discovered keys from the small mounted NAS1 tree. It will mostly:
- ingest/re-ingest a handful of Sophia voice/transcription artifacts under `/nas/fileserver/audio/...`
- skip media files that have no transcription data
- not see `/media/scott/NAS2/fileserver/dashcam`
- not see `/media/scott/NAS2/fileserver/audio`
- not ingest May/June 2026 dashcam video/metadata from NAS2
- not backfill the large NAS2 audio/transcript corpus

## Important implementation caution

`run_ingest_all.sh` previously had a bug/placeholder in its default root handling where the literal string `FILESERVER_ROOT` was not expanded. That has now been fixed so the script resolves `FILESERVER_ROOT` through `auto_ingest_config.py` when explicit env vars are absent.

Still, in Docker this is usually masked because `.env` provides `SCAN_ROOTS` and `DASHCAM_ROOT`. Prefer explicit paths/env vars for backfills, and use `scripts/backfill_audio_day.sh` for one-day audio batches instead of changing the continuous ingest loop to scan both full NAS trees.

## Safe next steps / applied guardrails

Do not blindly run a huge FORCE ingest. The safe sequence is now:

1. Keep the existing `/nas` mount behavior stable until an intentional restart/change window.
2. Use the added dual mount variables instead of overloading `NAS_ROOT`:
   - `.env`: `NAS1_ROOT=/media/scott/NAS1`
   - `.env`: `NAS2_ROOT=/media/scott/NAS2`
   - Compose mounts these as `/nas1` and `/nas2` anywhere `/nas` is also mounted.
3. Run audio backfill one day at a time with `scripts/backfill_audio_day.sh`.
   - This script scans only `audio/YYYY/MM/DD` and `audio/transcriptions/YYYY/MM/DD` for the requested day under both NAS1 and NAS2.
   - It sets `RTTM_DIRS` to the same day roots so RTTM discovery does not walk the whole audio corpus.
   - It disables dashcam metadata roots for this audio-only pass.
   - Start with `--dry-run --limit 5`; remove `--dry-run` only when the roots and key count look sane.
4. QA each day with `scripts/qa_audio_day.sh YYYY-MM-DD` before moving to the next day.
   - Check Transcription/Segment/Utterance counts.
   - Critical audio vector check: `segments_missing_embedding` and `utterances_missing_embedding` should be zero or explained.
5. Dashcam stays separate. Run `dashcam_yolo_embeddings.py` later for `DashcamClip`, `DashcamEmbedding`, and `Frame` coverage; do not mix it into the audio embedding backfill.
6. Backfill the PhoneLog gap separately from raw phone-log/SMS sources; the current media ingest loop does not appear to be the PhoneLog importer.
7. After any large backfill batch, take a fresh Neo4j backup to NAS1/NAS2.

Example audio batch commands (use Compose image on this host; local `/usr/bin/python3` does not currently have torch):

```bash
cd /home/scott/git/auto-ingest
scripts/backfill_audio_day.sh 2026-05-31 --print-roots
docker compose run --rm --no-deps ingest-service bash -lc '/app/scripts/backfill_audio_day.sh 2026-05-31 --dry-run --limit 5'
docker compose run --rm --no-deps ingest-service bash -lc '/app/scripts/backfill_audio_day.sh 2026-05-31'
scripts/qa_audio_day.sh 2026-05-31
```

## Do-not-hallucinate rules for future agents

- Do not claim auto-ingest is correctly ingesting all NAS dashcam/audio just because containers are `Up`.
- Do not claim `/media/scott/NAS1/fileserver/dashcam` exists; it did not exist during this check.
- Do not claim May/June 2026 dashcam is represented in Neo4j; `DashcamClip` month counts were 0 for 2026-05 and 2026-06.
- Do not assume `run_ingest_all.sh` performs Whisper transcription of raw media. The observed behavior skips media without transcription data.
- Do not assume PhoneLog recovery is handled by the dashcam/audio ingest loop.
- Do not run `FORCE=1` against the whole corpus without explicit user approval.
