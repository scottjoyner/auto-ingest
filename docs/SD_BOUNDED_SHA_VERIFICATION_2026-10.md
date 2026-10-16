# Bounded SD to NAS verification — October 16, 2026

**Status: evidence only; SD-card source release HOLD.** Continues draft PRs #28, #29 and #30.

## Modern fleet correction

PR #30 GitHub CI #314 failed two tests because the NAS indexer embedded a previous x1 removable-device mount prefix in source code. The indexer now obtains card mounts from UUID-bound campaign.json objects, and its staging root from the recorded historical witness. The frozen 4,268-file classification remains unchanged. The code no longer embeds the banned removable mount prefix.

## Live checks (x1-370, October 16)

- Two previously hashed Deathstar-alternate MP4 files were freshly SHA-256 hashed **on the NAS**: both match their historical source digests; 301,989,888 NAS bytes read. The SD originals were not rehashed in this check.
- First bounded source + NAS MP4 batch: **8/8 identical SHA-256**, stable before/after, 298,713,088 bytes read.
- Second bounded source + NAS MP4 batch: **8/8 identical SHA-256**, stable before/after, 931,135,488 bytes read, skipping the previously measured eight objects.
- Current source/NAS SHA pairs matched: **16/74 indexed candidate MP4 files**, leaving 58 candidate MP4 files untested. The two prior-witness MP4s were destination-only rechecked separately.

**None of these checks authorizes deletion.** All SD cards remain mounted read-write, so even matching SHA and stable metadata remain point-in-time witness evidence. The x1 CIFS NAS identity is not independently proven as the authoritative Beelink Btrfs archive and the required producer-quiescence and storage-retention gates remain open.
## Durable local evidence

- /home/scott/nas-reconciliation-audit-20261016/auto-ingest-pr30-two-video-nas-digest-recheck-20261016.json
- /home/scott/nas-reconciliation-audit-20261016/auto-ingest-eight-smallest-mp4-source-nas-sha-20261016.json
- /home/scott/nas-reconciliation-audit-20261016/auto-ingest-second-batch-eight-mp4-source-nas-sha-20261016.json

## Read-only command (repeatable with unique output filename)

    python3 -m auto_ingest.custody.verify_candidates \
      --index /home/scott/nas-reconciliation-audit-20261016/auto-ingest-nas-indexed-candidates-4268-media-classified-20261016.jsonl \
      --campaign-root /home/scott/nas-reconciliation-audit-20261016/auto-ingest-campaigns-v1 \
      --max-files 8 --max-read-bytes 350000000 \
      --out /home/scott/nas-reconciliation-audit-20261016/unique-new-evidence.json

For the next batch, pass --prior-report referencing the preceding saved report. It only skips stable matching pairs with the same candidate index and matching method; it does not establish live custody and cannot enable release. The source UUID is validated against the kernel, the candidate destination is confined under the NAS namespace, files are opened without following a final symlink, and a bounded SHA read records pre/post filesystem stability. Only local evidence files are created, never original SD/NAS media.

## Remaining gates

Reconcile additional indexed archive namespaces for the 3,619 MP4 names with no candidate in the limited index set. Verify the remaining 58 current candidates with bounded read-only source/destination comparisons. Obtain independent Beelink physical-Btrfs identity, snapshot/retention, and a quiescent read-only source observation before entering the separate custody-release validation. Preserve DCIM directories and sidecars. **No bulk copying, graph write, mount change, merge, deployment, or source unlink in this workstream.**
