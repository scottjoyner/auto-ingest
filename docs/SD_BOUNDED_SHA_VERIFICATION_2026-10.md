# Bounded SD to NAS verification — October 16, 2026

**Status: evidence only; SD-card source release HOLD.** Continues draft PRs #28, #29 and #30.

## Modern fleet correction

PR #30 GitHub CI #314 failed two tests because the NAS indexer embedded a previous x1 removable-device mount prefix in source code. The indexer now obtains card mounts from UUID-bound campaign.json objects, and its staging root from the recorded historical witness. The frozen 4,268-file classification remains unchanged. The code no longer embeds the banned removable mount prefix.

## Live checks (x1-370, October 16)

- Two previously hashed Deathstar-alternate MP4 files were freshly SHA-256 hashed **on the NAS**: both match their historical source digests; 301,989,888 NAS bytes read. The SD originals were not rehashed in this check.
- First bounded source + NAS MP4 batch: **8/8 identical SHA-256**, stable before/after, 298,713,088 bytes read.
- Second bounded source + NAS MP4 batch: **8/8 identical SHA-256**, stable before/after, 931,135,488 bytes read, skipping the previously measured eight objects.
- Third source/NAS MP4 batch: **6/6 matching SHA**, metadata stable, 1,413,480,448 bytes read, 16 prior checks skipped within a 1.5 GB cap.
- Fourth source/NAS MP4 batch: **6/6 matching SHA**, metadata stable, 1,736,441,856 bytes read, 22 previous matches skipped within a 2 GB cap.
- Current source/NAS SHA pairs matched: **28/74 indexed candidate MP4 files**, leaving **46** candidate MP4 files untested. The two prior-witness MP4s were destination-only rechecked separately.
- GitHub Actions **CI #317 passed** on parent PR #32, resolving the previous hard-coded mount path failures. This new multi-report slice requires its own exact-head CI. Every current comparison remains source-release HOLD.

**None of these checks authorizes deletion.** All SD cards remain mounted read-write, so even matching SHA and stable metadata remain point-in-time witness evidence. The x1 CIFS NAS identity is not independently proven as the authoritative Beelink Btrfs archive and the required producer-quiescence and storage-retention gates remain open.
## Durable local evidence

- /home/scott/nas-reconciliation-audit-20261016/auto-ingest-pr30-two-video-nas-digest-recheck-20261016.json
- /home/scott/nas-reconciliation-audit-20261016/auto-ingest-eight-smallest-mp4-source-nas-sha-20261016.json
- /home/scott/nas-reconciliation-audit-20261016/auto-ingest-second-batch-eight-mp4-source-nas-sha-20261016.json
- /home/scott/nas-reconciliation-audit-20261016/auto-ingest-third-batch-eight-mp4-source-nas-sha-20261016.json
- /home/scott/nas-reconciliation-audit-20261016/auto-ingest-fourth-batch-mp4-source-nas-sha-20261016.json
- /home/scott/nas-reconciliation-audit-20261016/auto-ingest-28-current-mp4-sha-witnesses-summary-20261016.json — cross-checked 28 distinct source UUID/path/destination records, pinned report and candidate-index SHA, **not** a release ledger.

## Read-only command (repeatable with unique output filename)

    python3 -m auto_ingest.custody.verify_candidates \
      --index /home/scott/nas-reconciliation-audit-20261016/auto-ingest-nas-indexed-candidates-4268-media-classified-20261016.jsonl \
      --campaign-root /home/scott/nas-reconciliation-audit-20261016/auto-ingest-campaigns-v1 \
      --max-files 8 --max-read-bytes 350000000 \
      --out /home/scott/nas-reconciliation-audit-20261016/unique-new-evidence.json

For the next batch, repeat `--prior-report <path>` **once for each of the four completed reports**. The verifier matches UUID, source-relative path, expected byte count and exact NAS candidate path, and checks both previous SHA values, stability and comparison source index. New evidence also includes a SHA-256 checksum of the input candidate index. Earlier reports without that checksum are backwards-compatible only through the strict candidate identity checks. Duplicate report paths, inconsistent digests and mismatched indexes are rejected. It skips prior verified matches to avoid redundant reads but **never** upgrades them to live custody or grants source release. The source UUID is validated against the kernel, the candidate destination is confined under the NAS namespace, files are opened without following a final symlink, and a bounded SHA read records pre/post filesystem stability. Only local evidence files are created, never original SD/NAS media.

## Remaining gates

Reconcile additional indexed archive namespaces for the 3,619 MP4 names with no candidate in the limited index set. Verify the remaining 46 current candidates with bounded read-only source/destination comparisons. Obtain independent Beelink physical-Btrfs identity, snapshot/retention, and a quiescent read-only source observation before entering the separate custody-release validation. Preserve DCIM directories and sidecars. **No bulk copying, graph write, mount change, merge, deployment, or source unlink in this workstream.**
