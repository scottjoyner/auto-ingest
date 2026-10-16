# SD-card offline-readiness and NAS-only metadata census — 2026-10-16

**Status: read-only, report-only. SD source release remains HOLD.** This extends draft PRs #28, #29, #30, #32, and #33 without modifying the existing recorded hashes or custody bundles.

## Live discovery

At approximately 13:30 EDT on October 16, all three removable FAT UUIDs remained detectable by the block layer, but **none had a mounted source filesystem**: 69BE-B963, 68FC-E499, and 5140-6932. The historical mount directories were absent. The NAS remained mounted via x1 CIFS to `//192.168.1.202/fileserver` with approximately 39 TB free. No automated remount, USB reset, media transfer, deletion, or source hash was attempted.

GitHub Actions **#319 passed** at the exact head of parent draft PR #33. The four earlier point-in-time source/NAS matching SHA-256 batch reports still establish 28 distinct previously compared MP4 files (not live custody). The two independent earlier destination-only MP4 rechecks are kept separate.

## New source-offline readiness command

`auto_ingest.custody.candidate_readiness` accepts a pinned candidate index and the UUID-bound campaign bundles. It uses the same strict multi-report skip validation from `verify_candidates`, but reads **zero source file bytes and zero NAS payload bytes**. It only asks the kernel whether the named source UUID/mount is present, then `lstat`s at most `--max-candidates` candidate paths inside the NAS namespace and compares their byte sizes against the immutable manifest.

It deliberately does not run network `realpath`/full path resolution for the *metadata-only* probe; on CIFS that can block for several seconds **per path component**. The later content SHA verifier keeps stricter path/symlink checks. A metadata-size match is not a SHA match, and neither result alone grants source release.

### Observed NAS metadata results

| Check | Result |
| --- | ---: |
| Candidate MP4 files previously matching source and NAS SHA | 28 (prior witness, not rechecked today in this phase) |
| Remaining indexed MP4 candidate files | **46** |
| Fresh NAS path+size matches, no source card reads | **46 / 46** |
| Total logical bytes in these candidates | **6,880,755,712** |
| Source candidates blocked by unmounted/UUID-unverified cards | **46 / 46** |
| Current media payload bytes read | **0** |
| Source release eligible | **0** |

Raw report: `/home/scott/nas-reconciliation-audit-20261016/auto-ingest-offline-remaining-46-nas-metadata-20261016.json`.

### Reproduce (read-only)

From the checked-out auto-ingest worktree:

    R=/home/scott/nas-reconciliation-audit-20261016
    python3 -m auto_ingest.custody.candidate_readiness \
      --index "$R/auto-ingest-nas-indexed-candidates-4268-media-classified-20261016.jsonl" \
      --campaign-root "$R/auto-ingest-campaigns-v1" \
      --prior-report "$R/auto-ingest-eight-smallest-mp4-source-nas-sha-20261016.json" \
      --prior-report "$R/auto-ingest-second-batch-eight-mp4-source-nas-sha-20261016.json" \
      --prior-report "$R/auto-ingest-third-batch-eight-mp4-source-nas-sha-20261016.json" \
      --prior-report "$R/auto-ingest-fourth-batch-mp4-source-nas-sha-20261016.json" \
      --max-candidates 46

Optional `--out "$R/unique-file.json"` creates a **new** OS-local report and refuses overwrites. The report includes full rows, explicit truncated/uninspected counts when a smaller scope is requested, current card mount state, and the input index SHA-256. No data is sent to a release/copy/hash ledger.

## Next gate

Before continuing SD-to-NAS content comparisons, validate exact /dev/disk/by-uuid identity and a deliberate source read-only observation under producer quiescence. Do not use an absent mount folder as proof of missing media. When the three card mounts are restored through a separately reviewed procedure, resume at the remaining 46 (6.88 GB) of indexed video candidates under explicit bounded I/O budgets. Broaden digest-indexed archive search for the other 3,619 video names without declaring global absence or starting a bulk copy. Independent Beelink physical Btrfs identity/snapshots/retention remain a separate release gate. Preserve DCIM/, non-video card data, and all original recordings.
