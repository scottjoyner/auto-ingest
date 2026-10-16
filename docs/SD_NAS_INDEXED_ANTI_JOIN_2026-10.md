# NAS indexed evidence anti-join — 2026-10-16

**Owner:** auto-ingest, following draft PRs #28 and #29. **Status:** read-only, report-only. Source card release, NAS transfers, service reloads and automatic model dispatch remain HOLD.

## Inputs and trust

This tool reads only four already-generated NAS candidate/witness reports plus the pinned 4,268-source frozen manifest and the three UUID-bound auto-ingest campaigns. It performs **zero live NAS or media payload reads**. Its digest routine hashes these *small input report files* to record provenance, not video bytes.

- Frozen source: `/media/scott/SSD_4TB/hermes-home/FLEET-STATE/sdcard-ingest/20261012/20261013-tranche17-independent-2573/complete-4268-source-2573-verified-custody.jsonl`
- Source manifest SHA-256: `0a2213f72c921e6443c6753660109b5d681e3a50a1751d7cf2638f0f80e8afa5`
- Local bundles: `/home/scott/nas-reconciliation-audit-20261016/auto-ingest-campaigns-v1/`
- Input reports under `/home/scott/nas-reconciliation-audit-20261016`: `sd-5140-nas-staging-source-1554-metadata-20261016.csv`, `sd-cards-68FC-69BE-alternate-root-candidates-20261016.csv`, `sd-5140-nas-staging-bounded-18-hash-witness-20261016.json`, and `sd-two-alternate-content-hash-witness-20261016.json`.

A candidate is joined by **card UUID, source-relative path and byte count**. It cannot claim that an unrelated same-size file has the right digest. Witness records must additionally carry equal source/destination SHA-256 digests, stable observation, known candidate path and matching file size. Even these are retained as **historical observations**, not live destination verification.

## Results from actual x1 2026-10-16 source reports

| Strict class | Files | Bytes | Required next proof |
| --- | ---: | ---: | --- |
| Prior point-in-time SHA witness | 20 | 310,241,245 | Fresh independently governed destination SHA, source stability + mount authority |
| Indexed path/size NAS candidate | 498 | 9,434,540,206 | Fresh NAS SHA plus authoritative source digest |
| No candidate in selected indexes | 3,750 | 522,634,433,419 | Search additional indexed archives; do NOT label globally missing |
| **Total** | **4,268** | **532,379,214,870** | **All remain source release HOLD** |


The 20 witnesses comprise 18 bounded staged objects on the 5140-6932 card and two Deathstar-alternate MP4s. **The apparent staged MP4 is actually a 4 KB AppleDouble file named `._1999_1128_202128_F.MP4`, not video.** The 18 staged witnesses are 12 CSV, 5 JPG and 1 AppleDouble metadata object. Only **2 genuine videos** have historical SHA matches, both on the Deathstar alternate path. This distinction is pinned in the reconciler's `video_mp4_filename_not_content_verified` flag.

### MP4 media-specific queue (AppleDouble excluded)

| Evidence class | Real MP4 files | Logical bytes |
| --- | ---: | ---: |
| Prior SHA witness requires recheck | 2 | 301,989,888 |
| Metadata candidate needs SHA | 74 | 9,070,641,152 |
| No candidate in indexed evidence | 3,619 | 522,581,727,880 |
| **Total MP4 video-named files** | **3,695** | **531,954,358,920** |

This includes the two pre-2000 clock-reset recordings that remain quarantined from canonical staging. Source has 3,696 MP4-named entries if the 4 KB AppleDouble object is counted as an MP4 by extension alone. `video_mp4_*` counters are based on filename identity, **not decoded video validity**.

## Durable local outputs

- `/home/scott/nas-reconciliation-audit-20261016/auto-ingest-nas-indexed-candidates-4268-media-classified-20261016.jsonl` — current per-object report with explicit media-name classifier and evidence class, candidate paths, point-in-time witness provenance, bytes and explicit current-NAS-verification=false.
- `/home/scott/nas-reconciliation-audit-20261016/auto-ingest-nas-indexed-candidates-4268-media-classified-summary-20261016.json` — counts, source UUID campaigns, pinned manifest hash and SHA-256 digests of input reports.

No evidence was imported into a custody hash or destination verification ledger. All three campaigns remain DISCOVERED, source mount rw, destination unresolved, source_release_allowed=false.

## Reproduce from a checked-out auto-ingest tree

```bash
R=/home/scott/nas-reconciliation-audit-20261016
F=/media/scott/SSD_4TB/hermes-home/FLEET-STATE/sdcard-ingest/20261012/20261013-tranche17-independent-2573/complete-4268-source-2573-verified-custody.jsonl

python3 -m auto_ingest.custody.nas_evidence_index \
  --frozen-ledger "$F" \
  --expected-sha256 0a2213f72c921e6443c6753660109b5d681e3a50a1751d7cf2638f0f80e8afa5 \
  --campaign-root "$R/auto-ingest-campaigns-v1" \
  --staging-csv "$R/sd-5140-nas-staging-source-1554-metadata-20261016.csv" \
  --alternate-csv "$R/sd-cards-68FC-69BE-alternate-root-candidates-20261016.csv" \
  --staging-witness "$R/sd-5140-nas-staging-bounded-18-hash-witness-20261016.json" \
  --alternate-witness "$R/sd-two-alternate-content-hash-witness-20261016.json"
```

The optional `--out` creates a *new* (never overwrites) local JSONL output with complete per-file classes. Leave it off to print a bounded JSON summary. This command has no `--execute` or deletion behavior.


## Next stage for the modern fleet

The first verification queue should cover **20 previously hashed source/destination pairs** and **498 metadata-only candidates**, scoped to explicit paths and bounded work windows. Check the independently governed NAS object identity and source receipt provenance before upgrading any record. Compare the source-UUID + digest against Beelink physical files, not simply x1 `/nas` strings. Keep Source+Destination readback and a snapshot/retention witness. For the 3,750 no-candidate records, broaden independent indexes before scheduling copies, and keep historical old NAS/Deathstar recoveries on their existing HOLD.

**Critical:** No result in this report is a copy instruction, overwrite recommendation, valid `destination.jsonl` row or SD-card `SAFE_TO_RELEASE` authorization. The original `DCIM/` tree is preserved.
