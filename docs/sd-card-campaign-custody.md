# SD-card campaign custody — deterministic state for card ingest

> **Status:** implemented. Read-only by default. Preservation-first.
> **Why:** Hermes should never have to reconstruct "what happened to this card?"
> by reading logs, `find` output and ad-hoc shell.

An SD card that is pulled from a dashcam has one job left: get every byte to the
canonical destination, prove it, and only then become releasable. Until now that
proof lived in log files and in an operator's head. This document and the
`auto_ingest.custody` package replace that with a derived state, an evidence
contract, a read-only planner and a fail-closed release gate.

---

## 0. What already decided "am I done?" (inventory, spec §1)

Before this slice the repository contained **22 independent definitions of
"complete"** — and not one of them asserted that bytes are present and verified
at a destination. Two of them could not even detect a truncated destination
file, because they define "done" as *the destination path exists* or *the
pipeline exited 0*.

| # | mechanism (file:line) | completion evidence | proves destination custody? | resumable from |
| --- | --- | --- | --- | --- |
| 1 | `auto_ingest/ingest_claim.py:190-191` | `stage == 'graph_written'` → `status='done'` | no — one graph write | `j.stages` map (reset by `:176-183`) |
| 2 | `auto_ingest/ingest_claim.py:101` | `status='queued'` (string absent from every `STATUS_*`) | no | — |
| 3 | `auto_ingest/ingest_claim.py:210-216` | `reap()` → `status='pending'` | no | TTL on `claimed_at` |
| 4 | `auto_ingest/ingest_write.py:74-92`, `:113-146` | artifact file **exists** ("for resume/claim logic", `:115`) | no — not even row-set completeness | file existence |
| 5 | `auto_ingest/ingest_import.py:145-151` | rows submitted | no | MERGE idempotency `:78-86` |
| 6 | `auto_ingest/outbox.py:118-120` + `:64-66` | `verify()` True → **DELETE** the row | no — `:Transcription{id}` exists (`:174-177`) | the SQLite row |
| 7 | `auto_ingest/fleet_batch.py:78`, `:130-131` | `"status": "READY"`; always `return 0` | no | nothing |
| 8 | `auto_ingest/ingest/transcripts.py:1402-1413` | graph-count + embedding-ratio heuristics | no | graph, per key |
| 9 | `auto_ingest/dashcam/yolo_embeddings.py:1426-1440` | DB row counts + duration from the DB | no | graph, per clip |
| 10 | `deploy/worker_ingest.sh:36-48` | `mv "$claim" "$DONE_DIR/x.done"` | no — the shell rc | the `mv` itself |
| 11 | `ingest_media.py:722-726` | `state["done"][sha]["ok"] = True` | no — and set even when sub-steps failed | `media_ingest.json`, written once at `:837` |
| 12 | `bulk_ingest_dashcam.py:125-129`, `:228-231` | container `returncode == 0`; `[SKIP]` also counts as success | no | none (delegates) |
| 13 | `bulk_ingest_dashcam.sh:57-63` | `local exit_code=$?` **after `tee`** → `[DONE]` | no — `tee`'s status | none |
| 14 | `run_ingest_all.sh:20-25`, `:139-141` | `flock` + `FORCE=1` override | no — delegates to #8 | graph (via child) |
| 15 | `run_ingest_daily.sh:24-40` | `SKIP loadavg` / `SKIP lock` / `rc=$?` — skip and success share exit 0 | no | nothing |
| 16 | `scripts/ingest_supervisor.py:140-143` | `status='ok'` iff `rc==0`; the recorded `nodes` count **never gates anything** | no | `ingest_ledger.json`, per day |
| 17 | `scripts/sync_knowledge_to_nas.sh:24-27` | `rsync -ni` diff empty → silent `exit 0`; `\|\| true` swallows probe failure | no — mtime+size diff | rsync's own comparison |
| 18 | `scripts/sync_cache_to_nas5.sh:21-23`, `:38` | `--ignore-existing` ⇒ destination path exists | **never** — a truncated destination file is silently accepted | destination path existence |
| 19 | `docs/current-ingest-state-2026-06-10.md:54`, `:158-186` | `processed=18 skipped=122 total=311`; FS-vs-DB key arithmetic | no | n/a (snapshot) |
| 20 | `docs/recovery-plan-2026-06-10.md:114-150` | node counts, "counts do not move in the expected direction" | no | MERGE-by-stable-ID |
| 21 | `docs/deathstar-cli-storage-migration.md:11-14` | "canonical storage layout" = NAS3 | asserts a destination, verifies none | n/a |
| 22 | `docs/OFFLINE_SWARM_INTEGRATION_PLAN.md:43-51`, `:159` | task lifecycle `complete/fail`; "artifact-exists based" skip | no | outbox replay |

Highest-value findings:

1. **`scripts/ingest_supervisor.py:136-143` computes a verification and then
   ignores it.** `nodes=neo4j_day_count(dk)` is written into the ledger and never
   gates anything, so `status='ok'` means only "rc was 0" — and the query itself
   (`:57`, `WHERE c.key STARTS WITH $p`) is a prefix count with no expected
   total, so 12-of-12 clips and 12-of-400 clips look identical.
2. **Two per-key completion oracles with different meanings.**
   `ingest_claim.py:190-191` makes `done` mean `graph_written` (six status
   strings, one of which — `"queued"` at `:101` — is not in any `STATUS_*`
   constant), while `ingest_write.py:115` makes it mean "the artifact file
   exists". Neither has anything to do with whether any byte exists anywhere.
3. **`bulk_ingest_dashcam.sh:57` captures `$?` after a pipeline ending in `tee`,
   and `set -e` (`:2`) makes the failure branch unreachable** — so `[DONE]`
   (`:63`) is printed off `tee`'s status, `failed_days` (`:70`, `:92`) is dead
   code, and `exit 0` (`:105`) is unreachable on failure.
4. **`sync_cache_to_nas5.sh:21` defines "done" as "a path exists at the
   destination"** (`--ignore-existing`), so a truncated or bit-rotted copy is
   permanently accepted and never repaired; its `findmnt` check (`:11-14`)
   validates a mountpoint string, not a storage identity.
5. **`ingest_media.py:722-726` writes `"ok": True` unconditionally** and `:837`
   persists the state file only after the whole loop finishes — so a file whose
   transcription failed is recorded as done, and a crash loses the entire run's
   progress despite the module advertising itself resumable (`:18`).

### What this package deliberately does not touch

| machinery | why |
| --- | --- |
| `ingest_claim.py`, `deploy/worker_ingest.sh`, `scripts/claim_job.py` | **different domain.** Coordination state about *workers* (`owner`/`claimed_at`/`stages`), not evidence about bytes. Custody reads around it. Replacing it is unnecessary: nothing calls `create_job`, and `worker_ingest.sh:24` already makes the filesystem move authoritative and treats the graph claim as best-effort. |
| `ingest_import.py`, `outbox.py`, `fleet_batch.py`, `ingest/__init__.py` | **different domain.** Graph-write and fleet-scheduling state (rows, nodes, Tasks). Custody has no correct answer for "did this 250k-row artifact import?" and must never infer a copy/verify verdict from `drained`/`kept`/`READY`. |
| `transcripts.py:should_reingest`, `yolo_embeddings.py` | **different domain.** Graph-coverage oracles over an already-durable corpus. Rewriting them would change transcription ingest behaviour; they remain the correct answer to "does this key need re-ingesting". |
| `bulk_ingest_dashcam.{py,sh}`, `ingest_supervisor.py`, `sync_*.sh` | **left on disk, outside the state machine.** They own *corpus-to-graph* ingest of already-durable trees. Recorded MANUAL_ONLY in §12 with disposition "do not reactivate". |
| `dashcam_copy.sh`, `bodycam_copy.sh`, `audio_copy.sh` | **gated, not wrapped.** These are the real byte-copy scripts. Custody describes and gates them; `custody verify --execute` refuses (§9). |
| `NAS3`/`NAS4`/`NAS5`/`SSD_4TB`/`/mnt/8TB_2025` literals | custody stores a *logical* destination and resolves the host path at runtime (§5), so it can coexist with five disagreeing canonical-root definitions elsewhere without joining the argument. |

---

## 1. The questions this answers

| question | answered by |
| --- | --- |
| What exactly has happened to this card? | derived `state` + `reasons` |
| What is verified? | `hash.verified`, `destination.verified_files/bytes` |
| What is missing? | `reconciliation.source_only`, `copy.in_flight_files` |
| Can this card safely resume? | `plan.safe_to_resume` + `plan.actions` |
| What would the resume plan do? | `plan.actions[]` (a description, never an execution) |
| Has destination custody been proven? | `state ∈ {VERIFIED, SAFE_TO_RELEASE}` |
| Can the source ever be released? | `source_release_allowed` / `release.allowed` |

---

## 2. The state model

States are **derived**, never declared. There is no setter; `derive_state()`
is a total pure function of `(campaign, evidence, policy)`. A `state` key in an
evidence file is ignored and reported in `ignored_declared_fields`.

```
DISCOVERED          card seen, nothing trustworthy observed yet
SOURCE_VERIFIED     source inventoried + inventory verified, hashing not begun
HASHING             hashing underway, coverage short of the inventory
HASH_COMPLETE       hashing complete, no copy plan recorded yet
COPY_PENDING        copy plan recorded, copy has not started
COPYING             copy started and a worker is still driving it
COPY_COMPLETE       copy result attested, destination verification not started
VERIFYING           destination verification underway, incomplete
VERIFIED            destination custody proven; release gate still closed
RECONCILE_REQUIRED  evidence is ambiguous or disagrees with the destination
BLOCKED             evidence contradicts itself; a human must intervene
SAFE_TO_RELEASE     custody proven AND every release condition satisfied
```

Precedence (first match wins) lives in `auto_ingest/custody/machine.py`:

| # | condition | state |
| --- | --- | --- |
| 0 | arithmetic contradiction, destination identity conflict | `BLOCKED` (`evidence_contradicts_itself`) |
| 0b | fatal campaign errors | `BLOCKED` (`campaign_has_fatal_errors`) |
| 0c | malformed counts (`coerced_fields`) | `BLOCKED` (`evidence_counts_are_malformed`) |
| 1 | nothing observed at all | `DISCOVERED` |
| 2 | inventory incomplete, work underway | `HASHING` |
| 3 | inventory incomplete, nothing underway | `DISCOVERED` |
| 4 | inventory complete + verified, hashing not started | `SOURCE_VERIFIED` |
| 5 | hash coverage short of the inventory | `HASHING` |
| 6 | hashing complete, no copy plan | `HASH_COMPLETE` |
| 7 | hashing complete, copy planned, not started | `COPY_PENDING` |
| 8 | copy started, result not attestable, worker idle | `RECONCILE_REQUIRED` |
| 9 | copy started, worker still alive | `COPYING` |
| 10 | copy attested, verification not started | `COPY_COMPLETE` |
| 11 | verification underway | `VERIFYING` |
| 12 | source-only / mismatched (destination-only only under strict scope) | `RECONCILE_REQUIRED` |
| 13 | custody proven, gate closed | `VERIFIED` |
| 14 | custody proven, gate open | `SAFE_TO_RELEASE` |

Two distinctions worth stating explicitly, because they are where hand-rolled
scripts usually go wrong:

* **`HASH_COMPLETE` vs `COPY_PENDING`** — `HASH_COMPLETE` means hashing is done
  and *no copy plan has been recorded*; `COPY_PENDING` means a plan exists and
  the copy has not started. Both are pre-copy states.
* **A stopped worker is a fact about a process, not a verdict about the data.**
  It only distinguishes "copy still in flight" (`COPYING`) from "copy result
  cannot be attested" (`RECONCILE_REQUIRED`). On its own it never blocks and
  never releases. `tests/test_custody_state_machine.py` pins all three cases.

---

## 3. Campaign identity

Identity is anchored on the **card**, not on where it was mounted:

```
card_key     = sha256(device | filesystem_uuid | serial)[:16]
campaign_id  = "sdcard-<card_key>-<sha256(created_at)[:8]>"
```

Comparison order is most-authoritative-first: `filesystem_uuid` → `serial` →
`device`. The first field **both sides carry** decides, so a matching UUID is
enough even when the card moved to another USB slot (the device node changes);
lower-priority fields are not consulted afterwards.

`CardIdentity.compare()` returns three outcomes, not two — and the distinction
matters operationally:

| outcome | meaning | `status` reports |
| --- | --- | --- |
| `matched` | the deciding field agrees | `observed_card_matches_campaign: true` |
| `different_fields` | it disagrees — provably another card | `kind: different_card` |
| `unprovable_fields` | one side carries a field the other lacks — sameness unprovable | `kind: identity_unprovable` |

The third case is not "the same card" and not "another card"; reuse is refused
either way, but reporting it as a hardware difference would send an operator
hunting for a card swap that did not happen. Both refusals surface in `status`
with a remedy (`custody new` into a separate bundle) rather than a bare `false`.

This is what stops a second unformatted card at `/media/scott/UNTITLED` from
inheriting CARD-01's evidence — every such card shares the label, and the label
is never treated as identity.

---

## 4. Evidence contract

`CampaignEvidence` is the bounded summary. Per-file detail lives in
`ledgers/*.jsonl` next to the bundle, read by `auto_ingest.custody.ledger`.

| block | fields |
| --- | --- |
| `inventory` | `discovered_files/bytes`, `complete`, `verified`, `roots` |
| `hash` | `verified_files/bytes`, `errors`, `complete`, `algorithm`, `exemptions`, `last_checkpoint` |
| `copy` | `planned{files,bytes}`, `completed{files,bytes}`, `started`, `result_complete`, `interrupted`, `ledger_complete` |
| `destination` | `verified_files/bytes`, `failures`, `verification_started/complete`, `unverified_present_files`, `observed_identity` |
| `reconciliation` | `source_only`, `destination_only`, `mismatched` |
| `worker` | `identity`, `status`, `started_at/stopped_at`, `last_checkpoint` |
| `errors` | `unresolved`, `fatal`, `summaries`, `witness` |

Boundedness: every free-form list (`roots`, `exemptions`, `error_summary`,
`summaries`) is capped at `policy.max_summary_entries` (default 20). The ledger
reader returns counts plus a capped error sample — never the record list.

### Phase A: what the kernel and the filesystem actually say

`campaign.source.read_only` is a **declared** field. Before Phase A, a bundle
that merely asserted `read_only: true` passed the release gate with **zero
blockers** — the single condition protecting the source was a claim in a file.
`/proc/mounts` states what the kernel actually applied, so three read-only
observers now supply the missing facts.

```bash
auto-ingest custody observe-mount --bundle PATH [--json] [--apply]
auto-ingest custody capacity      --bundle PATH [--json]
auto-ingest custody preflight     --bundle PATH [--json] [--job-dir PATH]
auto-ingest custody hash       --bundle PATH --root DIR [--limit N] [--apply]
auto-ingest custody verify     --bundle PATH [--destination DIR] [--recheck] [--apply]
```

| command | reads | answers |
| --- | --- | --- |
| `observe-mount` | `/proc/mounts` | is the source **actually** `ro`? does the declaration agree? |
| `capacity` | `statvfs` | is there room for the **outstanding** bytes, plus headroom? |
| `preflight` | all of the above + the release gate | could an executor run at all, and if not, which check failed? |

Three properties matter here:

* **The declaration is never overwritten.** `--apply` records
  `observed_read_only` *beside* `read_only`, because those are different claims —
  what a file asserts and what the kernel reports. Only a human resolves a
  disagreement, and the report says `CONFLICT` when they differ.
* **`observe-mount` exits 3 unless the source is observed read-only.** That is the
  gate a future executor needs; the current release gate deliberately still reads
  the declared field, and flipping that is a separate explicit decision.
* **Capacity uses outstanding bytes, not inventory.** A card 61% copied needs
  the remaining 39%. Headroom is 5% with a 1 GiB floor, because "exactly enough"
  is how a copy dies at 99%. An unstatable or unresolved destination reports
  `sufficient: false` — "I could not check" must never read as "fine".

`preflight` is deliberately **complete**: an executor may only proceed when it
says `safe_to_execute`, so anything it leaves unchecked is a hole in the gate.
It aggregates mount observation, destination resolution, capacity, queued
`.job` files (the `ingest-worker` race, `docker-compose.yml:49`), and the
release blockers into one answer with a remedy per failed check.

All three are clock-free, deterministic, and covered by the audit-hook
read-only proof.

### Phase B: `custody hash` - the first producer

Until this, every number in a campaign was asserted. `hash.jsonl` had a reader, a
documented contract, and **no writer anywhere in the repository** - so
`67,644` existed only in tests and in a hand-written fixture. Now it can be
measured.

```bash
auto-ingest custody hash --bundle PATH --root DIR [--limit N] [--apply]
```

Reads each object (`rb`, streamed in 1 MiB chunks), appends a compact
newline-terminated record, and never writes the source. The custody key is the
POSIX-relative path, which is what reconciliation joins on.

Four properties, each of which the fail-closed reader demands:

* **Every record is newline-terminated and fsynced.** An unterminated final line
  reads as a killed producer and voids the *entire* reconciliation.
* **Resuming never glues a record onto a damaged line.** This was a real defect
  found by the tests: after a crash, appending straight on produced
  `{"key":"broken"{"algorithm":...}` - so one crash cost *two* objects, the
  partial one and the one merged into it. The producer now terminates a damaged
  trailing line first. It is not *repaired* (guessing where truncated JSON ended
  would be inventing evidence); it is closed off, so every complete record stays
  readable and the damage stays visible.
* **Two runs over an unchanged source are byte-identical**, because records are
  written in sorted-key order.
* **Only a complete pass claims completion.** `--limit` produces a real partial
  count and `complete: false`, so the state machine reports `HASHING` rather than
  believing the card is hashed.

Read-only on the source is enforced two ways: a scoped audit-hook test that fails
on any write outside the campaign bundle, and an AST check that the module
references no `os.remove`/`shutil`/`subprocess`.

### Phase B2: `custody verify` - proving custody

`custody hash` says what the source contains. `custody verify` produces the other
half: what is actually present and correct **at the destination**. It is the only
thing that can prove custody, and therefore the only thing that can ever open the
release gate.

```bash
auto-ingest custody verify --bundle PATH [--destination DIR] [--recheck] [--apply]
```

For each key in the hash ledger it reads the expected digest from the ledger,
computes the actual digest from the destination file, and compares:

| condition | status | counts as custody |
| --- | --- | --- |
| digests match | `verified_at_destination` | **yes** |
| destination absent | `missing` | no |
| digests differ | `mismatch` | no |
| destination unreadable | `failed` | no |

It reads the destination and writes the campaign's own ledger. It **copies and
deletes nothing** — a verification pass that moved bytes would produce a result
that looks authoritative while proving nothing. `--execute` stays refused because
*copying* is the separately authorized part.

This is the answer to `--ignore-existing`. A destination file that exists but is
truncated, corrupt, or the wrong bytes is a `mismatch`, and it is counted against
release forever:

```
verified=2  mismatched=1  missing=1   # 4 objects, 2 proven, 2 not
```

#### Two cumulative-vs-delta defects the tests caught

Both producers originally reported **this pass's work**, so a resume wrote `0`
over a real count and regressed the campaign:

```
first pass : verified_files = 4
resume     : verified_files = 0     # -> regressed out of custody
```

Evidence fields describe the *ledger*, not the *run*. Both now report
`verified + skipped_existing`, so a re-run is idempotent instead of destructive.
This is the same class of bug as `import` replacing rather than merging — worth
noting because the shape recurs whenever "what happened" and "what is true" are
conflated.

`custody hash` also now records the **inventory it walked**. Omitting it left the
bundle self-contradictory — `hash.verified_files = 4` beside
`inventory.discovered_files = 0` — which the state machine correctly refused as
`BLOCKED`, so a campaign could never leave `HASHING` while its own producer knew
the count.

#### Where the pipeline stops today

`hash` → copy → `verify` produces this, which is the honest end state:

```
after hash          : state=HASH_COMPLETE  inventory=4  hash.verified=4
partial verify      : state=HASH_COMPLETE  release=false
all verified        : state=HASH_COMPLETE  release=false
                    blockers: copy_incomplete, copy_ledger_incomplete
```

The destination is proven byte-for-byte and release is **still refused** — because
a diff cannot attest that a *copy* happened. `copy.result_complete` and
`copy.ledger_complete` come from the executor, which is Phase D and separately
authorized. The pipeline now runs from measurement to a correct refusal; the only
remaining step is the one that writes bytes.

### Phase C: the campaign lock, and a race this package cannot fix alone

The investigation surfaced a hazard that is not about custody at all, but would
corrupt a campaign if left alone:

* `sync-service` (`docker-compose.yml:85`, every 10 min) and
  `deploy/cron/ingest.crontab:5` (every 5 min) run
  `deploy/sync_from_legacy_drop.sh`, which rsyncs `--archive --ignore-existing`
  into `$AUDIO_ROOT` / `$DASHCAM_ROOT` / `$BODYCAM_ROOT` — **the same roots a
  campaign writes to**. No lock, no campaign awareness. It will silently skip
  whatever a campaign produced, and can populate those roots independently.
* `ingest-worker` (`docker-compose.yml:49`, every 30 s) claims and executes
  arbitrary `.job` files from `$DROP_ROOT`, also uncoordinated.

`auto_ingest.custody.lock` provides the primitive:

| function | purpose |
| --- | --- |
| `campaign_lock(dest)` | advisory exclusive lock, held for a campaign's duration |
| `is_locked(dest)` | probe a destination is free — callable by a legacy writer that knows nothing about custody |
| `competing_activity()` | report writers that may touch the destination uncoordinated |

`auto_ingest.custody.lock` first made that dependency impossible to miss:
`preflight` gained a `no_uncoordinated_writers` check, and because
`legacy_drop_sync` is a *standing* property of this host rather than a transient
state, it is always reported.

### Phase C.5: the legacy sync stands down

`sync_from_legacy_drop.sh` is now patched. Its first act stats the campaign
marker, before any rsync:

```bash
CUSTODY_LOCK_ROOT="${CUSTODY_LOCK_ROOT:-/nas/custody-locks}"
CUSTODY_ACTIVE_MARKER="$CUSTODY_LOCK_ROOT/campaign-active"
CUSTODY_MARKER_TTL_SEC="${CUSTODY_MARKER_TTL_SEC:-21600}"
if [[ -e "$CUSTODY_ACTIVE_MARKER" ]]; then
  marker_age=$(( $(date +%s) - $(stat -c %Y "$CUSTODY_ACTIVE_MARKER" 2>/dev/null || echo 0) ))
  if [[ "$marker_age" -gt "$CUSTODY_MARKER_TTL_SEC" ]]; then
    echo "... marker is ${marker_age}s old, past the TTL — no live campaign, syncing"
  else
    echo "... custody campaign active — standing down this pass"
    exit 0
  fi
fi
```

Deliberately cruder than the campaign's own `flock`: this script only knows host
paths, while the campaign lock is keyed on a *logical destination name*, and the
two cannot be reconciled without inventing a shared naming convention. A marker
file is the whole handshake. It exits `0` so cron logs a reason rather than an
error, and the next run five minutes later picks up whatever was left behind.

**The root is `/nas`, not `/tmp`.** The first version of this probe defaulted to
`/tmp/auto_ingest_custody` and was inert in production: `sync-service` and
`ingest-cron` are separate containers with separate `/tmp`s, so each scheduler
read a *different* file and neither could ever see a marker dropped by
`custody execute`. Worse, the coordination still *looked* verified.
`/nas` is the mount all four ingest services already share
(`docker-compose.yml`; `DROP_ROOT=/nas/drop`), so no compose edit and no restart
was needed. `CUSTODY_LOCK_ROOT` still overrides it, and the library's
`DEFAULT_LOCK_ROOT` and the script's default are asserted to be the same string —
one root in two implementations is the entire invariant.

**The marker is a claim with a TTL**, mirroring `ingest_claim`'s `claimed_at`: the
file's mtime is the claim timestamp, and a marker older than
`CUSTODY_MARKER_TTL_SEC` (6h default) is stale — that pass *proceeds* and says
why. A SIGKILL skips the `finally` in `cli.py` that clears the marker, so it can
outlive its campaign; without the TTL the survivor halts ingest silently and
permanently, which is worse than the race the marker exists to prevent.

`ingest_claim` has no heartbeat primitive, so a fixed TTL would instead reap a
*healthy* campaign: a real one is ~90GB over USB and runs for hours.
`execute_copy` therefore refreshes the marker's mtime every
`MARKER_REFRESH_SEC` (30s) while it copies. The refresh is pure bookkeeping —
`touch_active()` never raises and never creates a marker, so it cannot fail a good
copy or start a campaign nobody authorized.

Three properties matter more than the mechanism:

* **Visibility is tested, not grepped.** `writer_consults_lock()` reading the
  script proves the probe *exists*; it cannot prove the writer can *see* the
  marker, which is precisely how a `/tmp` root passed review. `test_custody_lock.py`
  now runs two subprocesses with different `TMPDIR`s plus the real script against
  one shared root, and asserts the default is under `/nas` and not `/tmp`.
* **The claim is re-derived, never remembered.** `writer_consults_lock()` reads
  the script and looks for both the probe and the staleness branch, so reverting
  either half makes `no_uncoordinated_writers` fail again. Nothing in the report
  can go stale.
* **The repo is bind-mounted.** `docker-compose.yml` maps `./:/app`, and both
  entry points invoke `/app/deploy/sync_from_legacy_drop.sh`, so the deployed
  script *is* this file. The patch applies on the next tick without a rebuild or
  restart.

`no_uncoordinated_writers` now reports `none`, and `preflight` can return
`safe_to_execute: true` for a fully reconciled campaign.

Locks are advisory — they coordinate software that agrees to take them, and the
marker is deliberately not a `flock` so no writer needs a custody import. A
crashed executor releases its `flock` on the way out; `is_locked()` remains
available if a per-destination handshake is ever wanted in place of the marker.

### Phase D: `custody execute` - the authorized copy

The only command in the package that writes bytes to the destination. Narrow on
purpose.

```bash
# show the plan; copy nothing
auto-ingest custody execute --bundle PATH --source-root DIR --destination DIR \
  --i-have-stopped-the-sync-service --json

# the authorized run
auto-ingest custody execute ... --execute --i-have-stopped-the-sync-service --apply
```

**Two gates, one of them conditional.** `--execute` is required or nothing is
copied. `--i-have-stopped-the-sync-service` is required *only when coordination
cannot be verified* — the response reports which, as
`acknowledgment_required`. On this host the C.5 probe is verified and the sync
stands down on its own, so demanding the claim would be friction; where the
deployment is a copy rather than the repo, the operator's word is the only
evidence available, so it is a hard gate there. Either way the acknowledgment is
*recorded* in `copy.acknowledged_uncoordinated_writers` rather than assumed.
Exit codes: `0` means no blockers (with `--execute` absent, the plan is
executable and nothing was copied); `3` means refused, and `blockers` says why.

**The plan is derived from the ledgers, in three buckets:**

| bucket | action |
| --- | --- |
| already verified at the destination | nothing |
| present but unattested | `custody verify`, **not** a copy |
| genuinely absent | copy |

So the 7,000-objects case from §6 cannot turn into 7,000 wasted copies.

**Atomicity is the load-bearing property.** Each object is written to
`<dest>/.custody-tmp/`, fsynced, then `os.replace`d into place. A killed
executor therefore leaves a temp file and **never a half-written object under a
real name** — which is what makes a later `verify` meaningful, and the direct
opposite of `--ignore-existing`. Verified by `SIGKILL`ing a real 384 MiB copy
mid-flight: 20 of 24 objects complete, 1 leftover temp, **0 short real objects**.

Other guarantees, each tested:

* **Never overwrites.** An existing destination object is left byte-identical and
  recorded as `skipped`; verification decides. This is what protects a good copy
  from being clobbered by a second run.
* **Never deletes.** No `rmtree`, no `rmdir`, no `os.remove`. An unrelated file
  the operator put at the destination survives.
* **Never touches the source.** Opened `rb` and only read; mtime, size and mode
  are asserted unchanged after a full run.
* **Never escapes the destination.** A ledger key containing `../` or an absolute
  path is refused rather than trusted — keys come from a file on disk.
* **Never copies what is not in the plan.** An unhashed file on the card is not in
  the ledger, so it is not copied.
* **The recorded digest is of the bytes actually written**, computed while
  streaming, so "copied" means "these bytes were produced and hashed".

`copy.result_complete` / `copy.ledger_complete` come from here and nowhere else —
`custody verify` can prove bytes are correct, but only an executor can attest that
a *copy happened*. That is precisely why release stayed shut through Phases B
and B2.

#### The pipeline now closes

```
hash -> observe-mount -> execute -> verify
  state: HASH_COMPLETE -> COPY_COMPLETE -> VERIFIED -> SAFE_TO_RELEASE
  hash verified 5, copy complete, destination verified 5 / 10240 bytes
  source_release_allowed: true    blockers: none
```

Two tests pin this end state: the capstone above, and
`test_a_missing_digest_keeps_release_closed`, which proves the *same* verified
bytes on a *different* storage identity yield `BLOCKED` with
`destination_identity_conflict` — proven custody of the wrong volume is not
custody.

### Import merges; it never replaces

`custody import` overlays the incoming document **field by field**. What the
document states wins — including a declared `0` — and what it does not mention is
preserved. A narrow document (say, a reconciliation diff that knows nothing
about hashes) therefore cannot erase the hash evidence already on record. The
response lists what was preserved in `preserved_fields`.

This is not a nicety. A whole-document or whole-block replacement would let a
partial report silently zero out `verified_files`, `failures` or
`observed_identity` — precisely the class of mistake §0 exists to catalog.

### Reconciliation is computed, not shelled

`reconciliation.source_only / destination_only / mismatched` are the set
difference between what the *source* claims it hashed and what the *destination*
claims it verified. `custody reconcile` computes it from the two JSONL ledgers
instead of leaving it to `comm`/`jq`:

```
custody reconcile --bundle PATH [--json] [--max-samples N] [--require-release]
```

Per source key: present in both with equal digests → `verified`; present in both
with different digests → `mismatched`; source only → `source_only`; a record
with no digest on either side → `unverifiable`. Destination keys with no source
counterpart → `destination_only`.

Three fail-closed properties:

* **An absent ledger proposes nothing.** Otherwise the diff would be all zeros
  and importing it would overwrite real counts with zeros. `proposal()` returns
  `None`, `to_evidence()` raises, and the CLI prints
  *"a ledger is missing; no proposal is offered because zeroed counts must not
  overwrite real evidence"*. An *empty but present* ledger is still usable — that
  is a real, if alarming, fact rather than an absence.
* **A record without a digest is never custody.** It is counted as
  `unverifiable` and folded into `source_only` in the proposal, because it lacks
  *proven* custody.
* **Reconciliation cannot prove destination identity.** The ledger says objects
  matched; it cannot say which filesystem they were matched on. That stays a
  separately recorded fact, so a perfect diff with no recorded identity yields
  `VERIFIED`, not `SAFE_TO_RELEASE`.

The command is read-only and prints `state_if_imported` — what the campaign would
derive once the proposal is applied. Preview and import share one merge
implementation, so the preview cannot drift from what `--apply` actually does.

### Ledger record contract (what a producer must emit)

The reader is the specification. An executor written later must emit these
records, one JSON object per line, newline-terminated:

| field | required | meaning |
| --- | --- | --- |
| `key` | yes | stable object identity; **the join key between the two ledgers** |
| `digest` | yes | lowercase hex sha256 of the object's bytes |
| `status` | yes | `verified` / `hashed` on the source side; `verified` / `verified_at_destination` / `copied` on the destination side |
| `path` | no | recorded, not used for the diff |
| `size` | no | recorded, not used for the diff |
| `phase` | no | free-form |
| `detail` | no | free-form, surfaced in error samples |

Rules the reader enforces, and why:

* **`key` is the join key.** Two ledgers are only reconcilable if both sides use
  the same key for the same object. Anything else silently reconciles to
  "everything is source_only".
* **Records in a non-verified status do not participate** (`failed`,
  `mismatch`, `missing`, `skipped`, `pending` are read but excluded from the
  counts), so a half-finished writer cannot inflate custody.
* **Every line must be newline-terminated.** A final line without `\n` means an
  interrupted append and voids the whole diff. This is the single most likely
  corruption and the most dangerous one, because the rows that *did* land look
  perfectly healthy.
* **Unparseable lines and keyless records are counted, not skipped silently.**
* **The hash ledger is cross-checked against the recorded inventory.** If
  `evidence.inventory.discovered_files` is known and the ledger covers a
  different number of objects, the diff is void: a partially-written ledger
  cannot masquerade as a complete one.
* **Undecodable bytes must not raise** — the reader opens with
  `errors="replace"` and counts the damage.

Corruption voids the *entire* diff (`incoherent` lists why) rather than shrinking
it. A proposal that covers only the rows that survived is worse than no proposal,
because the uncovered rows then read as proven custody.

### Scale

Measured on a card-sized bundle — 67,644 objects, 21.5 MB of ledgers:

| command | wall time | report size |
| --- | --- | --- |
| `status` | 0.40 s | 3.9 KB |
| ledger summarise | 0.32 s | 0.8 KB |
| `reconcile` (preview) | 0.73 s | 3.9 KB |

Counts are exact; samples are capped, so the report does not grow with the card.
`tests/test_custody_scale.py` builds the full-size fixture and asserts the read
paths stay inside a generous budget and the summary stays under 32 KB.

---

## 5. Canonical destination abstraction

Three separate concepts, never collapsed:

| concept | field | meaning |
| --- | --- | --- |
| logical custody destination | `destination.logical` | host-independent name + relative path |
| resolved host path | `destination.host_path` | where it lives on *this* host right now |
| storage identity evidence | `destination.identity` | fs UUID / device / label / type |

Resolution order: `CUSTODY_DESTINATION_ROOT` → `custody.destination_root` in
`config.yaml`. **No historical destination is embedded in core logic** — with
neither configured, `host_path` is `None`, which blocks release instead of
silently picking `NAS3`, `NAS5`, `SSD_4TB` or any other mount. The current
runtime architecture is free to export `/nas`; the repository contract simply
does not assume it.

Tests assert the absence of those literals from every module in the package.

---

## 6. Resume planner (read-only)

`plan_resume()` returns a description. It has no filesystem, network, database
or subprocess access (asserted by an AST test), never copies, never deletes, and
always reports:

```json
{
  "safe_to_resume": true,
  "current_state": "RECONCILE_REQUIRED",
  "next_phase": "destination_reconciliation",
  "next_safe_action": "reconcile_destination",
  "source_mutation_allowed": false,
  "source_deletion_allowed": false,
  "requires_operator_authorization": true,
  "actions": [
    {"operation": "verify_existing_destination", "mutates_destination": false, "...": "..."},
    {"operation": "copy_objects", "excludes_verified": true, "gated_by": ["act-..."]},
    {"operation": "verify_destination", "excludes_verified": true}
  ]
}
```

**Verification before re-copy.** After an interrupted copy the destination may
already hold most of the payload. The plan therefore always emits
`verify_existing_destination` first, and every copy action carries
`excludes_verified=true`.

The outstanding set is then **partitioned**, not copied wholesale. Objects split
into three groups, and the two actions must add up to exactly the outstanding
count:

| group | action |
| --- | --- |
| verified at the destination | neither — done |
| present at the destination, unattested | `verify_existing_destination` |
| no destination presence at all | `copy_objects` |

Getting this wrong is easy and invisible: counting present-but-unattested objects
as copy work makes the plan tell the operator to *verify 7,000 objects and then
copy those same 7,000 again*. The copy estimate is therefore
`outstanding − present_unverified`, clamped at zero so a miscounted ledger cannot
produce a negative estimate.

**Idempotent task identity.** `action_id` is a digest of the action's own
content and `plan_fingerprint` digests the whole action set, so `inspect → plan
→ inspect → plan` against unchanged evidence yields byte-identical output and no
duplicate task ids.

---

## 7. Fail-closed release gate

`SAFE_TO_RELEASE` requires *every* condition below. The gate never
short-circuits, so the blocker list is a complete explanation.

```
source read-only                     destination resolved
destination mounted                  destination identity proven
source inventory complete            hash evidence complete
hash gaps only policy-declared       copy result attested complete
copy ledger complete                 destination verification complete
mismatches = 0                       missing destination = 0
unresolved errors = 0                copy plan covers the required scope
operator witness (if policy requires)
```

`required_scope` defaults to `all_inventory`: every inventoried object must be
accounted for, so a copy plan that quietly dropped objects is a blocker, not a
pass. `hashed_set` is available as a weaker, explicit opt-in.

"Required scope" has exactly one definition, `policy.required_objects()`,
called by **both** the state machine and the release gate. They used to disagree:
`hashed_set` weakened the gate's plan-scope check while the machine still
demanded full inventory coverage, so the knob silently did half of what it said.
A narrower scope is also never silent — releasing on a subset emits
`N inventoried objects are OUTSIDE the required scope (hashed_set): they are not
covered by this release`, because a later reader must not assume the whole card
was verified.

Hash exemptions only count when `custody.policy.declared_hash_exemptions`
declares the pattern; an exemption appearing only in evidence is a blocker
(`hash_exemptions_not_declared`).

### Malformed evidence

Counts that are not numbers (`"6,764"`, `"abc"`) are recorded in
`coerced_fields` and treated as a distinct failure from contradictory evidence,
because they send an operator to different places:

| input | result |
| --- | --- |
| `"6,764"`, `"abc"`, `{}` | coerced to 0, `BLOCKED` / `evidence_counts_are_malformed` |
| `"67644"` | accepted (numeric strings are fine) |
| `-5` | `BLOCKED` / `evidence_contradicts_itself` (impossible value, parses fine) |
| `3.9` | truncates to 3, not a blocker — and can never enable a release |

Malformed evidence fails *closed* in every case: a zeroed count contradicts any
non-zero hash evidence, so the machine blocks rather than progressing.

Two properties keep the diagnostic from evaporating:

* **It survives the round trip.** `import --apply` writes normalised evidence and
  reads it back; without rehydration the record of the malformation would be
  dropped by exactly the operation an operator runs to investigate it.
* **It is visible without `--json`.** `status` defaults to human output, so a
  diagnostic that only appears in JSON is invisible to the operator who needs it:

```
state              BLOCKED
reasons            evidence_counts_are_malformed
  MALFORMED COUNT  inventory.discovered_files='6,764'  (not numbers; fixed to 0 - fix the source evidence)
```

The same rule applies to `ignored_declared_fields`, so a dropped declared state
is reported in both modes.

---

## 8. The interrupted-copy case

```
hashing completed
copy may have partially occurred
worker stopped
destination ledger incomplete
        │
        ▼
RECONCILE_REQUIRED      (not COPY_PENDING, never SAFE_TO_RELEASE)
```

Reasons reported: `copy_started`, `copy_result_not_attestable`,
`copy_interrupted`, `worker_stopped`, `destination_ledger_incomplete`.

---

## 9. Commands

All read-only by default. `--require-release` exits 3 when the gate is closed,
which makes the commands usable as a CI/Hermes gate.

```bash
auto-ingest custody observe-mount --bundle /path/to/campaign      # what the kernel says
auto-ingest custody capacity      --bundle /path/to/campaign      # will it fit?
auto-ingest custody preflight     --bundle /path/to/campaign      # could an executor run?
auto-ingest custody new     --bundle /path/to/campaign --card-id CARD-02 \
  --uuid "$(blkid -s UUID -o value /dev/sdb1)" --device /dev/sdb1 \
  --label UNTITLED --mount /media/scott/UNTITLED --read-only --apply
auto-ingest custody status  --bundle /path/to/campaign          # human
auto-ingest custody status  --bundle /path/to/campaign --json   # machine
auto-ingest custody plan    --bundle /path/to/campaign --json
auto-ingest custody reconcile --bundle /path/to/campaign         # diff the two ledgers
                                                   [--require-proposal]
auto-ingest custody verify  --bundle /path/to/campaign          # describes; --execute is refused
auto-ingest custody import  --bundle /path/to/campaign --evidence ev.json   # validate only
auto-ingest custody import  --bundle /path/to/campaign --evidence ev.json --apply
```

Observed hardware can be cross-checked inline:

```bash
auto-ingest custody status --bundle ... --json \
  --observed-uuid "$(blkid -s UUID -o value /dev/sdb1)" --observed-device /dev/sdb1
```

`observed_card_matches_campaign: false` means the card now at this campaign's
mount point is not provably the one the campaign describes. `observed_card_conflict`
distinguishes `different_card` (a recorded field disagrees) from
`identity_unprovable` (one side carries an identity field the other lacks) and
carries the remedy.

### plan vs execute

`new`, `status`, `plan`, `verify` and `import` (without `--apply`) never execute
anything against media. The **only** two writes in the package are
`import --apply` and `new --apply`, both explicit and both inside the campaign
bundle directory. `verify` deliberately has no execution path:
`custody verify --execute` returns exit 3 and a refusal message, because
destination verification is an operator-authorized step performed by an executor
outside this package. There is no implicit execution as a side effect of reading
state.

`custody new` is the only command that reads a clock (a campaign needs a
creation instant), which is why `--created-at` exists and why its output is a
write. `status`, `plan` and `verify` stay clock-free and random-free — pinned by
`tests/test_custody_legacy_watchers.py::test_writing_commands_are_the_only_ones_taking_a_clock`.

`custody status` and `custody reconcile` are the commands an operator or Hermes
polls. Both are pure functions of files on disk, so they work from any working
directory and never need the card mounted:

```bash
auto-ingest custody status --bundle ...
auto-ingest custody --help            # REMAINDER forwarding swallows --help; handled for you
```

The only writes in the package (`import --apply`, `new --apply`) go through one
atomic writer whose temp file name is unique per process and per call, so an
operator and Hermes applying evidence at the same moment cannot silently lose
one update. A failed write leaves no temp file behind.

`custody new` fails closed three ways: a label alone is not identity (`UNTITLED`
proves nothing, so the campaign is not created); it never overwrites an existing
`campaign.json`; and an unresolved destination is recorded as unresolved rather
than guessed — which blocks release, not onboarding.

---

## 10. Operator loop

```
observe physical source        # lsblk / blkid, read-only; mount read-only
  → custody new --apply        # mint campaign identity from observed hardware
  → update/import evidence     # auto-ingest custody import ... --apply
  → custody status             # what state is this card in?
  → custody plan               # what would the next safe operation be?
  → operator-authorized execution
  → custody reconcile          # diff the ledgers: what is actually in custody?
  → custody import --apply     # record the reconciliation (or just observe it)
  → custody verify             # describe/confirm the verification set
  → custody status             # has custody been proven?
```

Each step is a separate, individually authorized command. `new` records who the
card is, `import` records what has been observed about it, `reconcile` computes
the set difference from the ledgers, `status`/`plan` read, and only an explicitly
authorized executor moves bytes.

Hermes can then answer every question in the table in §1 without reconstructing
anything from logs.

---

## 11. CARD-01 (latest observed)

Fixture: `tests/fixtures/custody/card-01/` (static JSON, not wired to the card).

| fact | value |
| --- | --- |
| source | `/media/scott/UNTITLED`, read-only |
| hashes verified | 67,644 |
| copy | incomplete |
| destination verification | 0 files / 0 bytes |
| source deletion | none |
| worker | stopped |

Derived: **`RECONCILE_REQUIRED`**, `source_release_allowed: false`,
`next_safe_action: reconcile_destination`. Not `COPY_PENDING` (a copy was
attempted), certainly not `SAFE_TO_RELEASE`.

---

## 12. Legacy watcher disposition

Audit of repository paths referring to the old autonomous watchers:

| path / literal | class | disposition |
| --- | --- | --- |
| `sdcard-nas-ingest-watcher` | **DEAD / UNTRACKED** | zero occurrences in the repo or in any of its 20 refs; it lives only at host/system level |
| `untitled-sd-ingest` | **DEAD / UNTRACKED** | same — no code, no unit, no cron, no udev rule in version control |
| `/media/scott/UNTITLED` | HISTORICAL PATH | appears in the fixture as recorded evidence only; custody never opens it |
| `dashcam_copy.sh`, `bodycam_copy.sh`, `audio_copy.sh` | **MANUAL_ONLY** | human-run copy scripts; nothing schedules them. **Not reactivated, not referenced by custody** |
| `runall.sh` copy chain | MANUAL_ONLY | chains the three copy scripts above; unchanged |
| `scripts/ingest_supervisor.py` | MANUAL_ONLY | no `.timer`/`.service` ships for it; unshipped scheduler |
| `scripts/sync_cache_to_nas5.sh`, `sync_knowledge_to_nas.sh` | MANUAL_ONLY | rsync mirrors; not invoked by custody |
| `NAS3` / `NAS5` / `SSD_4TB` literals | HISTORICAL PATH | config-only; **not embedded in custody core**, asserted by tests |
| `deploy/watchdog.py`, `KnowledgeVaultWatcher` (`HLD.md`, `LLD.md`) | DEAD_REFERENCE | design-doc references to a file that does not exist |
| `systemd/*.service` | ACTIVE (unrelated) | Neo4j health / re-embed / dashcam-vision over NAS archives; never removable media |

**The custody state machine does not depend on any of them.** It has no udev
rule, no `.path`/`.timer` unit, no cron entry, no inotify/watchdog import and no
autonomous trigger: it cannot start itself, and it derives its state purely from
a campaign bundle on disk. `tests/test_custody_legacy_watchers.py` asserts this
in code — including that this slice adds no `.rules`, `.path`, `.timer` or
`.mount` file.

Disposition: **do not reactivate.** If a card needs ingesting, an operator
observes it, imports evidence, reads `custody status` / `custody plan`, and
authorizes execution explicitly.

---

## 13. Safety properties (and where they are proven)

| property | where proven |
| --- | --- |
| state is derived, not declared | `test_custody_state_machine.py::test_declared_state_is_ignored`, `test_derive_state_signature_rejects_a_state_argument` |
| release is fail-closed | `test_custody_release_gate.py` (18 cases) |
| plan never mutates the source | `test_custody_planner.py::test_no_action_ever_mutates_the_source` |
| status/plan cause zero filesystem mutation | `test_custody_readonly.py` (CPython audit hook in a subprocess) |
| no duplicate task generation | `test_custody_idempotency.py` |
| a different card cannot inherit state | `test_custody_identity.py` |
| a card's identity is unprovable without a UUID/device/serial | `test_custody_new_campaign.py::test_label_only_is_refused` |
| evidence is never overwritten | `test_custody_new_campaign.py::test_existing_campaign_is_never_overwritten` |
| JSON serialisation is deterministic | `test_custody_idempotency.py::test_status_json_keys_are_sorted` |
| import merges, never replaces | `test_custody_reconcile.py::test_import_merges_a_narrow_document_instead_of_replacing`, `test_import_preserves_recorded_subfields_the_document_does_not_mention`, `test_an_explicit_zero_in_the_document_wins` |
| an absent ledger proposes nothing | `test_custody_reconcile.py::test_missing_ledgers_are_unusable_and_propose_nothing` |
| a truncated ledger proposes nothing (fails open otherwise) | `test_custody_reconcile.py::test_interrupted_producer_does_not_claim_custody_for_rows_it_reached` |
| a partial ledger cannot masquerade as complete | `test_custody_reconcile.py::test_a_partially_written_hash_ledger_cannot_masquerade_as_complete` |
| the summary stays bounded at card scale | `test_custody_scale.py` |
| an observation can contradict the declaration | `test_custody_observers.py::test_a_declaration_alone_passes_the_gate_but_observation_refuses` |
| capacity uses outstanding bytes, not inventory | `test_custody_observers.py::test_capacity_uses_outstanding_not_total` |
| the observers cannot write | `test_custody_observers.py::test_the_observers_emit_no_mutating_syscall` |
| preflight names every failed check | `test_custody_observers.py::test_preflight_refuses_card01_and_names_every_reason` |
| resuming never merges onto a partial line | `test_custody_hash.py::test_resuming_never_glues_a_record_onto_a_partial_line` |
| a crashed producer voids its own diff | `test_custody_hash.py::test_a_simulated_crash_voids_the_whole_diff` |
| two runs are byte-identical | `test_custody_hash.py::test_two_runs_over_an_unchanged_source_are_byte_identical` |
| a partial pass never claims completion | `test_custody_hash.py::test_a_partial_pass_does_not_claim_completion` |
| a resume reports the ledger total, not the pass delta | `test_custody_hash.py::test_a_resume_reports_the_ledger_total_not_the_pass_delta`, `test_custody_verify.py::test_a_verification_resume_reports_the_ledger_total` |
| presence is never custody | `test_custody_verify.py::test_a_present_but_corrupt_object_is_a_mismatch` |
| a truncated copy is a mismatch | `test_custody_verify.py::test_a_truncated_object_is_a_mismatch` |
| verification alone cannot release | `test_custody_verify.py::test_the_whole_pipeline_from_measurement_to_correct_refusal` |
| a lock is released even when the holder crashes | `test_custody_lock.py::test_lock_releases_on_exception` |
| a lock held by another process is seen | `test_custody_lock.py::test_a_lock_held_by_another_process_is_detected` |
| live sync stands down for a campaign | `test_custody_lock.py::test_the_sync_script_stands_down_when_a_campaign_is_active` |
| supervisor won't retire a day on a zero exit alone | `test_ingest_supervisor_verification.py::test_a_silent_no_op_day_stays_pending` |
| the probe is verified from the script, not asserted | `test_custody_lock.py::test_the_live_sync_script_consults_the_campaign_marker` |
| reverting the probe fails preflight closed | `test_custody_lock.py::test_preflight_refuses_when_the_live_probe_is_missing` |
| a stale marker is never left behind | `test_custody_lock.py::test_the_marker_appears_and_is_cleared_around_execution` |
| nothing is copied without --execute | `test_custody_execute.py::test_without_execute_nothing_is_copied` |
| the sync acknowledgment is required and recorded | `test_custody_execute.py::test_the_sync_acknowledgment_is_required`, `test_the_acknowledgment_is_recorded_when_given` |
| a killed copy never leaves a short object | `test_custody_execute.py::test_a_killed_executor_never_leaves_a_short_object` |
| an existing destination object is never overwritten | `test_custody_execute.py::test_an_existing_destination_object_is_never_overwritten` |
| present-but-unverified is verified, not re-copied | `test_custody_execute.py::test_present_unverified_objects_are_verified_not_recopied` |
| a key cannot escape the destination | `test_custody_execute.py::test_a_key_escaping_the_destination_is_refused` |
| the source is never modified | `test_custody_execute.py::test_the_source_is_never_modified` |
| nothing is deleted at the destination | `test_custody_execute.py::test_nothing_is_deleted_at_the_destination` |
| the full pipeline reaches SAFE_TO_RELEASE | `test_custody_execute.py::test_a_fully_executed_and_verified_campaign_is_safe_to_release` |
| custody of the wrong volume does not release | `test_custody_execute.py::test_a_missing_digest_keeps_release_closed` |
| a digest-less record is never custody | `test_custody_reconcile.py::test_a_record_without_a_digest_never_counts_as_custody` |
| reconciliation cannot prove destination identity | `test_custody_reconcile.py::test_reconciliation_alone_cannot_prove_destination_identity` |
| machine and gate agree on the required scope | `test_custody_state_machine.py::test_machine_and_gate_agree_on_the_required_count` |
| a narrower scope is never silent | `test_custody_release_gate.py::test_a_narrower_scope_is_never_silent` |
| malformed counts block and say why | `test_custody_state_machine.py::test_a_non_numeric_count_blocks_with_a_malformed_reason` |
| diagnostics survive the import round trip | `test_custody_cli.py::test_the_malformed_count_diagnostic_survives_a_round_trip` |
| diagnostics are visible in the default human mode | `test_custody_cli.py::test_malformed_counts_are_visible_without_json` |
| preview == what import would do | `test_custody_reconcile.py::test_preview_agrees_with_import` |
| the only writes are the two authorized commands | `test_custody_legacy_watchers.py::test_the_only_writes_are_the_two_explicitly_authorized_commands` |
| read commands take no clock/randomness | `test_custody_legacy_watchers.py::test_writing_commands_are_the_only_ones_taking_a_clock` |
| custody does not depend on legacy watchers | `test_custody_legacy_watchers.py` |
| the fixture is not wired to the card | `test_custody_fixture_card01.py` |