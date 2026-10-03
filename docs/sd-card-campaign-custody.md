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
| 0 | arithmetic contradiction, fatal error, destination identity conflict | `BLOCKED` |
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
`device`. A matching UUID is enough even when the card moved to another USB slot
(the device node changes); a *different* UUID never matches; and when a stored
record carries a UUID the current observation does not, reuse is refused because
sameness is unprovable.

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
`excludes_verified=true` with an estimate that already subtracts the verified
objects. Copying a byte-identical file again is wasted I/O and pointless churn.

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

Hash exemptions only count when `custody.policy.declared_hash_exemptions`
declares the pattern; an exemption appearing only in evidence is a blocker
(`hash_exemptions_not_declared`).

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
auto-ingest custody new     --bundle /path/to/campaign --card-id CARD-02 \
  --uuid "$(blkid -s UUID -o value /dev/sdb1)" --device /dev/sdb1 \
  --label UNTITLED --mount /media/scott/UNTITLED --read-only --apply
auto-ingest custody status  --bundle /path/to/campaign          # human
auto-ingest custody status  --bundle /path/to/campaign --json   # machine
auto-ingest custody plan    --bundle /path/to/campaign --json
auto-ingest custody verify  --bundle /path/to/campaign          # describes; --execute is refused
auto-ingest custody import  --bundle /path/to/campaign --evidence ev.json   # validate only
auto-ingest custody import  --bundle /path/to/campaign --evidence ev.json --apply
```

Observed hardware can be cross-checked inline:

```bash
auto-ingest custody status --bundle ... --json \
  --observed-uuid "$(...)" --observed-device /dev/sdb1 --observed-label UNTITLED
```

`observed_card_matches_campaign: false` means a different card is sitting where
this campaign's card used to be.

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
  → custody verify             # describe/confirm the verification set
  → custody status             # has custody been proven?
```

Each step is a separate, individually authorized command. `new` records who the
card is, `import` records what has been observed about it, `status`/`plan` read,
and only an explicitly authorized executor moves bytes.

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
| the only writes are the two authorized commands | `test_custody_legacy_watchers.py::test_the_only_writes_are_the_two_explicitly_authorized_commands` |
| read commands take no clock/randomness | `test_custody_legacy_watchers.py::test_writing_commands_are_the_only_ones_taking_a_clock` |
| custody does not depend on legacy watchers | `test_custody_legacy_watchers.py` |
| the fixture is not wired to the card | `test_custody_fixture_card01.py` |