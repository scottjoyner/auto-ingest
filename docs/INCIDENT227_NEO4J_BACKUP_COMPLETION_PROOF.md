# Incident-227 — Neo4j primary/mirror completion proof (review-only)

October 27, 2026. This offline prototype does not create a backup,
alter the fleet, or certify a restoration.

## Actual observed control plane

Deathstar cron.service logged backup_bodycam_kg.sh invoked at 02:27:01
EDT and mirror_bodycam_to_nas5.sh invoked at 03:27:01 EDT on October 27,
verified with the pre-existing SSH alias and strict host-key checks.
The runbook documents primary output to Deathstar's local HDD and a
second-stage rsync mirror to NAS5. Cron CMD does **not** prove exit success,
copy completion, byte identity, database consistency or restoration.

The NAS currently shows 42 bodycam backup run directories, each with
one approximately 29.58 GiB archive, against a historical runbook mirror
retention target of three. None of these copies may be pruned without
independent policy/custody review. A NAS archive's 03:00 mtime likely
reflects primary output, not completion of the 03:27 mirror.

The current trusted SSH account cannot read the protected local HDD
primary backup tree or redirected primary/mirror logs. No permission
bypass, alternate identity or production backup modification is approved.

## Proposed unsigned completion receipt contract

A future instrumented primary and mirror job would emit one atomic,
durable, minimal receipt after checked command completion. This requires
owner review and deployment. Fields must be exact:

| Field | Requirement |
|---|---|
| schema | neo4j-backup-completion-evidence-v1 |
| role | primary or mirror |
| runId | same opaque 8-64 character ID for both stages |
| startUtc and finishUtc | UTC ISO-8601 ending Z, ordered |
| exitCode | integer; zero only if that stage reports success |
| archiveBytes | positive integer equal to exact file length |
| archiveSha256 | 64 lowercase hex characters |

No secrets, credentials, commands, raw graph data, account names, client
addresses or unredacted private filesystem paths belong in the receipt.

## Evidence levels and deliberate limits

**Self-reported claim:** The strict validator checks schema, timing,
digest and exit-code shapes. Even zero claimed exit code is reported only
as unsigned-success-claim-not-certified. The JSON can be forged.

**Independent local readback:** A selected regular .backup file may be
hashed only when the operator supplies its path and an I/O ceiling.
Default: 16 MiB; absolute cap: 32 GiB. Wrong size, symlinks, a changed
inode/mtime, unsafe caps or digest mismatch fail closed. A matching
local hash does NOT authenticate Deathstar or certify Neo4j restoration.

**Primary versus mirror byte identity:** The offline pair verifier
requires matching stage claims, two distinct regular file inodes and
separate full-file reads below the selected cap. It rejects same-file
and hardlink aliases, absent primary evidence and corrupted mirrors.
Matching bytes still does NOT establish trusted writer identity,
signed custody, transaction consistency or a successful restore.

For disaster-recovery certification, additionally require a pinned
trusted signature verifier (e.g. Ed25519) authenticating the job owner,
the correct source database/transaction generation, independent
checksum reads and a successful controlled offline Neo4j restore and
consistency test. Do not store signing private keys in this repository.

## Commands, test and rollout constraints

The stdlib-only verifier in tools/neo4j_backup_receipt_verifier.py
reads zero archive bytes with receipt-only mode. Synthetic example:

~~~sh
python3 tools/neo4j_backup_receipt_verifier.py --receipt ./fixture-receipt.json
python3 tools/neo4j_backup_receipt_verifier.py \
  --receipt ./fixture-receipt.json --archive ./fixture.backup \
  --max-bytes 16777216
~~~

Exit 0 on file-check mode means only that selected file's SHA-256
matches an unsigned receipt. Receipt-only mode returns nonzero.

Run isolated unit tests:

~~~sh
PYTHONPATH=tools python3 -m unittest -q tests/test_neo4j_backup_receipt_verifier.py
~~~

The tests use synthetic tiny files, never NAS or production Neo4j.
No job is installed on Deathstar, no cron is edited and no backup bytes
are read unless a future operator explicitly selects a file.

Do not hash both production 30-GiB copies until an authorized owner
can safely expose the trusted primary file, sufficient free storage and
CPU/I/O budgets are independently verified, and the low-swap x1-370
host is protected from further pressure. The earlier graph consistency
test exited 137 under its isolated memory limit. Current low free swap
remains below the independent 2 GiB new-model admission fence.

**This is a review candidate only.** Future producer instrumentation
needs an owner-approved integration design, atomic receipt output,
constrained signed key handling, failure-injection tests and rollback.
The Incident-173 privacy/credential HOLD, disabled archive-export timers
and existing model-load restrictions remain independent and unchanged.
