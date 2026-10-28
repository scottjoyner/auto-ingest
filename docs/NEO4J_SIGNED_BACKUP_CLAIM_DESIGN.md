# Offline, role-pinned Ed25519 completion claims for Neo4j backups

**Draft protocol — not deployed to a backup owner.** This optional
follow-up extends the separate offline unsigned receipt verifier.
It authenticates a signer holding a preapproved key; it does not prove
process exit, backup consistency, source/mirror byte identity, or restore.

## What the signatures add

An untrusted caller can fabricate a zero exit code, file size,
timestamps, and SHA-256 digest in JSON. That is why the unsigned
receipt is only a claim. A role-pinned Ed25519 signature shows that a
holder of a **previously approved key** signed those exact fields,
but it cannot reveal *how* the signer learned the process result.

The envelope contains exactly schema, keyId, claim and signature.
The claim uses the eight strict v1 unsigned receipt fields.
The signature binds a domain separator, key ID and canonical JSON.
Changing any signed field invalidates the signature. Public keys
are **never accepted from an untrusted envelope**.

Primary and mirror must have distinct approved signer IDs and distinct
public keys, matching run/size/hash claims, zero claimed exits and
a cold-mirror start no earlier than primary finish. A valid pair is
only **matching signed claims**, never byte proof or restore proof.

## Owner-only producer integration: future work

The approved backup owner, not an external monitoring account,
would need to integrate this module into the **existing** job:

1. Establish an immutable unique run ID and observe the stage start.
2. Supervise the existing backup operation and capture its **real**
   process exit status. Merely seeing cron invoke the command is
   insufficient.
3. Close/sync the completed archive, then independently read its
   full digest within an approved I/O budget. A failed exit, missing
   or changed archive, partial transfer or digest mismatch must not
   be signed as success. Signed-v1 deliberately only permits zero
   exit claims. An aborted/failed schema is separate.
4. Keep the stage's approved Ed25519 private key in protected owner
   custody, outside repositories, logs, shell arguments, receipts,
   model traces and operator messages.
5. Publish the small receipt to an **existing owner-only local
   directory**, exclusive 0600 final file, without overwriting
   earlier runs. The publisher creates a private temporary file,
   syncs it, atomically creates a no-overwrite final hardlink and
   removes the temporary name.
6. Observe the independent mirror job's own exit and mirror file
   bytes with a different authorized signing key. Never claim a
   mirror hash merely by copying the primary digest.

The protected job integration, access control, key rotation, key
revocation, receipt collection, clock validation, durable run
sequence and replay detection require separate security and
owner approval. A signed historical claim is not automatically
evidence of a **current** backup. Signer authentication also does
not make a potentially incorrect signer truthful.

## Exactly what the review code does

The standalone tool in tools/neo4j_signed_completion_receipt.py
offers in-memory Ed25519 claim signing, role-pinned verification,
matching signed-claim comparison, bounded JSON loading, and private
atomic publication. Verification reads **only a small supplied claim
and preapproved in-memory public keys**. It does not read archive
contents, connect to Neo4j or SSH, start a backup or model, modify
NAS, or execute any privileged command.

The optional cryptography dependency performs Ed25519. There is
no handwritten signature implementation or insecure fallback.
The accompanying CI modification installs the dependency in both
test environments and checks for an available Ed25519 backend.

The publisher is **never called automatically**. It rejects missing,
symlinked, non-owner and group/world-accessible output directories,
creates only small private 0600 records, refuses duplicates and
cleans incomplete private temporary files after ordinary failures.
A future owner must independently verify that its chosen storage
is local and private rather than an unintended network share.

Synthetic tests use disposable in-memory signing keys and small
fake receipts. Positive results explicitly leave job execution,
archive integrity, cross-host byte identity, graph consistency,
restore success, deletion and all host/model changes unverified.

## Optional explicit signature plus independent byte verification

The offline helper verify_signed_pair_with_readback() composes
the role-pinned signed-claim and independent bounded SHA-256 verifier.
It requires an explicit approved primary/mirror pair, a separately
supplied exact expected run ID, an independently supplied UTC window
of at most 24 hours, two explicitly selected archive paths and a
separately reviewed I/O budget.

Before ANY file access, both signatures must validate against
previously approved distinct public keys, the claimed run must equal
the independently specified expected run, and all primary/mirror
start and finish times must be within the caller's UTC interval.
The caller must obtain that interval and run ID from an independent
owner-controlled schedule or durable job ledger, not echo the claims.

Only then does it invoke the existing bounded independent archive
readback checker. The default per-file I/O cap is 16 MiB, and the
maximum combined read is twice the explicit per-file cap. The
checker requires two distinct regular inodes with stable metadata
and independently matching source and mirror SHA-256 digests.
There is no automatic path discovery, bulk content read, or NAS
write. A larger cap must be explicitly requested and approved.

The strongest result only confirms that TWO role-pinned signatures
and the two explicitly selected file byte streams agreed with their
claimed digests at the time of independent readback. It does NOT
prove the producer observed a real job exit, database generation,
transactional consistency, restorability, deletion eligibility,
or custody of the owner's signing keys.

The run/time scope reduces accidental cross-run replay but is NOT
a durable anti-replay ledger. Trusted issuance, monotonic sequencing,
clock source review, key rotation and revocation remain owner gates.
Synthetic tests cover good separate files, stale/wrong run scopes,
24-hour boundaries, tampered signatures and file bytes, same-inode
hardlink or symlink copies, I/O caps and no implicit archive reads.
No real production archive or owner private key was read.

## Remaining recovery acceptance

A valid signed primary+mirror pair still requires separately
authorized full-file verification of **two distinct source/mirror
files**, trusted database/transaction chain evidence, and a
successful isolated Neo4j restore and consistency check under
reviewed resource limits and rollback controls.

**No automatic merge, owner-account deployment, privileged access,
production backup command, secret creation or NAS write.**

## Offline review follow-up: publication failure semantics

An independent synthetic fault-injection review found that the
original private publisher could raise after its final hardlink had
been created (for example on the directory durability sync), leaving
a visible final signed receipt despite reporting an exception.
This could confuse an owner retrying an otherwise correct job receipt.

The isolated review patch changes publication ordering to:

1. Write a new private 0600 temporary receipt; sync the file.
2. Atomically hardlink to the exclusive, no-overwrite final name.
3. Remove the temporary name.
4. Sync the parent directory **after** the temporary cleanup.

If an error happens after the final link but before the final
directory sync, the publisher attempts **only an inode-matched
rollback** of the newly linked final name, then syncs the
directory again. A failed rollback/sync explicitly raises a
publication-uncertain error and requires inspection of the private
owner directory before any retry. Other existing names are never
deleted or overwritten. The tests exercise a one-shot directory
sync error, persistent sync failures and a temporary-link unlink
failure using synthetic receipts and disposable local directories.

**Important limitation:** Atomic hardlink visibility is not the
same as confirmed persistence, and failed filesystem durability
cannot always be rolled back after a crash or storage error.
A transient final name can be visible during the pre-sync interval.
The producer must never treat an exception as proof that no receipt
ever became visible or durable. A future owner-authorized integration
must reconcile run ID and owner-private final filenames before
retries, and a verifier must still require approved role keys,
independent job-exit custody, independent archive-byte readback
and an actual isolated Neo4j restore.

This remains **offline, synthetic-only** review code. It does not
authorize a production receipt publisher, create signing keys,
read production archives, or change the protected backup jobs.

## Ancestor-symlink publication safety review

The original owner-private publisher used `O_NOFOLLOW` only on the final
receipt directory. An attacker-controlled ancestor symlink could redirect
a syntactically non-symlink final path into a different directory. A
synthetic-only test reproduced the unintended publication successfully.

The publisher now requires an absolute path without parent traversal and
opens every ancestor component via `O_PATH|O_DIRECTORY|O_NOFOLLOW`, pinned
by directory file descriptors. Only the final private directory is opened
for read and synchronization, and all subsequent writes/links/unlinks use
its pinned file descriptor. The function fails closed on a symlink in any
component; a missing directory still reports an ordinary OS error. No
symlinked ancestor is followed even if the symlink points to an existing,
owner-only directory. Synthetic tests cover nested ancestor symlinks,
`..` traversal, relative paths and non-directory ancestors.

**Limitation:** no-follow pathname handling does not establish that the
underlying filesystem is local: bind mounts and network mounts are still
possible, and the owner must independently inspect the actual destination
filesystem and its access controls before any producer integration. The
private publisher remains **undeployed** and offline review code only.
