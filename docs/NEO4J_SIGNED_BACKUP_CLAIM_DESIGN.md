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

## Remaining recovery acceptance

A valid signed primary+mirror pair still requires separately
authorized full-file verification of **two distinct source/mirror
files**, trusted database/transaction chain evidence, and a
successful isolated Neo4j restore and consistency check under
reviewed resource limits and rollback controls.

**No automatic merge, owner-account deployment, privileged access,
production backup command, secret creation or NAS write.**
