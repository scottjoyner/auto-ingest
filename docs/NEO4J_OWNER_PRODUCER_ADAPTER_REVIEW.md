# Owner-controlled Neo4j backup receipt adapter — review only

**Status: experimental and undeployed.** No production backup schedule,
producer signing key, or protected account is configured by this code.
The currently installed private owner backup scripts and completion logs
were not readable to the authorized review account.

## Purpose and interface

The signed-claim verifier authenticates a claim, but not the actual
producer job or its exit. The experimental owner adapter provides one
narrower observation: it can launch **one explicitly preapproved local
executable** with shell=False, wait for its **direct** return code,
then hash a newly created, explicitly named backup archive subject to
an operator-specified byte ceiling, and publish an owner-signed claim.

There is NO executable CLI or scheduler entry point. The public callable
run_owner_stage has owner_execute=False by default. A trusted operator
must explicitly supply the script/approved-root, fresh run ID,
new archive filename/approved root, preexisting owner-private receipt
directory, conservative I/O ceiling and timeout, and an owner-held
role-specific signer/key ID. Calling with owner_execute=True is a
**real job execution** and is not authorized by passing tests or this
design document. Even a subsequent failure can leave files or child
processes behind. Once a subprocess is attempted, the report retains
jobProcessInvoked=True, regardless of final success.

## Fail-closed evidence rules

* The script must be a single regular, single-link, owner-executable
  and not group/world-writable file under the trusted job directory.
  No shell strings, job arguments, stdin or captured job log output.
* All directory components are opened by pinned no-follow FDs. Paths
  must be absolute without parent traversal. Archives are never
  silently reused. A preexisting archive causes an immediate hold.
* Nonzero or unavailable direct process exit, timeout, missing new
  archive, symlink, nonregular/multi-link archive, oversized file,
  short read, inode/timestamp change or publication error refuses a
  signed success receipt. Interrupted archive reads report an UNKNOWN
  byte count, not a fictitious zero.
* A mirror job requires independently approved signed primary claim,
  matching run ID, post-primary start, and an exactly matching size
  and SHA-256 of a separately newly created mirror archive. Two
  distinct role keys and subsequent independent pair verification
  remain required.
* Owner-private publisher uses exclusive no-overwrite hardlinks,
  ancestor-no-follow, and inode-checked rollback on sync failures.
  Storage failures can still leave ambiguous data; the owner must
  inspect the private directory before retries.

## Important limits

The adapter records the direct child return code, not all descendants.
A shell script that forks a long-running backup and exits early can
mislead the observer. A concurrent writer could also create the
new archive within the observed interval. Producer isolation, locking,
trusted process tree completion and source generation must be reviewed.
Executable pathname identity is checked before and after launch, but
there remains a TOCTOU window at exec without fd-bound execution.

No proof of archive content consistency, complete transaction chain,
true source/mirror equivalence, durable writer identity or successful
isolated Neo4j restore follows from this observation. A hash tied to a
signed claim must still be independently verified from both sources.
It does not establish filesystem locality, key custody, encryption,
clock trust, replay prevention, key rotation or revocation.

This module cannot enable swap, clear inference admission, restart
production Neo4j, delete archives, modify retention or run a restore.
The owner must separately approve any real producer integration,
heavy archive read budgets, secure signing keys, task isolation, local
versus bind/network mount policy and independent restore acceptance.

## Offline regression

    python -m pytest -q tests/test_neo4j_owner_job_adapter.py

Only disposable local shell scripts and tiny synthetic backup files
are used in tests, with randomly generated test-only Ed25519 keys.
The combined signed/unsigned/custody suites and hosted CI must pass
before independent security review. No real producer or archive
was accessed to create this proposal.
