# Neo4j backup completion receipts: offline verifier

This change provides a **read-only, standard-library prototype** for evaluating backup completion receipts. It does not run or modify backup jobs, connect to production databases, change retention, or certify restoration.

## Evidence levels

A scheduler invocation proves only that a job started. An unsigned completion receipt reporting exit code zero, size, and SHA-256 is an **untrusted claim**, even if its structure is valid.

For a **cold-mirror** workflow, the mirror's claimed UTC start must be
at or after the primary's claimed completion; overlapping stages are
rejected. This is an unsigned timeline check, not evidence of a job exit.

An independent bounded read of an explicitly selected archive can verify that the file's bytes match the claimed digest. Reading two distinct source and mirror files and matching their digests can establish byte identity. **Neither result authenticates the writer, proves transaction consistency, or verifies Neo4j recovery.**

Real recovery certification needs authenticated producer custody, source database identity, valid transaction-chain evidence, and a successful isolated restore and consistency test.

## Receipt fields

| Field | Requirement |
| --- | --- |
| schema | neo4j-backup-completion-evidence-v1 |
| role | primary or mirror |
| runId | Matching opaque 8–64 character identifier |
| startUtc, finishUtc | Ordered ISO-8601 UTC timestamps ending in Z |
| exitCode | Integer; zero is a claim, not proof |
| archiveBytes | Exact positive integer file length |
| archiveSha256 | 64 lowercase hexadecimal characters |

Do not put secrets, credentials, access tokens, private addresses, arbitrary internal file paths, or database content in receipts. Authentication requires an independently reviewed, pinned trusted signature-verification mechanism; this prototype does not implement that capability.

## Usage and safety

The verifier lives at tools/neo4j_backup_receipt_verifier.py. Supplying only --receipt reads at most 32 KiB of JSON and **no archive content**. The tool rejects duplicate JSON fields, invalid schema values, invalid date order, numeric booleans, and unknown receipt fields.

An optional explicit --archive argument enables a SHA-256 read of a selected regular .backup file. The default ceiling is 16 MiB. A greater budget must be deliberately passed through --max-bytes, with a hard ceiling of 32 GiB. Symlinks, file-size mismatches, changed metadata, malformed receipts, or mismatched hashes fail closed.

The library routine compare_independently_read_files() requires separately read source and mirror files with distinct inodes, matching run IDs, and matching checksums. A successful result confirms observed bytes only—not the writer's identity or a restorable database.

All automated fixtures are tiny synthetic files. Run the tests using:

    PYTHONPATH=tools python3 -m unittest -q tests/test_neo4j_backup_receipt_verifier.py

Do not raise byte limits or read large production archives without explicit operational approval, storage and I/O qualification, an isolated test plan, and a rollback procedure. No mode authorizes archive deletion, changes to backup schedules, database restarts, or model inference.

Producer-side receipt creation, atomic durable publication, trustworthy signatures, job exit verification, and any deployment to actual backup hosts are separate, owner-reviewed work.