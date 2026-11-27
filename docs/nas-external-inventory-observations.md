# Read-only NAS reconciliation observations

The fleet's large NAS inventories are **not custody receipts**. A filename and
size match is not a cryptographic identity, and a missing canonical path does
not prove absence from other stores or an archive. Operator-safe observations
belong beside the existing auto_ingest.custody state machine, **not inside
the campaign's proof-bearing destination ledgers**.

## Importing a pinned observation

The library auto_ingest.custody.external_inventory.observe_metadata_csv
accepts a frozen CSV from the read-only NAS metadata scanner with:

* An exact expected row count and independently pinned SHA-256 of the CSV;
  missing, mutated, duplicated or oversized inputs are denied.
* A card UUID, source identity, candidate relative path and expected byte count.
* One of the bounded metadata states (absent, name+size, mismatch, symlink,
  permission/IO uncertainty and so on).

Before summarizing an external CSV, the importer also rejects path aliases
under Unicode NFC normalization and case-folding: two source paths in the same
card, or two target paths in the same campaign, cannot collapse to one portable
name. This safeguard addresses an observed October 16 NAS5 research-vault
collision: SSD names Fig1.png and fig1.png contained different SHA-256
payloads, but a single NAS Fig1.png held only the lowercase variant's data.
The importer fails closed rather than silently accepting such ambiguous paths.
It remains observation-only and does not authorize copying or release.

The importer **does not** read media payloads, generate hash-equality proof,
write campaign bundles or mutate a custody state. It refuses any
nas_payload_sha256_verified_today=True or source_release_authorized=True
assertions. These require the existing independent custody verification and
release gates.

Read-only example:

    from auto_ingest.custody.external_inventory import observe_metadata_csv

    result = observe_metadata_csv(
        "/home/scott/nas-reconciliation-audit-20261016/sd-exact-path-fd-safe-repro-20261016.csv",
        expected_sha256="e0026bdc3975f1f0f7fd5db529edb65105918c900dc89655e9c2a1ffc822d484",
        expected_rows=4268,
    )
    assert result.destination_verified_files == 0
    assert not result.source_release_allowed

This is a **historical observation snapshot**, not continuous monitoring.
Before planning new writes, independently check the current source identity,
physical NAS identity, hash receipts and conflicting active writers. No source
release, archival cleanup, or transfer is authorized by this importer.

Research-vault repository directory and file queues are a distinct dataset.
Do not import them as SD-card campaigns: they require their own typed contract
without implicit source-release semantics.

Run the isolated 16-test module via:

    PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q tests/test_custody_external_inventory.py

The tests use synthetic temporary CSVs only, with no NAS mount required.
See also docs/sd-card-campaign-custody.md for the canonical policy.
