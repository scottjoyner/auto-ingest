# Fleet storage namespace contract

> Status: implementation contract for the storage-namespace convergence slice.
> Scope: path semantics only. This document does not authorize a mount change,
> Samba restart, Beelink pool remount, data move, deletion, or production rollout.

## Core rule

Auto-ingest now distinguishes three different storage concepts instead of using
`fileserver_root` as an overloaded synonym for all of them:

| Concept | Resolver/config | Meaning |
|---|---|---|
| Shared data plane | `shared_root`, `get_shared_root()` | Fleet-wide canonical namespace, normally `/nas` |
| Hot/local processing | `hot_ssd_root`, `get_hot_root()` | Host-local SSD/HDD working tier |
| Local fileserver/processing compatibility | `fileserver_root`, `get_fileserver_root()` | Existing host-local processing tree while callers are migrated deliberately |
| Durable/cold root | `nas_mirror_root`, `get_cold_root()` | Durable shared tier; currently resolves to `/nas` |

The stable cross-host address is:

```text
/nas/<relative-path>
```

A local cache or processing path is not authoritative merely because it contains
a copy of the same bytes.

## Why this is an evolution, not a rewrite

Auto-ingest already had:

- a container-side `/nas` convention;
- machine-aware path resolution in `auto_ingest_config.py`;
- distributed `:IngestJob` coordination;
- replayable stage intermediates;
- a durable graph outbox;
- knowledge-vault <-> Neo4j projection;
- custody abstractions that separate logical destination, resolved host path,
  and storage identity.

This contract promotes the existing `/nas` idea to a fleet-wide host and
container namespace while preserving the existing local-fast-storage design.

## Host behavior

### x1-370

Current intended resolution:

```text
shared_root       /nas
cold_root         /nas
fileserver_root   /media/scott/SSD_4TB/fileserver
hot_root          /media/scott/SSD_4TB/audio
```

Observed during this slice:

```text
/nas -> //192.168.1.202/fileserver (CIFS)
local processing filesystem -> /dev/nvme1n1p6 (ext4)
```

Therefore x1 can address durable shared bytes through `/nas` while retaining
SSD-class local processing.

### Deathstar

The committed profile intentionally keeps:

```text
fileserver_root   /mnt/8TB_2025/fileserver
hot_root          /mnt/8TB_2025/fileserver/audio
shared_root       /nas
cold_root         /nas
```

This preserves the local first-cache/processing model while making durable
promotion and shared access use the same fleet path.

### Beelink transition exception

Beelink physically owns the NAS, but today its `/nas` is the Btrfs pool root,
while clients mount Beelink's `/nas/fileserver` share as their `/nas`.

That means the same literal path can still identify different bytes on Beelink
versus clients.

Do not deploy this host-side contract to Beelink as though that exception were
already resolved.

The long-term target is:

```text
physical Btrfs pool -> /srv/nas-pool
fleet namespace     -> /nas
Samba fileserver    -> /nas
```

The maintenance-window cutover needs its own identity verification, rollback
plan, and service review.

## Canonical custody location

The default logical custody destination is now:

```text
primary:archive/dashcam
```

With a client-side custody root of `/nas`, the resulting path is:

```text
/nas/archive/dashcam
```

Custody still keeps the destination root and relative path separate. Existing
campaign bundles that explicitly record legacy destinations such as
`primary:fileserver/dashcam` remain valid historical records and are not
rewritten.

The custody release gate remains authoritative for byte custody. An
`:IngestJob` stage or Neo4j record is not proof that bytes were copied and
verified.

## Knowledge vault

The durable/canonical vault projection is addressed through:

```text
/nas/shared-knowledge
```

The active local knowledge checkout remains local. This keeps the human-editable
control plane responsive while giving the fleet one shared durable address.

## Container contract

The example container profiles use `/nas`-relative paths for shared data:

```text
NAS_ROOT=/nas
DROP_ROOT=/nas/drop
DASHCAM_ROOT=/nas/dashcam
BODYCAM_ROOT=/nas/bodycam
...
```

This slice deliberately does **not** invent a `/hot` container mount.

If a service needs SSD-class working storage inside a container, add an explicit
local-working-tier mount in a separate change after auditing that service's
read/write behavior. Do not make `/nas` pretend to be both the durable shared
tier and the local hot tier.

## Resolver safety

New shared-storage code should use:

```python
from auto_ingest_config import get_shared_root, get_shared_path
```

The shared resolver:

- prefers `NAS_ROOT` over the machine profile;
- fails closed when no shared root is configured;
- rejects an empty unmounted directory as a valid storage root;
- rejects relative paths that escape the shared namespace.

The local processing, hot, and cold resolvers also no longer silently probe old
NAS/SSD locations when their configured roots are absent.

This removes a major historical failure mode: an old directory that happens to
exist must not silently become the destination.

## Neo4j is not storage authority

The intended future path is:

```text
knowledge repo
  -> bounded semantic projection
Neo4j
  -> narrow read-only MCP discovery
agent
  -> requests payload only if required
/nas
  -> optional bounded local working set
local SSD/NVMe
```

Neo4j can index location, provenance, lifecycle, and processing state. It does
not independently prove byte presence or custody.

Before adding new storage labels, audit the live graph for existing file/media,
source, manifest, path, device, and artifact concepts.

## Current graph-audit blocker

A read-only live graph audit attempted during this slice found runtime/config
drift on x1:

- committed auto-ingest config resolves x1 Neo4j to `127.0.0.1:7687`;
- the healthy Docker Neo4j service is currently published on host port `7688`;
- the configured auto-ingest credential does not authenticate to that live
  endpoint.

No secret was extracted or exposed to bypass this.

Resolve the Neo4j endpoint/auth contract as a separate bounded change before
using live schema results to design the storage projection.

## Branch reconciliation notes

The October custody state-machine branch has already been merged into main.

Do **not** merge `housekeeping/storage-path-audit` wholesale: it is based on the
August mainline and overlaps many files changed by the custody work. Reuse its
fail-closed resolver and test ideas selectively.

The older `dashcam-card-custody` branch adds a standalone copy/hash/verify tool.
Its behavior now overlaps heavily with the merged custody package. Treat it as
prior art until a narrow gap analysis proves functionality that the package does
not provide.

## Validation

On the convergence branch:

- focused namespace/config/custody suite: 202 passed;
- live x1 resolver check proved shared `/nas` + local SSD processing split;
- full repository suite: 792 passed, 6 skipped, 4 failed;
- the same four failures reproduce on current main in
  `auto_ingest/tests/test_ml_pure.py`, so this slice adds zero new failures.

The four baseline failures are outside this storage slice and should not be
silently repaired here.
