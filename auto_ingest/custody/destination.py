"""auto_ingest.custody.destination - canonical destination abstraction.

Three separate concepts, deliberately not collapsed into one string:

1. **Logical custody destination** - a host-independent name plus a relative
   path (e.g. ``primary`` + ``fileserver/dashcam``). This is what a campaign
   record persists and what contracts talk about.
2. **Resolved host path** - where that logical destination lives *on this host,
   right now*. Resolved from ``CUSTODY_DESTINATION_ROOT`` then
   ``custody.destination_root`` in ``config.yaml``. Never hardcoded here: the
   repository contract must not embed one historical mount.
3. **Storage identity evidence** - filesystem UUID / device / label / type,
   recorded so a later run can prove it is talking to the *same* storage even
   if the path moved or a different box was mounted there.

Resolution is fail-closed: with no configured destination, ``host_path`` is
``None`` and every release condition that depends on it blocks. Reading the
mount table is a ``stat`` of one path - no mounting, no unmounting, no writes.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

#: Environment variables consulted for the destination, in precedence order.
DESTINATION_ENV_VARS: Tuple[str, ...] = (
    "CUSTODY_DESTINATION_ROOT",
    "CUSTODY_DESTINATION_PATH",
)

# Logical destination identity. Needed because "primary" cannot describe two
# unrelated archives on one host: the dashcam corpus and the DVD-ripper rips are
# different projects with different landing places, and both resolving to
# `primary:fileserver/dashcam` made them contend for one campaign lock - so a
# second campaign was refused as "locked by another campaign" when the only thing
# they shared was a default.
DESTINATION_NAME_ENV_VARS: Tuple[str, ...] = ("CUSTODY_DESTINATION_NAME",)
DESTINATION_RELATIVE_ENV_VARS: Tuple[str, ...] = (
    "CUSTODY_DESTINATION_RELATIVE_PATH",)

# Declared identity of the destination storage, in the environment so that a
# committed config.yaml stays portable between hosts. Precedence: env then config.
DESTINATION_IDENTITY_ENV_VARS: Tuple[str, ...] = (
    "CUSTODY_DESTINATION_FILESYSTEM_UUID",
    "CUSTODY_DESTINATION_DEVICE",
    "CUSTODY_DESTINATION_FILESYSTEM_TYPE",
)

#: Config keys consulted (inside the top-level ``custody:`` block).
DESTINATION_CONFIG_KEY = "destination_root"
DESTINATION_NAME_KEY = "destination_name"
DESTINATION_RELATIVE_KEY = "destination_relative_path"

# Declared destination identity. An operator states what the destination IS, so
# that an observation has something to be compared against.
#:
# A block filesystem is normally identified by ``destination_filesystem_uuid``.
# A network share cannot be: there is no block device, so no /dev/disk/by-uuid
# entry will ever exist for it, and declaring a UUID would be a claim nothing can
# confirm - which `match_identity` correctly refuses. For those, the share is
# identified by the device string the kernel reports (``//host/share`` for CIFS),
# which names the server and the share and is the strongest thing available.
DESTINATION_IDENTITY_UUID_KEY = "destination_filesystem_uuid"
DESTINATION_IDENTITY_DEVICE_KEY = "destination_device"
DESTINATION_IDENTITY_TYPE_KEY = "destination_filesystem_type"

#: Host-independent default layout of a custody destination under its root.
DESTINATION_DEFAULT_RELATIVE_PATH = "fileserver/dashcam"

SOURCE_ENV = "env"
SOURCE_CONFIG = "config"
SOURCE_UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class StorageIdentity:
    """Evidence about *which storage* was observed, independent of its path."""

    filesystem_uuid: Optional[str] = None
    device: Optional[str] = None
    label: Optional[str] = None
    filesystem_type: Optional[str] = None
    size_bytes: Optional[int] = None
    observed_at: Optional[str] = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "StorageIdentity":
        raw = raw or {}
        size = raw.get("size_bytes")
        return cls(
            filesystem_uuid=_opt_str(raw.get("filesystem_uuid") or raw.get("uuid")),
            device=_opt_str(raw.get("device")),
            label=_opt_str(raw.get("label")),
            filesystem_type=_opt_str(raw.get("filesystem_type") or raw.get("fstype")),
            size_bytes=int(size) if isinstance(size, (int, float)) else None,
            observed_at=_opt_str(raw.get("observed_at")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device": self.device,
            "filesystem_type": self.filesystem_type,
            "filesystem_uuid": self.filesystem_uuid,
            "label": self.label,
            "observed_at": self.observed_at,
            "size_bytes": self.size_bytes,
        }

    @property
    def has_uuid(self) -> bool:
        return bool(self.filesystem_uuid)

    def fingerprint(self) -> str:
        """Stable identity fingerprint. Empty string when nothing is known."""
        blob = "|".join(
            [
                (self.filesystem_uuid or "").strip().upper(),
                (self.device or "").strip(),
                (self.label or "").strip(),
            ]
        )
        if not blob.replace("|", ""):
            return ""
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _opt_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


@dataclass(frozen=True)
class DestinationMatch:
    """Result of comparing two storage identities."""

    matched: bool
    reason: str
    comparable: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {"comparable": self.comparable, "matched": self.matched, "reason": self.reason}


def match_identity(expected: Optional[StorageIdentity],
                   observed: Optional[StorageIdentity]) -> DestinationMatch:
    """Compare an expected destination identity with what was observed.

    Fail closed: an absent side, or two sides with no comparable field, is
    *not* a match.
    """
    if expected is None or observed is None:
        return DestinationMatch(False, "identity_not_proven", comparable=False)
    if expected.has_uuid and observed.has_uuid:
        a = (expected.filesystem_uuid or "").strip().upper()
        b = (observed.filesystem_uuid or "").strip().upper()
        if a != b:
            return DestinationMatch(False, f"filesystem_uuid_mismatch:{a}!={b}")
        return DestinationMatch(True, f"filesystem_uuid_match:{a}")
    # A UUID on one side only is UNPROVABLE, not satisfied by a weaker field.
    # Network destinations (CIFS/SMB) have no block device, so `uuid_for_device`
    # can never supply one: an operator who recorded a UUID for such a share
    # recorded something the observation can never confirm. Falling through to the
    # device comparison anyway reported `matched=True`, so a claim the system
    # cannot check was presented as checked. This mirrors CardIdentity.compare's
    # rule - a field present on one side blocks a match when it is more
    # authoritative than the deciding field, and a UUID is.
    if expected.has_uuid != observed.has_uuid:
        side = "expected" if expected.has_uuid else "observed"
        return DestinationMatch(
            False,
            f"filesystem_uuid_unverifiable:{side}_only",
            comparable=False,
        )
    if expected.device and observed.device and expected.device == observed.device:
        return DestinationMatch(True, f"device_match:{expected.device}", comparable=True)
    return DestinationMatch(False, "no_comparable_identity_field", comparable=False)


@dataclass(frozen=True)
class LogicalDestination:
    """Host-independent identity of a custody destination."""

    name: str = "primary"
    relative_path: str = ""

    @property
    def canonical(self) -> str:
        if not self.relative_path:
            return self.name
        return f"{self.name}:{self.relative_path.rstrip('/')}"

    def to_dict(self) -> Dict[str, Any]:
        return {"canonical": self.canonical, "name": self.name,
                "relative_path": self.relative_path}


@dataclass(frozen=True)
class DestinationRef:
    """A logical destination plus how (and whether) it resolved on this host."""

    logical: LogicalDestination
    host_path: Optional[str] = None
    resolved_from: str = SOURCE_UNRESOLVED
    mounted: Optional[bool] = None
    identity: Optional[StorageIdentity] = None

    @property
    def resolved(self) -> bool:
        return bool(self.host_path)

    def campaign_path(self) -> Optional[str]:
        """The per-campaign subdirectory on this host, if resolvable."""
        if not self.host_path:
            return None
        return str(Path(self.host_path) / self.logical.relative_path) if self.logical.relative_path \
            else self.host_path

    def to_dict(self) -> Dict[str, Any]:
        return {
            "host_path": self.host_path,
            "identity": self.identity.to_dict() if self.identity else None,
            "logical": self.logical.to_dict(),
            "mounted": self.mounted,
            "resolved": self.resolved,
            "resolved_from": self.resolved_from,
        }


def _expand(value: Optional[str], env: Mapping[str, str]) -> Optional[str]:
    """Resolve a ``${VAR}`` / ``$VAR`` placeholder in a configured value."""
    if not value:
        return None
    text = value.strip()
    if text.startswith("${") and text.endswith("}"):
        name = text[2:-1]
        return env.get(name) or None
    if text.startswith("$") and len(text) > 1:
        return env.get(text[1:]) or None
    return text or None


def load_custody_config(repo_root: Optional[str | Path] = None) -> Dict[str, Any]:
    """Read the ``custody:`` block of ``config.yaml`` (empty dict when absent).

    Read-only. Returns an empty mapping rather than raising so callers can fail
    closed on the specific missing piece instead of on a broken config file.
    """
    try:
        import yaml  # type: ignore
    except Exception:  # pragma: no cover - PyYAML is a declared dependency
        return {}
    candidates = []
    if repo_root:
        candidates.append(Path(repo_root) / "config.yaml")
    try:  # the repo-root module knows where config.yaml lives
        import auto_ingest_config  # type: ignore

        found = auto_ingest_config._find_config_path()  # noqa: SLF001
        if found:
            candidates.append(Path(found))
    except Exception:
        pass
    candidates.append(Path.cwd() / "config.yaml")
    for candidate in candidates:
        try:
            if not candidate.exists():
                continue
            data = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        block = data.get("custody")
        if isinstance(block, Mapping):
            return dict(block)
        return {}
    return {}


def resolve_destination(
    config: Optional[Mapping[str, Any]] = None,
    *,
    env: Optional[Mapping[str, str]] = None,
    identity: Optional[StorageIdentity] = None,
    check_mount: bool = True,
) -> DestinationRef:
    """Resolve the configured destination for this host.

    Precedence: ``CUSTODY_DESTINATION_ROOT`` -> ``custody.destination_root``.
    With neither present the destination is *unresolved* (``host_path=None``),
    which blocks release rather than silently falling back to some historical
    mount.
    """
    env = os.environ if env is None else env
    config = config or {}

    host_path = None
    resolved_from = SOURCE_UNRESOLVED
    for var in DESTINATION_ENV_VARS:
        value = _expand(env.get(var), env)
        if value:
            host_path = value
            resolved_from = f"env:{var}"
            break
    if host_path is None:
        value = _expand(config.get(DESTINATION_CONFIG_KEY), env)
        if value:
            host_path = value
            resolved_from = f"config:custody.{DESTINATION_CONFIG_KEY}"

    if identity is None:
        # Declared, never observed. An operator states what the destination is; the
        # gate's job is to check that claim against the kernel, which is the whole
        # point of keeping the two apart.
        declared_uuid = (_opt_str(env.get(DESTINATION_IDENTITY_ENV_VARS[0]))
                        or _opt_str(config.get(DESTINATION_IDENTITY_UUID_KEY)))
        declared_device = (_opt_str(env.get(DESTINATION_IDENTITY_ENV_VARS[1]))
                           or _opt_str(config.get(DESTINATION_IDENTITY_DEVICE_KEY)))
        declared_type = (_opt_str(env.get(DESTINATION_IDENTITY_ENV_VARS[2]))
                         or _opt_str(config.get(DESTINATION_IDENTITY_TYPE_KEY)))
        if declared_uuid or declared_device:
            identity = StorageIdentity(
                filesystem_uuid=declared_uuid,
                device=declared_device,
                filesystem_type=declared_type,
            )

    logical = LogicalDestination(
        name=(_opt_str(env.get(DESTINATION_NAME_ENV_VARS[0]))
              or _opt_str(config.get(DESTINATION_NAME_KEY)) or "primary"),
        relative_path=(_opt_str(env.get(DESTINATION_RELATIVE_ENV_VARS[0]))
                       or _opt_str(config.get(DESTINATION_RELATIVE_KEY))
                       or DESTINATION_DEFAULT_RELATIVE_PATH),
    )

    mounted: Optional[bool] = None
    if host_path is not None:
        if check_mount:
            try:
                mounted = os.path.ismount(host_path)
            except OSError:  # pragma: no cover - ismount does not raise in practice
                mounted = False
            if not mounted:
                # ismount() is false for a directory *inside* a mount, which is the
                # normal shape of a network destination: the share is at /nas, the
                # archive is at /nas/fileserver/headcam. Falling back to isdir()
                # alone would be wrong - an ordinary directory would pass - so the
                # containing mount is consulted instead.
                from .mounts import observe_mount

                mounted = observe_mount(host_path, allow_containing=True).present
        else:
            mounted = None

    return DestinationRef(
        logical=logical,
        host_path=host_path,
        resolved_from=resolved_from,
        mounted=mounted,
        identity=identity,
    )


def load_policy(config: Optional[Mapping[str, Any]] = None,
                env: Optional[Mapping[str, str]] = None) -> "Any":
    """Build the :class:`~auto_ingest.custody.policy.CustodyPolicy` from config."""
    from .policy import CustodyPolicy

    config = config or {}
    raw = config.get("policy")
    if not isinstance(raw, Mapping):
        raw = {}
    policy = CustodyPolicy.from_dict(raw)

    env = os.environ if env is None else env
    witness = _opt_str(env.get("CUSTODY_REQUIRE_OPERATOR_WITNESS"))
    if witness is not None:
        from dataclasses import replace

        policy = replace(
            policy,
            require_operator_witness=witness.strip().lower() in {"1", "true", "yes", "on"},
        )
    return policy


__all__ = [
    "DESTINATION_ENV_VARS",
    "DestinationMatch",
    "DestinationRef",
    "LogicalDestination",
    "StorageIdentity",
    "load_custody_config",
    "load_policy",
    "match_identity",
    "resolve_destination",
]
