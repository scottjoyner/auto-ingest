"""auto_ingest.custody.policy - the configured custody contract.

The policy is the *only* place where a deployment may relax the fail-closed
release gate. It is intentionally boring: a frozen dataclass built from a plain
dict (usually the ``custody.policy`` block of ``config.yaml``), with every
knob defaulting to the strict answer.

Default posture (preservation-first):

* every inventoried object must have a verified destination copy;
* hash evidence must be complete, or explicitly exempted by a pattern that is
  itself declared in the policy (a policy that never declares an exemption
  cannot produce one);
* a stopped worker is never evidence of success or failure on its own;
* no operator witness is required by default, but one can be demanded.

Nothing here reads the clock, the network, or the filesystem, so policy
resolution is deterministic and safe to call from tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

#: Cap on any free-form summary list carried in a campaign bundle. The campaign
#: summary must stay bounded; per-file detail belongs in the external ledgers.
MAX_SUMMARY_ENTRIES = 20


@dataclass(frozen=True)
class CustodyPolicy:
    """Release-gate configuration for SD-card ingest campaigns."""

    #: Which objects the campaign is responsible for. ``all_inventory`` means
    #: every discovered source object; ``hashed_set`` means only the objects
    #: covered by hash evidence (weaker, opt-in).
    required_scope: str = "all_inventory"

    #: Whether hash evidence may be completed by explicit exemptions.
    allow_hash_exemptions: bool = True

    #: The ONLY exemption patterns honoured. An exemption present in evidence
    #: but absent here never counts toward coverage.
    declared_hash_exemptions: Tuple[str, ...] = ()

    #: Key patterns the operator declares OUT OF SCOPE for this campaign, so the
    #: walk skips them. Empty by default: nothing is out of scope until someone
    #: says so.
    #:
    #: This exists for filesystem bookkeeping that is not campaign content - on
    #: the real card, `.Trashes/` and `System Volume Information/` were 3,831 of
    #: 67,644 objects. Copying macOS AppleDouble stubs into the archive is
    #: pollution, and hashing them is wasted passes.
    #:
    #: The same rule as hash exemptions applies, and it is the reason this is a
    #: policy field and not a hard-coded filter: an exclusion that appears in
    #: evidence but is absent here never counts. A typo in a pattern therefore
    #: cannot quietly shrink the campaign - the gate reports it instead. Patterns
    #: match custody keys (POSIX-relative paths) with fnmatch semantics.
    declared_source_exclusions: Tuple[str, ...] = ()

    #: Require a recorded operator witness before source release.
    require_operator_witness: bool = False

    #: Treat extra destination objects as a release blocker instead of a warning.
    strict_destination_scope: bool = False

    #: Require proven storage identity for the destination before release.
    require_destination_identity: bool = True

    #: Require the source mount to be read-only before release.
    require_read_only_source: bool = True

    #: Require the resolved destination host path to be a real mount.
    require_mounted_destination: bool = True

    #: Bound on summary lists kept inside the campaign summary.
    max_summary_entries: int = MAX_SUMMARY_ENTRIES

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "CustodyPolicy":
        """Build a policy from a mapping, ignoring unknown keys."""
        raw = raw or {}
        known = {
            "required_scope",
            "allow_hash_exemptions",
            "declared_hash_exemptions",
            "declared_source_exclusions",
            "require_operator_witness",
            "strict_destination_scope",
            "require_destination_identity",
            "require_read_only_source",
            "require_mounted_destination",
            "max_summary_entries",
        }
        kwargs: Dict[str, Any] = {}
        for key in known:
            if key in raw and raw[key] is not None:
                kwargs[key] = raw[key]
        for field_name in ("declared_hash_exemptions", "declared_source_exclusions"):
            if field_name in kwargs:
                value = kwargs[field_name]
                if isinstance(value, str):
                    value = [value]
                kwargs[field_name] = tuple(str(v) for v in value)
        scope = str(kwargs.get("required_scope", "all_inventory"))
        if scope not in {"all_inventory", "hashed_set"}:
            raise ValueError(
                f"unknown required_scope {scope!r}; expected 'all_inventory' or 'hashed_set'"
            )
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allow_hash_exemptions": self.allow_hash_exemptions,
            "declared_hash_exemptions": list(self.declared_hash_exemptions),
            "declared_source_exclusions": list(self.declared_source_exclusions),
            "max_summary_entries": self.max_summary_entries,
            "require_destination_identity": self.require_destination_identity,
            "require_mounted_destination": self.require_mounted_destination,
            "require_operator_witness": self.require_operator_witness,
            "require_read_only_source": self.require_read_only_source,
            "required_scope": self.required_scope,
            "strict_destination_scope": self.strict_destination_scope,
        }

    # -- scope helpers ----------------------------------------------------
    @property
    def hash_exemptions_allowed(self) -> bool:
        return self.allow_hash_exemptions and bool(self.declared_hash_exemptions)

    def required_objects(self, inventory_files: int, hashed_files: int) -> int:
        """How many objects this policy demands custody for.

        Single definition of "required scope", used by *both* the state machine
        and the release gate. They used to disagree: ``hashed_set`` weakened the
        gate's plan-scope check while the machine still demanded full inventory
        coverage, so a policy knob silently did half of what it said.
        """
        if self.required_scope == "hashed_set":
            return max(hashed_files, 0)
        return max(inventory_files, 0)

    def honour_exemption(self, pattern: str) -> bool:
        """True when ``pattern`` is an exemption this policy actually honours."""
        return self.hash_exemptions_allowed and pattern in self.declared_hash_exemptions

    def excludes_source(self, key: str) -> bool:
        """Whether a custody key is declared out of scope for this campaign.

        fnmatch semantics against POSIX-relative keys, so ``.Trashes/*`` covers
        the directory's contents. Matching a directory pattern (``.Trashes``)
        also excludes everything beneath it, because a caller writing that
        plainly means the whole tree.

        Returns False when nothing is declared: the default campaign covers
        everything the walk finds.
        """
        if not self.declared_source_exclusions:
            return False
        import fnmatch

        for pattern in self.declared_source_exclusions:
            if fnmatch.fnmatch(key, pattern):
                return True
            # A bare directory name covers its contents too.
            prefix = pattern.rstrip("*").rstrip("/")
            if prefix and (key == prefix or key.startswith(prefix + "/")):
                return True
        return False

    def undeclared_exemptions(self, observed: Tuple[str, ...]) -> Tuple[str, ...]:
        """Exemptions claimed by evidence that this policy does not honour."""
        return tuple(p for p in observed if not self.honour_exemption(p))


DEFAULT_POLICY = CustodyPolicy()

__all__ = ["CustodyPolicy", "DEFAULT_POLICY", "MAX_SUMMARY_ENTRIES"]
