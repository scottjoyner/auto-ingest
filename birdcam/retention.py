"""Irreversible clip pruning with validated, config-supplied limits.

``apply_retention`` deletes files permanently. Its two limits come from YAML
(:mod:`birdcam.config`), and ``yaml.safe_load`` happily produces ``0``,
negative numbers, booleans and ``.inf``/``.nan``. None of those is a meaningful
retention policy, and the old arithmetic treated them all as instructions:

* ``max_gb = 0`` or a negative value made ``max_bytes`` non-positive, so the
  size loop never broke and deleted **every** clip;
* ``max_age_days = 0`` or a negative value moved the cutoff to now or into the
  future, so the age loop deleted **every** clip;
* non-finite values crashed mid-pass with ``OverflowError``/``ValueError``
  rather than refusing cleanly.

The limits are now checked at the destructive boundary, before a single
``unlink``: both must be finite, strictly positive, real numbers (not ``bool``,
which is an ``int`` subclass in Python and would otherwise silently mean one
unit). A rejected config raises ``ValueError`` naming the offending field and
leaves the archive untouched. Valid configs keep their exact old behaviour.
"""
from __future__ import annotations

import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Above this, a finite cap cannot be converted to an int byte count without
# overflowing; it is a safe "never prune by size" direction, not an error.
_MAX_GB_FOR_BYTES = sys.float_info.max / (1024 ** 3)
# Far beyond any real clip archive; keeps timedelta() from overflowing on a
# finite but astronomical age limit. Also a safe "never prune by age" direction.
_MAX_AGE_DAYS = 365_000


def _checked_positive(value: object, name: str) -> float:
    """A finite, strictly positive config number, or a named ``ValueError``.

    Validation happens here, at the only place that unlinks files, so a bad
    value can never reach the deletion loops. ``bool`` is rejected explicitly:
    ``retention: {max_age_days: true}`` must not quietly become one day.
    """
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be a finite positive number, got {value!r}") from None
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")
    return number


def apply_retention(root: str, max_gb: float, max_age_days: int, repo=None):
    gb = _checked_positive(max_gb, "max_gb")
    days = _checked_positive(max_age_days, "max_age_days")
    rootp = Path(root)
    now = datetime.utcnow()
    max_bytes = int(gb * (1024 ** 3)) if gb < _MAX_GB_FOR_BYTES else sys.maxsize
    cutoff = now - timedelta(days=min(days, _MAX_AGE_DAYS))
    files = sorted(
        [p for p in rootp.glob("clips/**/*.mp4") if p.is_file()],
        key=lambda p: p.stat().st_mtime,
    )
    removed = []
    for p in list(files):
        if datetime.utcfromtimestamp(p.stat().st_mtime) < cutoff:
            p.unlink(missing_ok=True)
            removed.append(str(p))
            if repo:
                repo.driver.execute("MATCH (c:Clip {path:$path}) SET c.deleted=true,c.deleted_at=$ts,c.deletion_reason='max_age'", path=str(p), ts=now.isoformat())
    files = sorted(
        [p for p in rootp.glob("clips/**/*.mp4") if p.is_file()],
        key=lambda p: p.stat().st_mtime,
    )
    total = sum(p.stat().st_size for p in files)
    for p in files:
        if total <= max_bytes:
            break
        sz = p.stat().st_size
        p.unlink(missing_ok=True)
        total -= sz
        removed.append(str(p))
        if repo:
            repo.driver.execute("MATCH (c:Clip {path:$path}) SET c.deleted=true,c.deleted_at=$ts,c.deletion_reason='max_size'", path=str(p), ts=now.isoformat())
    return removed
