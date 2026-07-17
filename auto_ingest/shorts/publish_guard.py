"""Hard safety guard for live publishing.

Publishing is intentionally HELD until accounts/OAuth are deliberately set up.
This module enforces that: a real upload is only permitted when
``AUTO_INGEST_LIVE=1`` is explicitly set AND at least one platform's
credentials are present in the environment. Everything else (the default) is a
safe dry-run / disk-staging operation that never touches a network API.

Call :func:`require_live_mode` at the very top of any code path that would
perform a real upload. It raises :class:`LivePublishForbidden` otherwise.
"""
from __future__ import annotations

import os
from typing import List

LIVE_ENV = "AUTO_INGEST_LIVE"

# Per-platform credential env vars that, when present, mean "this platform is
# wired for a real upload". Mirrors uploader.ADAPTERS ``.needs``.
PLATFORM_CREDS = {
    "youtube_shorts": ["YT_TOKEN_JSON", "YT_CLIENT_SECRET_JSON"],
    "tiktok": ["TIKTOK_ACCESS_TOKEN"],
    "instagram": ["IG_ACCESS_TOKEN", "IG_USER_ID"],
}


class LivePublishForbidden(RuntimeError):
    """Raised when a live upload is attempted without explicit opt-in."""


def configured_platforms() -> List[str]:
    """Return the platforms that have credentials present in the env."""
    out: List[str] = []
    for plat, vars_ in PLATFORM_CREDS.items():
        if any(os.getenv(v) for v in vars_):
            out.append(plat)
    return out


def is_live_opt_in() -> bool:
    return os.getenv(LIVE_ENV, "").strip() in ("1", "true", "yes")


def safe_to_run() -> bool:
    """True only when an explicit live opt-in AND >=1 platform is configured."""
    return is_live_opt_in() and bool(configured_platforms())


def require_live_mode(where: str = "upload") -> None:
    """Raise :class:`LivePublishForbidden` unless a real upload is explicitly allowed.

    ``where`` is a label used in the error/warning message (e.g. the command).
    """
    if is_live_opt_in() and not configured_platforms():
        raise LivePublishForbidden(
            f"Refusing live {where}: AUTO_INGEST_LIVE=1 set but no platform "
            f"credentials are present in the environment. Configure one of "
            f"{sorted(PLATFORM_CREDS)} first, or run without AUTO_INGEST_LIVE "
            f"for a safe dry-run.")
    if not safe_to_run():
        raise LivePublishForbidden(
            f"Live {where} is forbidden by default (publishing is held). "
            f"Set AUTO_INGEST_LIVE=1 in the environment AND configure at least "
            f"one platform's credentials. All other runs are safe dry-runs.")
