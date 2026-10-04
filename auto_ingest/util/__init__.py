"""Small, dependency-free helpers shared across the repo.

Unlike ``auto_ingest.ingest`` / ``diarize`` / ``dashcam`` / ``content``, nothing
here imports an ML stack, a database driver, or the network, so it is safe to
import from a cron-invoked script that must start even when its dependencies
are broken:

  from auto_ingest.util.atomic import write_json_atomic
"""
from __future__ import annotations

__all__ = ["atomic"]
