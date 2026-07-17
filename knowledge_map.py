#!/usr/bin/env python3
"""Repo-root entry point so ``python3 -m knowledge_map`` resolves (K-N7).

The real implementation lives in ``auto_ingest/knowledge_map.py`` (a thin
wrapper over ``scripts/knowledge_harvest*.py``). This shim just puts the repo
root / package on the path and delegates, so the documented
``python3 -m knowledge_map sync_vault_to_neo4j`` command works from the repo
root without a ModuleNotFoundError.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from auto_ingest.knowledge_map import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
