"""Thin ``knowledge_map`` module wrapper (K-N7 / DEEP_DIVE §3.7 plan).

The docs (HLD/LLD) and live scripts reference ``python3 -m knowledge_map`` but
no such module existed — the real sync code is ``scripts/knowledge_harvest*.py``.
This module is a THIN WRAPPER that:

  * makes ``python3 -m knowledge_map --help`` work (no ModuleNotFoundError),
  * exposes ``sync_vault_to_neo4j`` (delegates to the existing knowledge-harvest
    pipeline) and ``sync_neo4j_to_vault`` (placeholder, clearly flagged),
  * does NOT rewrite the existing harvesting logic — it imports and runs it.

It is intentionally dependency-light at import time so the module loads under
system python (neo4j is imported lazily inside the subcommands).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

__all__ = ["sync_vault_to_neo4j", "sync_neo4j_to_vault", "main"]


def sync_vault_to_neo4j(config: str | None = None, dry_run: bool = False) -> int:
    """Run the vault -> Neo4j knowledge sync via the existing harvest pipeline.

    The real implementation lives in ``scripts/knowledge_harvest.py``
    (idle-agent KG harvesting + report writing). We delegate to its ``main``.
    """
    logging.info("knowledge_map.sync_vault_to_neo4j: delegating to knowledge_harvest")
    scripts = REPO_ROOT / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import knowledge_harvest  # noqa: WPS433 (lazy, runtime delegation)

    if dry_run:
        logging.info("[dry-run] would run knowledge_harvest.main()")
        return 0
    return knowledge_harvest.main()


def sync_neo4j_to_vault(config: str | None = None, dry_run: bool = False) -> int:
    """Placeholder for the Neo4j -> vault direction.

    Not yet implemented as a distinct pass; flagged clearly so callers/script
    references resolve without error rather than raising ModuleNotFoundError.
    """
    logging.warning(
        "knowledge_map.sync_neo4j_to_vault: not implemented yet (vault->neo4j "
        "via knowledge_harvest is the active path). No-op."
    )
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        prog="knowledge_map",
        description="Thin wrapper over scripts/knowledge_harvest*.py (K-N7).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    s1 = sub.add_parser("sync_vault_to_neo4j", help="Sync vault -> Neo4j (delegates to knowledge_harvest).")
    s1.add_argument("--config", type=str, default=None, help="Path to config.yaml (passed through).")
    s1.add_argument("--dry-run", action="store_true", help="Do not run the harvest; just report intent.")
    s2 = sub.add_parser("sync_neo4j_to_vault", help="Neo4j -> vault (placeholder, no-op).")
    s2.add_argument("--config", type=str, default=None)
    s2.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.cmd == "sync_vault_to_neo4j":
        return sync_vault_to_neo4j(config=args.config, dry_run=args.dry_run)
    if args.cmd == "sync_neo4j_to_vault":
        return sync_neo4j_to_vault(config=args.config, dry_run=args.dry_run)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
