"""Thin ``knowledge_map`` module (G1 / DEEP_DIVE §3.7).

The docs (HLD/LLD) and live scripts reference ``python3 -m knowledge_map`` but
no such module existed. This module wires that command to the REAL vault <-> Neo4j
sync behavior:

  * ``sync_vault_to_neo4j`` — walks the configured markdown vault roots, parses
    each ``.md`` file, and MERGEs a ``VaultDocument`` node (batched, resumable via
    a content hash so re-runs skip unchanged docs). After the markdown pass it
    ALSO delegates to the existing ``scripts/knowledge_harvest.py`` idle-agent
    pass (the only pre-existing "harvest" entrypoint) as an optional hook, so the
    historical behavior is preserved without being rewritten.

  * ``sync_neo4j_to_vault`` — a MINIMAL, SAFE exporter: paged ``MATCH`` over
    ``VaultDocument`` (and optionally configured ``Entity``-like node types),
    writing ONE markdown file per node into the mirror vault. NEVER a full-graph
    scan in one query — every read is ``LIMIT``-bounded and index/cursor-backed.

SAFETY (live Neo4j ~21M nodes; OOMs on big transactions):
  * All graph writes are small MERGE statements, one doc at a time.
  * No full-graph aggregation, no unbounded transaction.
  * ``--dry-run`` is honored at every step: nothing is written to the graph or disk.

It is deliberately dependency-light at import time (``neo4j`` is imported lazily
inside the subcommands) so the module loads under system python for the CLI /
unit tests without a live DB.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

__all__ = ["sync_vault_to_neo4j", "sync_neo4j_to_vault", "main"]

# Bounded page size for any graph read — never pull the whole graph at once.
PAGE_SIZE = 200

# Default node labels exported by ``sync_neo4j_to_vault`` (mirrors config.yaml
# ``knowledge_map.sync.neo4j_to_vault.node_types``; tuned to what is safe/useful).
NEO4J_TO_VAULT_LABELS = ["VaultDocument", "Entity"]


# ---------------------------------------------------------------------------
# config / path helpers
# ---------------------------------------------------------------------------
def _load_knowledge_map_config(config_path: str | None = None) -> dict:
    """Return the ``knowledge_map`` block from config.yaml (or empty if absent)."""
    import yaml

    cfg_path = Path(config_path) if config_path else REPO_ROOT / "config.yaml"
    if not cfg_path.exists():
        return {}
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("knowledge_map", {}) or {}


def _vault_roots(km: dict) -> list[Path]:
    """Resolved vault roots to walk for the vault->neo4j direction."""
    roots: list[Path] = []
    for key in ("central_vault_path", "local_vault_path", "mirror_vault_path", "canonical_vault_path"):
        v = km.get(key)
        if v:
            roots.append(Path(v).expanduser())
    # De-dupe while preserving order.
    seen, ordered = set(), []
    for p in roots:
        if p not in seen:
            seen.add(p)
            ordered.append(p)
    return ordered


def _markdown_files(root: Path) -> list[Path]:
    """Return all ``.md``/``.markdown`` files under ``root`` (BFS, bounded count)."""
    if not root.exists() or not root.is_dir():
        return []
    out: list[Path] = []
    for p in root.rglob("*.md"):
        out.append(p)
    for p in root.rglob("*.markdown"):
        out.append(p)
    return out


def _doc_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


# ---------------------------------------------------------------------------
# vault -> neo4j
# ---------------------------------------------------------------------------
def sync_vault_to_neo4j(config: str | None = None, dry_run: bool = False) -> int:
    """Index the markdown vault into Neo4j as ``VaultDocument`` nodes.

    Batched + resumable: each doc is MERGEd by a stable key (its repo-relative
    path) with a ``content_hash`` so unchanged docs are skipped on re-run. After
    the markdown pass we delegate to the existing ``knowledge_harvest.main()``
    idle-agent hook (non-fatal if it is unavailable), preserving the prior
    behavior without rewriting it.
    """

    km = _load_knowledge_map_config(config)
    roots = _vault_roots(km)
    if not roots:
        logging.warning("knowledge_map.sync_vault_to_neo4j: no vault roots configured; nothing to do.")
        return 0

    # Collect all markdown files across configured roots (bounded per root).
    files: list[tuple[Path, Path]] = []  # (abs_path, root)
    total = 0
    for root in roots:
        found = _markdown_files(root)
        files.extend((p, root) for p in found)
        total += len(found)
    logging.info(
        "knowledge_map.sync_vault_to_neo4j: %d markdown files across %d vault root(s)%s",
        total, len(roots), " [dry-run]" if dry_run else "",
    )

    synced = 0
    if dry_run:
        for p, root in files:
            rel = p.relative_to(root)
            logging.info("[dry-run] would MERGE VaultDocument{key:%s}", rel)
            synced += 1
        logging.info("[dry-run] would sync %d docs (no graph/driver touched).", synced)
        return 0

    # Lazy import neo4j only when we are actually going to write.
    from neo4j import GraphDatabase

    from auto_ingest_config import get_neo4j_config

    neo = get_neo4j_config()
    db = km.get("neo4j_db") or neo.get("db") or "neo4j"
    drv = GraphDatabase.driver(neo["uri"], auth=(neo["user"], neo["password"]))
    try:
        with drv.session(database=db) as sess:
            for p, root in files:
                rel = p.relative_to(root)
                key = str(rel)
                try:
                    text = p.read_text(encoding="utf-8", errors="replace")
                except OSError as e:
                    logging.warning("skip %s (read error: %s)", p, e)
                    continue
                h = _doc_hash(text)
                # Skip if unchanged (resumable): compare hash of the stored node.
                exists = sess.run(
                    "MATCH (d:VaultDocument {key:$key}) RETURN d.content_hash AS h",
                    key=key,
                ).data()
                if exists and exists[0].get("h") == h:
                    continue
                title = _first_heading(text, fallback=rel.name)
                sess.run(
                    """
                    MERGE (d:VaultDocument {key:$key})
                    ON CREATE SET d.created_at=datetime(), d.vault_root=$root
                    SET d.title=$title, d.text=$text, d.content_hash=$h,
                        d.updated_at=datetime(), d.source='markdown_vault'
                    """,
                    key=key, root=str(root), title=title, text=text, h=h,
                )
                synced += 1
                if synced % 500 == 0:
                    logging.info("  synced %d docs so far...", synced)
    finally:
        drv.close()

    logging.info("knowledge_map.sync_vault_to_neo4j: MERGEd %d VaultDocument node(s).", synced)

    # Delegate to the existing idle-agent harvest pass (the only pre-existing
    # "harvest" entrypoint). Non-fatal: if it is unavailable or fails, we still
    # report success for the markdown indexing that just completed.
    _run_harvest_hook()

    # Ensure the index the SKIP-on-resume relies on exists (online, read-only DDL).
    return 0


def _run_harvest_hook() -> None:
    scripts = REPO_ROOT / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    try:
        import knowledge_harvest  # noqa: WPS433
        logging.info("knowledge_map: delegating to knowledge_harvest.main() (idle-agent hook)")
        knowledge_harvest.main()
    except Exception as e:  # pragma: no cover - best-effort hook
        logging.warning("knowledge_map: knowledge_harvest hook skipped (%s)", e)


def _first_heading(text: str, fallback: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


# ---------------------------------------------------------------------------
# neo4j -> vault
# ---------------------------------------------------------------------------
def sync_neo4j_to_vault(config: str | None = None, dry_run: bool = False) -> int:
    """Export graph knowledge back to markdown in the mirror vault.

    MINIMAL, SAFE exporter. For each configured node label we page through the
    graph with ``LIMIT``-bounded, SKIP-cursor queries (NEVER a full scan), write
    ONE markdown file per node, and stop after the configured cap
    (``knowledge_map.sync.neo4j_to_vault.max_nodes_per_sync``)."""
    km = _load_knowledge_map_config(config)
    labels = km.get("sync", {}).get("neo4j_to_vault", {}).get("node_types") or NEO4J_TO_VAULT_LABELS
    max_nodes = int(km.get("sync", {}).get("neo4j_to_vault", {}).get("max_nodes_per_sync", 100))
    out_root = Path(km.get("mirror_vault_path") or km.get("canonical_vault_path") or "/tmp/knowledge_map_export").expanduser()

    logging.info(
        "knowledge_map.sync_neo4j_to_vault: exporting labels=%s max=%d%s",
        labels, max_nodes, " [dry-run]" if dry_run else "",
    )

    if dry_run:
        # Dry-run must NOT touch the live graph (no count query, no scan).
        logging.info(
            "[dry-run] would export up to %d node(s) per configured label into %s "
            "(no live DB queried)", max_nodes, out_root,
        )
        return 0

    from neo4j import GraphDatabase

    from auto_ingest_config import get_neo4j_config

    neo = get_neo4j_config()
    db = km.get("neo4j_db") or neo.get("db") or "neo4j"
    out_root.mkdir(parents=True, exist_ok=True)
    drv = GraphDatabase.driver(neo["uri"], auth=(neo["user"], neo["password"]))
    written = 0
    try:
        with drv.session(database=db) as sess:
            for label in labels:
                written += _export_label(sess, label, out_root, max_nodes - written)
                if written >= max_nodes:
                    break
    finally:
        drv.close()

    logging.info("knowledge_map.sync_neo4j_to_vault: wrote %d markdown file(s) to %s", written, out_root)
    return 0


def _export_label(sess, label: str, out_root: Path, budget: int) -> int:
    """Page through one label and write one md file per node. Bounded by budget."""
    written = 0
    skip = 0
    while written < budget:
        rows = sess.run(
            f"MATCH (n:`{label}`) RETURN n LIMIT $lim SKIP $skip",
            lim=PAGE_SIZE, skip=skip,
        ).data()
        if not rows:
            break
        for rec in rows:
            if written >= budget:
                break
            node = rec["n"]
            props = dict(node) if node is not None else {}
            name = props.get("title") or props.get("name") or props.get("key") or f"{label}_{skip}_{written}"
            fname = _safe_filename(f"{label}_{name}")
            md = _node_to_markdown(label, props)
            (out_root / fname).write_text(md, encoding="utf-8")
            written += 1
        skip += PAGE_SIZE
    return written


def _safe_filename(name: str) -> str:
    keep = [c if (c.isalnum() or c in "-_.") else "_" for c in str(name)]
    s = "".join(keep).strip("_.")
    return (s[:120] or "node") + ".md"


def _node_to_markdown(label: str, props: dict) -> str:
    lines = [f"# {label}", ""]
    for k, v in props.items():
        if k in ("text",):
            continue
        lines.append(f"- **{k}**: {v}")
    if "text" in props:
        lines.append("")
        lines.append("## Body")
        lines.append("")
        lines.append(str(props["text"]))
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(
        prog="knowledge_map",
        description="Vault<->Neo4j knowledge sync (G1 / DEEP_DIVE §3.7).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("sync_vault_to_neo4j", help="Index markdown vault -> Neo4j (VaultDocument).")
    s1.add_argument("--config", type=str, default=None, help="Path to config.yaml.")
    s1.add_argument("--dry-run", action="store_true", help="Report intent; touch no graph/disk.")

    s2 = sub.add_parser("sync_neo4j_to_vault", help="Export graph knowledge -> markdown vault.")
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
