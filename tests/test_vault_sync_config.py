"""Tests for G2 canonical-vault config (no hardcoded IPs can sneak back).

Uses system python + pyyaml only. Guards:
  * `knowledge_map.vault_peers` is a non-empty list of hostnames.
  * None of those hostnames is a raw IPv4 address.
  * `knowledge_map.canonical_vault_path` (the single writer) is set.
  * The sync scripts contain no hardcoded 100.x.x.x IPs.
"""

import re
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG = REPO_ROOT / "config.yaml"

# Matches IPv4 like 100.78.106.121 — used to forbid raw IPs in host lists
# and to scan the sync scripts.
IPV4_RE = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")


def _load_config():
    assert CONFIG.exists(), f"config.yaml missing at {CONFIG}"
    with open(CONFIG) as f:
        return yaml.safe_load(f)


def test_config_has_knowledge_map_section():
    cfg = _load_config()
    assert "knowledge_map" in cfg, "knowledge_map section missing in config.yaml"


def test_vault_peers_is_nonempty_hostname_list():
    cfg = _load_config()
    peers = cfg["knowledge_map"].get("vault_peers")
    assert isinstance(peers, list) and peers, "vault_peers must be a non-empty list"
    for p in peers:
        assert isinstance(p, str) and p.strip(), f"peer entry invalid: {p!r}"
        assert not IPV4_RE.fullmatch(p.strip()), (
            f"vault_peers must be Tailscale hostnames, not raw IPs: {p!r}"
        )


def test_no_raw_ip_in_vault_peers():
    cfg = _load_config()
    for p in cfg["knowledge_map"]["vault_peers"]:
        assert not IPV4_RE.search(p), f"raw IP found in vault_peers: {p!r}"


def test_canonical_vault_path_set():
    cfg = _load_config()
    canonical = cfg["knowledge_map"].get("canonical_vault_path")
    assert canonical and isinstance(canonical, str), "canonical_vault_path must be set"
    assert not IPV4_RE.search(canonical), "canonical_vault_path must not be a raw IP"


def test_sync_scripts_have_no_hardcoded_ips():
    """Grep the two sync scripts for 100.x.x.x style IPs — there must be none."""
    scripts = [
        REPO_ROOT / "scripts" / "knowledge_sync_all.sh",
        REPO_ROOT / "scripts" / "sync_vault_to_canonical.sh",
    ]
    bad = []
    for script in scripts:
        if not script.exists():
            continue
        text = script.read_text()
        for m in IPV4_RE.findall(text):
            bad.append(f"{script.name}: {m}")
    assert not bad, f"hardcoded IPs still present in sync scripts: {bad}"


def test_vault_sync_section_present():
    cfg = _load_config()
    vs = cfg["knowledge_map"].get("vault_sync")
    assert isinstance(vs, dict), "knowledge_map.vault_sync section missing"
    for key in ("ssh_user", "git_branch", "rsync_opts", "peer_timeout_sec"):
        assert key in vs, f"knowledge_map.vault_sync.{key} missing"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
