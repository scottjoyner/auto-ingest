"""Regression tests for machine-agnostic shared configuration."""
from __future__ import annotations

import auto_ingest_config as config


def test_shared_neo4j_fallback_is_localhost(monkeypatch):
    monkeypatch.delenv("NEO4J_URI", raising=False)
    monkeypatch.delenv("NEO4J_USER", raising=False)
    monkeypatch.setattr(config, "_find_config_path", lambda: None)

    resolved = config.get_neo4j_config()
    assert resolved["uri"] == "bolt://localhost:7687"
    assert resolved["user"] == "neo4j"
    assert not resolved["uri"].startswith("bolt://100.")


def test_explicit_neo4j_environment_override_wins(monkeypatch):
    monkeypatch.setenv("NEO4J_URI", "bolt://example.internal:7687")
    monkeypatch.setenv("NEO4J_USER", "ci-user")
    monkeypatch.setattr(config, "_find_config_path", lambda: None)

    resolved = config.get_neo4j_config()
    assert resolved["uri"] == "bolt://example.internal:7687"
    assert resolved["user"] == "ci-user"


def test_unmatched_hostname_uses_generic_profile_not_arbitrary_machine(monkeypatch, tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """
machine_paths:
  x1-370:
    hostname_pattern: x1-370
    neo4j_uri: bolt://100.64.43.123:7687
    neo4j_user: neo4j
  any:
    hostname_pattern: ''
    neo4j_uri: bolt://localhost:7687
    neo4j_user: neo4j
""".lstrip()
    )

    monkeypatch.delenv("NEO4J_URI", raising=False)
    monkeypatch.setattr(config, "_find_config_path", lambda: cfg)
    monkeypatch.setattr(config.socket, "gethostname", lambda: "unrecognized-ci-host")

    resolved = config.get_neo4j_config()
    assert resolved["uri"] == "bolt://localhost:7687"
