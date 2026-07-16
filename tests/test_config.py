"""
Tests for the config layer:
  - auto_ingest_config.py (repo-root canonical config)
  - auto_ingest/_config.py (package shim with inline fallback)

No network, no Neo4j. Uses tmp_path / monkeypatch for env and sys.modules.
"""

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _write_config(tmp_path, body):
    p = tmp_path / "config.yaml"
    p.write_text(body)
    return p


CONFIG_BODY = """
machine_paths:
  x1-370:
    fileserver_root: /srv/fileserver
    neo4j_uri: bolt://1.2.3.4:7687
    neo4j_user: neo4j
    neo4j_password: ${NEO4J_PASSWORD}
    hostname_pattern: x1-370
    hot_ssd_root: /srv/hot
    nas_mirror_root: /srv/cold
knowledge_map:
  hot_layer_path: /km/hot
  mirror_vault_path: /km/cold
"""


@pytest.fixture
def config_in_tmp(tmp_path, monkeypatch):
    """Write a config.yaml into a temp dir and force the loader to use it."""
    p = _write_config(tmp_path, CONFIG_BODY)
    monkeypatch.setattr("auto_ingest_config._find_config_path", lambda: p)
    return p


def _clear_override_env(monkeypatch):
    for var in (
        "HOT_STORAGE_ROOT",
        "COLD_STORAGE_ROOT",
        "FILESERVER_ROOT",
        "NEO4J_URI",
        "NEO4J_USER",
        "NEO4J_PASSWORD",
        "NEXTCLOUD_ROOT",
        "NEXTCLOUD_URL",
        "NEXTCLOUD_USER",
        "NEXTCLOUD_PASS",
        "NEO4J_DB",
    ):
        monkeypatch.delenv(var, raising=False)


# --------------------------------------------------------------------------- #
# Root module: config.yaml discovery + read
# --------------------------------------------------------------------------- #
def test_storage_layout_from_config(config_in_tmp, monkeypatch):
    import auto_ingest_config as cfg

    _clear_override_env(monkeypatch)
    layout = cfg.get_storage_layout()
    assert layout["hot_root"] == "/srv/hot"
    assert layout["cold_root"] == "/srv/cold"
    assert layout["fileserver_root"] == "/srv/fileserver"


def test_neo4j_config_from_config(config_in_tmp, monkeypatch):
    import auto_ingest_config as cfg

    _clear_override_env(monkeypatch)
    monkeypatch.setenv("NEO4J_PASSWORD", "supersecret")
    nc = cfg.get_neo4j_config()
    assert nc["uri"] == "bolt://1.2.3.4:7687"
    assert nc["user"] == "neo4j"
    assert nc["password"] == "supersecret"


def test_fileserver_path_joins(config_in_tmp, monkeypatch):
    import auto_ingest_config as cfg

    _clear_override_env(monkeypatch)
    assert cfg.get_fileserver_path("dashcam") == os.path.join(
        "/srv/fileserver", "dashcam"
    )
    assert cfg.get_fileserver_path("a", "b", "c") == os.path.join(
        "/srv/fileserver", "a", "b", "c"
    )


def test_fileserver_root_and_hot_root(config_in_tmp, monkeypatch):
    import auto_ingest_config as cfg

    _clear_override_env(monkeypatch)
    assert cfg.get_fileserver_root() == "/srv/fileserver"
    assert cfg.get_hot_root() == "/srv/hot"


def test_nextcloud_root_and_webdav(config_in_tmp, monkeypatch):
    import auto_ingest_config as cfg

    _clear_override_env(monkeypatch)
    assert cfg.get_nextcloud_root() == "/media/scott/SSD_4TB/nextcloud"

    monkeypatch.setenv("NEXTCLOUD_URL", "https://nc.example/remote.php")
    monkeypatch.setenv("NEXTCLOUD_USER", "admin")
    monkeypatch.setenv("NEXTCLOUD_PASS", "pw")
    url, user, pw = cfg.get_nextcloud_webdav()
    assert url == "https://nc.example/remote.php"
    assert user == "admin"
    assert pw == "pw"

    # Missing url/user -> None tuple
    monkeypatch.delenv("NEXTCLOUD_URL", raising=False)
    monkeypatch.delenv("NEXTCLOUD_USER", raising=False)
    assert cfg.get_nextcloud_webdav() == (None, None, None)


# --------------------------------------------------------------------------- #
# Env override precedence (root module)
# --------------------------------------------------------------------------- #
def test_env_override_precedence(tmp_path, monkeypatch):
    import auto_ingest_config as cfg

    p = _write_config(tmp_path, CONFIG_BODY)
    monkeypatch.setattr("auto_ingest_config._find_config_path", lambda: p)
    monkeypatch.setenv("HOT_STORAGE_ROOT", "/env/hot")
    monkeypatch.setenv("COLD_STORAGE_ROOT", "/env/cold")
    monkeypatch.setenv("FILESERVER_ROOT", "/env/fs")
    monkeypatch.setenv("NEO4J_URI", "bolt://env:7687")
    monkeypatch.setenv("NEO4J_USER", "envuser")
    monkeypatch.setenv("NEO4J_PASSWORD", "envpw")

    layout = cfg.get_storage_layout()
    assert layout["hot_root"] == "/env/hot"
    assert layout["cold_root"] == "/env/cold"
    assert layout["fileserver_root"] == "/env/fs"
    assert cfg.get_fileserver_path("x") == os.path.join("/env/fs", "x")

    nc = cfg.get_neo4j_config()
    assert nc["uri"] == "bolt://env:7687"
    assert nc["user"] == "envuser"
    assert nc["password"] == "envpw"


def test_no_config_falls_back_to_env(tmp_path, monkeypatch):
    import auto_ingest_config as cfg

    monkeypatch.setattr("auto_ingest_config._find_config_path", lambda: None)
    monkeypatch.setenv("FILESERVER_ROOT", "/fallback/fs")
    monkeypatch.setenv("HOT_STORAGE_ROOT", "/fallback/hot")
    monkeypatch.setenv("COLD_STORAGE_ROOT", "/fallback/cold")
    monkeypatch.setenv("NEO4J_URI", "bolt://fb:7687")
    monkeypatch.setenv("NEO4J_USER", "fb")
    monkeypatch.setenv("NEO4J_PASSWORD", "fbpw")

    layout = cfg.get_storage_layout()
    assert layout["fileserver_root"] == "/fallback/fs"
    assert layout["hot_root"] == "/fallback/hot"
    assert layout["cold_root"] == "/fallback/cold"
    nc = cfg.get_neo4j_config()
    assert nc["uri"] == "bolt://fb:7687"
    assert nc["user"] == "fb"
    assert nc["password"] == "fbpw"


# --------------------------------------------------------------------------- #
# build_artifact_ref
# --------------------------------------------------------------------------- #
def test_build_artifact_ref_under_fileserver(config_in_tmp, monkeypatch):
    import auto_ingest_config as cfg

    _clear_override_env(monkeypatch)
    full = os.path.join("/srv/fileserver", "dashcam", "clip.mp4")
    ref = cfg.build_artifact_ref(
        full, storage_root="nas", retention_class="audit", tailscale_host_hint="host1"
    )
    assert ref["storage_root"] == "nas"
    assert ref["relative_path"] == os.path.join("dashcam", "clip.mp4")
    assert ref["host_path"] == full
    assert ref["container_path"] == "/nas/" + os.path.relpath(full, "/").lstrip("/")
    assert ref["tailscale_host_hint"] == "host1"
    assert ref["sha256"] is None
    assert ref["retention_class"] == "audit"


def test_build_artifact_ref_defaults(config_in_tmp, monkeypatch):
    import auto_ingest_config as cfg

    _clear_override_env(monkeypatch)
    # path not under any known root -> relative_path == host_path
    full = os.path.join("/some/other/place", "file.txt")
    ref = cfg.build_artifact_ref(full)
    assert ref["storage_root"] == "local-ssd"
    assert ref["retention_class"] == "keep"
    assert ref["relative_path"] == full
    assert ref["host_path"] == full
    assert ref["container_path"] is None
    assert ref["tailscale_host_hint"] is None
    assert ref["sha256"] is None


def test_build_artifact_ref_under_hot(config_in_tmp, monkeypatch):
    import auto_ingest_config as cfg

    _clear_override_env(monkeypatch)
    full = os.path.join("/srv/hot", "audio", "x.wav")
    ref = cfg.build_artifact_ref(full)
    assert ref["relative_path"] == os.path.join("audio", "x.wav")
    assert ref["container_path"] is not None


# --------------------------------------------------------------------------- #
# _config.py shim: preferred (root) path
# --------------------------------------------------------------------------- #
def test_shim_preferred_reexports_root(config_in_tmp, monkeypatch):
    from auto_ingest import _config

    _clear_override_env(monkeypatch)
    assert _config.get_fileserver_root() == "/srv/fileserver"
    assert _config.get_fileserver_path("d") == os.path.join("/srv/fileserver", "d")
    layout = _config.get_storage_layout()
    assert layout["hot_root"] == "/srv/hot"
    assert _config.get_hot_root() == "/srv/hot"
    nc = _config.get_neo4j_config()
    assert nc["uri"] == "bolt://1.2.3.4:7687"
    assert _config.get_nextcloud_root() == "/media/scott/SSD_4TB/nextcloud"


# --------------------------------------------------------------------------- #
# _config.py shim: fallback path (simulate import of root module failing)
# --------------------------------------------------------------------------- #
def test_shim_fallback_from_env(monkeypatch):
    import importlib

    # Force a fresh import of the shim with the root module unavailable.
    monkeypatch.setenv("FILESERVER_ROOT", "/env/fs")
    monkeypatch.setenv("HOT_STORAGE_ROOT", "/env/hot")
    monkeypatch.setenv("COLD_STORAGE_ROOT", "/env/cold")
    monkeypatch.setenv("NEO4J_URI", "bolt://shim:7687")
    monkeypatch.setenv("NEO4J_USER", "shimuser")
    monkeypatch.setenv("NEO4J_PASSWORD", "shimpw")
    monkeypatch.setenv("NEXTCLOUD_ROOT", "/env/nc")

    saved = sys.modules.get("auto_ingest_config")
    monkeypatch.setitem(sys.modules, "auto_ingest_config", None)

    # Make the `from auto_ingest_config import ...` raise so the except branch runs.
    real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    def fake_import(name, *a, **k):
        if name == "auto_ingest_config":
            raise ImportError("simulated missing root module")
        return real_import(name, *a, **k)

    monkeypatch.setattr("builtins.__import__", fake_import)

    try:
        import auto_ingest._config as shim

        importlib.reload(shim)
        assert shim.get_fileserver_root() == "/env/fs"
        assert shim.get_fileserver_path("a", "b") == os.path.join("/env/fs", "a", "b")
        layout = shim.get_storage_layout()
        assert layout["hot_root"] == "/env/hot"
        assert layout["cold_root"] == "/env/cold"
        assert layout["fileserver_root"] == "/env/fs"
        assert shim.get_hot_root() == "/env/hot"
        nc = shim.get_neo4j_config()
        assert nc["uri"] == "bolt://shim:7687"
        assert nc["user"] == "shimuser"
        assert nc["password"] == "shimpw"
        assert shim.get_nextcloud_root() == "/env/nc"
        assert shim.get_nextcloud_webdav() == (None, None, None)
    finally:
        if saved is not None:
            sys.modules["auto_ingest_config"] = saved
        else:
            sys.modules.pop("auto_ingest_config", None)
        monkeypatch.undo()
        importlib.reload(shim)


def test_shim_fallback_defaults(monkeypatch):
    import importlib

    for var in (
        "FILESERVER_ROOT",
        "HOT_STORAGE_ROOT",
        "COLD_STORAGE_ROOT",
        "NEO4J_URI",
        "NEO4J_USER",
        "NEO4J_PASSWORD",
        "NEXTCLOUD_ROOT",
    ):
        monkeypatch.delenv(var, raising=False)

    monkeypatch.setitem(sys.modules, "auto_ingest_config", None)
    real_import = __import__

    def fake_import(name, *a, **k):
        if name == "auto_ingest_config":
            raise ImportError("simulated")
        return real_import(name, *a, **k)

    monkeypatch.setattr("builtins.__import__", fake_import)

    try:
        import auto_ingest._config as shim

        importlib.reload(shim)
        assert shim.get_fileserver_root() == "/media/scott/SSD_4TB/fileserver"
        assert shim.get_storage_layout()["hot_root"] == "/media/scott/SSD_4TB/audio"
        assert shim.get_neo4j_config()["uri"] == "bolt://localhost:7687"
        assert shim.get_nextcloud_root() == "/media/scott/SSD_4TB/nextcloud"
    finally:
        sys.modules.pop("auto_ingest_config", None)
        monkeypatch.undo()
        importlib.reload(shim)


# --------------------------------------------------------------------------- #
# get_neo4j_db / get_neo4j_env
# --------------------------------------------------------------------------- #
def test_neo4j_db_from_config(tmp_path, monkeypatch):
    import auto_ingest_config as cfg

    _clear_override_env(monkeypatch)
    p = _write_config(tmp_path, CONFIG_BODY + "\nneo4j_db: memories\n")
    monkeypatch.setattr("auto_ingest_config._find_config_path", lambda: p)
    assert cfg.get_neo4j_db() == "memories"


def test_neo4j_db_env_default(monkeypatch):
    import auto_ingest_config as cfg

    _clear_override_env(monkeypatch)
    monkeypatch.setattr("auto_ingest_config._find_config_path", lambda: None)
    assert cfg.get_neo4j_db() == "neo4j"
    monkeypatch.setenv("NEO4J_DB", "graph2026")
    assert cfg.get_neo4j_db() == "graph2026"


def test_neo4j_env_tuple(config_in_tmp, monkeypatch):
    import auto_ingest_config as cfg

    _clear_override_env(monkeypatch)
    monkeypatch.setenv("NEO4J_PASSWORD", "pw")
    uri, user, password, db = cfg.get_neo4j_env()
    assert uri == "bolt://1.2.3.4:7687"
    assert user == "neo4j"
    assert password == "pw"
    assert db == "neo4j"


# --------------------------------------------------------------------------- #
# get_nextcloud_root from config.yaml (not just env)
# --------------------------------------------------------------------------- #
def test_nextcloud_root_from_config(tmp_path, monkeypatch):
    import auto_ingest_config as cfg

    _clear_override_env(monkeypatch)
    p = _write_config(tmp_path, CONFIG_BODY + "\nnextcloud_root: /srv/nextcloud\n")
    monkeypatch.setattr("auto_ingest_config._find_config_path", lambda: p)
    assert cfg.get_nextcloud_root() == "/srv/nextcloud"


def test_nextcloud_webdav_from_config(tmp_path, monkeypatch):
    import auto_ingest_config as cfg

    _clear_override_env(monkeypatch)
    body = CONFIG_BODY + (
        "\nnextcloud:\n"
        "  webdav_url: https://nc.example/remote.php/dav/\n"
        "  user: scott\n"
        "  pass_env: NC_TOKEN\n"
    )
    p = _write_config(tmp_path, body)
    monkeypatch.setattr("auto_ingest_config._find_config_path", lambda: p)
    monkeypatch.setenv("NC_TOKEN", "app-token")
    url, user, pw = cfg.get_nextcloud_webdav()
    assert url == "https://nc.example/remote.php/dav"  # trailing slash stripped
    assert user == "scott"
    assert pw == "app-token"


# --------------------------------------------------------------------------- #
# get_neo4j_driver / neo4j_session — driver construction is mocked
# --------------------------------------------------------------------------- #
def test_neo4j_driver_uses_config(config_in_tmp, monkeypatch):
    import auto_ingest_config as cfg

    _clear_override_env(monkeypatch)
    monkeypatch.setenv("NEO4J_PASSWORD", "pw")

    created = {}

    class _FakeSession:
        def __init__(self, database):
            created["session_db"] = database

    class _FakeDriver:
        def session(self, database=None):
            return _FakeSession(database)

    class _FakeGraphDatabase:
        @staticmethod
        def driver(uri, auth=None):
            created["uri"] = uri
            created["auth"] = auth
            return _FakeDriver()

    fake_neo4j = type("neo4j", (), {"GraphDatabase": _FakeGraphDatabase})
    monkeypatch.setitem(sys.modules, "neo4j", fake_neo4j)

    drv = cfg.get_neo4j_driver()
    assert drv is not None
    assert created["uri"] == "bolt://1.2.3.4:7687"
    assert created["auth"] == ("neo4j", "pw")

    sess = cfg.neo4j_session(database="memories")
    assert created["session_db"] == "memories"
    assert sess is not None


def test_neo4j_driver_unavailable(monkeypatch):
    import auto_ingest_config as cfg

    _clear_override_env(monkeypatch)
    monkeypatch.setitem(sys.modules, "neo4j", None)

    real_import = __import__

    def fake_import(name, *a, **k):
        if name == "neo4j":
            raise ImportError("no neo4j")
        return real_import(name, *a, **k)

    monkeypatch.setattr("builtins.__import__", fake_import)
    assert cfg.get_neo4j_driver() is None
    assert cfg.neo4j_session() is None
