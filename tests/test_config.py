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


def _config_body_for(fileserver, hot, cold):
    """Render a config.yaml body whose paths point at real (existing) dirs.

    Needed because the mount-aware resolvers (get_fileserver_root, ...) return
    the first path that actually exists on disk, so tests must use live paths.
    """
    return f"""
machine_paths:
  x1-370:
    fileserver_root: {fileserver}
    neo4j_uri: bolt://1.2.3.4:7687
    neo4j_user: neo4j
    neo4j_password: ${{NEO4J_PASSWORD}}
    hostname_pattern: x1-370
    hot_ssd_root: {hot}
    nas_mirror_root: {cold}
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


@pytest.fixture
def live_config_in_tmp(tmp_path, monkeypatch):
    """Like config_in_tmp but the storage roots are REAL existing dirs.

    Use this for the mount-aware resolvers (get_fileserver_root / get_hot_root /
    build_artifact_ref) which only return a configured path when it exists.
    Returns the (fileserver, hot, cold) Paths so tests can assert on them.
    """
    fileserver = tmp_path / "fileserver"
    hot = tmp_path / "hot"
    cold = tmp_path / "cold"
    for d in (fileserver, hot, cold):
        d.mkdir()
        # non-empty so _first_existing_path treats a bare dir as usable
        (d / ".keep").write_text("")
    p = _write_config(tmp_path, _config_body_for(fileserver, hot, cold))
    monkeypatch.setattr("auto_ingest_config._find_config_path", lambda: p)
    return fileserver, hot, cold


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


def test_fileserver_path_joins(live_config_in_tmp, monkeypatch):
    import auto_ingest_config as cfg

    fileserver, _hot, _cold = live_config_in_tmp
    _clear_override_env(monkeypatch)
    assert cfg.get_fileserver_path("dashcam") == os.path.join(
        str(fileserver), "dashcam"
    )
    assert cfg.get_fileserver_path("a", "b", "c") == os.path.join(
        str(fileserver), "a", "b", "c"
    )


def test_fileserver_root_and_hot_root(live_config_in_tmp, monkeypatch):
    import auto_ingest_config as cfg

    fileserver, hot, _cold = live_config_in_tmp
    _clear_override_env(monkeypatch)
    assert cfg.get_fileserver_root() == str(fileserver)
    assert cfg.get_hot_root() == str(hot)


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

    # get_fileserver_path is mount-aware -> FILESERVER_ROOT must be a live dir.
    env_fs = tmp_path / "env_fs"
    env_fs.mkdir()
    (env_fs / ".keep").write_text("")

    p = _write_config(tmp_path, CONFIG_BODY)
    monkeypatch.setattr("auto_ingest_config._find_config_path", lambda: p)
    monkeypatch.setenv("HOT_STORAGE_ROOT", "/env/hot")
    monkeypatch.setenv("COLD_STORAGE_ROOT", "/env/cold")
    monkeypatch.setenv("FILESERVER_ROOT", str(env_fs))
    monkeypatch.setenv("NEO4J_URI", "bolt://env:7687")
    monkeypatch.setenv("NEO4J_USER", "envuser")
    monkeypatch.setenv("NEO4J_PASSWORD", "envpw")

    layout = cfg.get_storage_layout()
    assert layout["hot_root"] == "/env/hot"
    assert layout["cold_root"] == "/env/cold"
    assert layout["fileserver_root"] == str(env_fs)
    assert cfg.get_fileserver_path("x") == os.path.join(str(env_fs), "x")

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
def test_build_artifact_ref_under_fileserver(live_config_in_tmp, monkeypatch):
    import auto_ingest_config as cfg

    fileserver, _hot, _cold = live_config_in_tmp
    _clear_override_env(monkeypatch)
    full = os.path.join(str(fileserver), "dashcam", "clip.mp4")
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


def test_build_artifact_ref_under_hot(live_config_in_tmp, monkeypatch):
    import auto_ingest_config as cfg

    _fileserver, hot, _cold = live_config_in_tmp
    _clear_override_env(monkeypatch)
    full = os.path.join(str(hot), "audio", "x.wav")
    ref = cfg.build_artifact_ref(full)
    assert ref["relative_path"] == os.path.join("audio", "x.wav")
    assert ref["container_path"] is not None


# --------------------------------------------------------------------------- #
# _config.py shim: preferred (root) path
# --------------------------------------------------------------------------- #
def test_shim_preferred_reexports_root(live_config_in_tmp, monkeypatch):
    from auto_ingest import _config

    fileserver, hot, _cold = live_config_in_tmp
    _clear_override_env(monkeypatch)
    assert _config.get_fileserver_root() == str(fileserver)
    assert _config.get_fileserver_path("d") == os.path.join(str(fileserver), "d")
    layout = _config.get_storage_layout()
    assert layout["hot_root"] == str(hot)
    assert _config.get_hot_root() == str(hot)
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


# --------------------------------------------------------------------------- #
# Mount-aware resolvers (deathstar merge): _first_existing_path,
# _require_mounted, get_cold_root / get_audio_root / get_dashcam_root /
# get_bodycam_root. These probe the filesystem so we use real tmp dirs.
# --------------------------------------------------------------------------- #
def test_first_existing_path_skips_missing_and_empty(tmp_path):
    import auto_ingest_config as cfg

    missing = tmp_path / "missing"          # does not exist -> skipped
    empty = tmp_path / "empty"
    empty.mkdir()                           # exists but empty non-mount -> skipped
    live = tmp_path / "live"
    live.mkdir()
    (live / ".keep").write_text("")         # exists + non-empty -> chosen

    got = cfg._first_existing_path(str(missing), str(empty), str(live))
    assert got == str(live)


def test_first_existing_path_falls_back_to_first_candidate(tmp_path):
    import auto_ingest_config as cfg

    a = tmp_path / "a"          # neither exists
    b = tmp_path / "b"
    # nothing usable -> returns first candidate verbatim (caller's default)
    assert cfg._first_existing_path(str(a), str(b)) == str(a)
    # no candidates at all -> empty string
    assert cfg._first_existing_path() == ""


def test_subpath_or_root():
    import auto_ingest_config as cfg

    assert cfg._subpath_or_root("/data", "audio") == os.path.join("/data", "audio")
    # already ends with the subdir -> unchanged
    assert cfg._subpath_or_root("/data/audio", "audio") == "/data/audio"


def test_require_mounted_ok_and_fail(tmp_path):
    import auto_ingest_config as cfg

    live = tmp_path / "mnt"
    live.mkdir()
    (live / ".keep").write_text("")
    assert cfg._require_mounted(str(live)) == str(live)

    # missing path -> RuntimeError
    with pytest.raises(RuntimeError):
        cfg._require_mounted(str(tmp_path / "nope"))

    # empty, non-mount dir -> RuntimeError (fail closed on unmounted share)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(RuntimeError):
        cfg._require_mounted(str(empty))


def test_get_audio_root_env_and_layout(live_config_in_tmp, monkeypatch, tmp_path):
    import auto_ingest_config as cfg

    _fileserver, hot, _cold = live_config_in_tmp
    _clear_override_env(monkeypatch)

    # AUDIO_ROOT env wins when it points at a live dir
    env_audio = tmp_path / "env_audio"
    env_audio.mkdir()
    (env_audio / ".keep").write_text("")
    monkeypatch.setenv("AUDIO_ROOT", str(env_audio))
    assert cfg.get_audio_root() == str(env_audio)

    # Without env, falls to hot_root/audio when that exists
    monkeypatch.delenv("AUDIO_ROOT", raising=False)
    (hot / "audio").mkdir()
    (hot / "audio" / ".keep").write_text("")
    assert cfg.get_audio_root() == os.path.join(str(hot), "audio")


def test_get_dashcam_and_bodycam_root_env(live_config_in_tmp, monkeypatch, tmp_path):
    import auto_ingest_config as cfg

    _clear_override_env(monkeypatch)
    d = tmp_path / "dash"
    b = tmp_path / "body"
    monkeypatch.setenv("DASHCAM_ROOT", str(d))
    monkeypatch.setenv("BODYCAM_ROOT", str(b))
    assert cfg.get_dashcam_root() == str(d)
    assert cfg.get_bodycam_root() == str(b)


def test_get_dashcam_root_local_cache(live_config_in_tmp, monkeypatch):
    import auto_ingest_config as cfg

    fileserver, _hot, _cold = live_config_in_tmp
    _clear_override_env(monkeypatch)
    # No env -> uses fileserver_root/dashcam local cache since fileserver exists
    assert cfg.get_dashcam_root() == os.path.join(str(fileserver), "dashcam")


def test_get_cold_root_requires_mount(live_config_in_tmp, monkeypatch):
    import auto_ingest_config as cfg

    _fileserver, _hot, cold = live_config_in_tmp
    _clear_override_env(monkeypatch)
    # cold dir exists + non-empty -> returned
    assert cfg.get_cold_root() == str(cold)
