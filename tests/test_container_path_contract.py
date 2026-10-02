"""Container-path contract: encodes the DECISION still owed by an operator.

The repository disagrees with itself about container-side storage paths:

* `docker-compose.yml` injects `SCAN_ROOTS` / `DASHCAM_ROOT` / `AUDIO_ROOT` /
  `BODYCAM_ROOT` / `TRANSCRIPT_ROOT` into containers, where **only the bind mounts
  exist** (`${NAS_ROOT}:/nas`, plus the deprecated `/nas1`, `/nas2` aliases).
* `deploy/path_profiles.env.example` fills those variables with **host** paths
  (`/media/scott/NAS5/...`), which are not visible inside a container.
* `docs/system_design.md` documents a third shape: `/nas/fileserver/<family>`,
  which assumes `NAS_ROOT` points at a *fileserver parent*, whereas the env
  template sets `NAS_ROOT=/media/scott/NAS5` (the NAS root itself).

Choosing between `/nas/<family>` and `/nas/fileserver/<family>` depends on the real
layout of the NAS5 mount, which cannot be inspected from this repository (and
media/storage inspection is out of scope for housekeeping). The tests below are
therefore marked ``xfail(strict=False)``: they document the desired invariant and
will start failing loudly only as `XPASS` if somebody lands the decision, at which
point the markers can be removed and the values updated.

To resolve: pick the layout that matches the live mount, set the values in
`deploy/path_profiles.env.example` (and the live `.env`), then drop the xfail
markers in this file. See
`docs/housekeeping/storage-path-and-dead-code-audit-2026-10-02.md` §1.
"""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# Variables that docker-compose passes into containers (i.e. that must be
# container-visible paths, never raw host paths).
CONTAINER_VISIBLE_VARS = [
    "SCAN_ROOTS",
    "DASHCAM_ROOT",
    "AUDIO_ROOT",
    "BODYCAM_ROOT",
    "TRANSCRIPT_ROOT",
    "LEGACY_DROP_ROOT",
    "YOLO_ROOT",
    "META_ROOT",
    "DROP_ROOT",
]


def _env_values() -> dict:
    values = {}
    for line in (REPO / "deploy/path_profiles.env.example").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        values[key.strip()] = val.strip()
    return values


@pytest.mark.xfail(strict=False, reason="operator decision pending: host vs container paths")
def test_container_visible_vars_are_nas_paths():
    values = _env_values()
    bad = {
        k: v
        for k, v in values.items()
        if k in CONTAINER_VISIBLE_VARS and v and "/media/scott/" in v
    }
    assert not bad, f"container-visible vars still hold host paths: {bad}"


@pytest.mark.xfail(strict=False, reason="operator decision pending: /nas/<family> vs /nas/fileserver/<family>")
def test_documented_container_layout_matches_nas_root():
    values = _env_values()
    nas_root = values["NAS_ROOT"]
    design = (REPO / "docs/system_design.md").read_text()
    # docs document /nas/fileserver/<family>; that is only coherent if NAS_ROOT
    # is a fileserver parent. NAS_ROOT=/media/scott/NAS5 is the NAS root itself.
    documents_fileserver_subdir = "/nas/fileserver/" in design
    nas_root_has_fileserver = nas_root.rstrip("/").endswith("fileserver")
    assert documents_fileserver_subdir == nas_root_has_fileserver, (
        f"docs document /nas/fileserver/<family> but NAS_ROOT={nas_root} "
        "(NAS root, not a fileserver parent)"
    )


def test_drop_root_already_conforms():
    """DROP_ROOT is the one container-visible var that is already /nas-prefixed."""
    values = _env_values()
    assert values["DROP_ROOT"].startswith("/nas")


def test_compose_mounts_cover_nas():
    compose = (REPO / "docker-compose.yml").read_text()
    assert "${NAS_ROOT}:/nas" in compose
    # any container-visible var must have a mount backing it
    assert re.search(r"\$\{NAS\d?_ROOT[^\n]*\}:/nas\d?", compose)