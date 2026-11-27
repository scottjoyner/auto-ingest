"""Date-root index: the file's own stamp must outrank an ancestor's date.

`media_index.py` is the catalog the shorts/worker stages read to choose sources.
When a file sits under a dated root (``.../2019-12-31-archive/...``) the catalog
date must describe the *file*, not the folder that happens to hold it. A catalog
that silently stamps every clip in a dated tree with the tree's date makes every
downstream "date-rooted" lookup select the wrong sources.
"""

from __future__ import annotations

import media_index


def test_date_for_prefers_filename_over_dated_ancestor(tmp_path):
    """A date in the file name is more specific than one in a parent dir."""
    dated_root = tmp_path / "2019-12-31-archive"
    dated_root.mkdir()
    f = dated_root / "clip_2024-07-19_120000.MP4"
    f.write_bytes(b"x")

    # Before the fix, DATE_RE.search on the full path matched the left-most
    # date - the ancestor's - so the file was labelled 2019-12-31.
    assert media_index._date_for(f) == "2024-07-19"


def test_date_for_falls_back_to_ancestor_when_name_undated(tmp_path):
    """With no stamp in the name, the dated folder is still the source."""
    dated_root = tmp_path / "2021" / "03" / "04"
    dated_root.mkdir(parents=True)
    f = dated_root / "IMG_0001.jpg"
    f.write_bytes(b"x")

    assert media_index._date_for(f) == "2021-03-04"


def test_scan_root_catalog_uses_filename_date(tmp_path):
    """The defect is visible in the emitted catalog, not just the helper."""
    root = tmp_path / "2020-01-02-camera"
    root.mkdir()
    (root / "a_2024-07-19.MP4").write_bytes(b"x")
    (root / "b.jpg").write_bytes(b"x")

    cat = media_index.scan_root(str(root), progress=False)
    a = next(x for x in cat["video"] if x["stem"].startswith("a"))
    b = next(x for x in cat["image"] if x["stem"].startswith("b"))
    assert a["date"] == "2024-07-19"
    assert b["date"] == "2020-01-02"
