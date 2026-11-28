"""Round-10: the hash evidence ``errors`` block must hold *all* findings.

``to_evidence`` has two independent facts to record under ``errors``:

* ``declared_exclusions`` / ``excluded_objects`` - what the policy kept out of
  the campaign, auditable from the bundle alone;
* ``unresolved`` / ``summaries`` - name problems that deny release.

They are **not** mutually exclusive. ``custody hash --apply`` passes both on a
card that has a policy exclusion *and* a case-colliding pair, and the merge rule
for evidence is block-by-block: losing either half makes the campaign
self-contradictory (the operator sees unresolved errors but not why some objects
are absent, or sees exclusions but not that release is denied).

Hermetic: ``tmp_path`` only. No model, live DB, network, media or mount access.
"""
from __future__ import annotations

from pathlib import Path

from auto_ingest.custody.hashing import hash_source, to_evidence


def _card(root: Path, names: tuple[str, ...]) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    keys: dict[str, Path] = {}
    for name in names:
        path = root / name
        path.write_bytes(name.encode("utf-8"))
        keys[name] = path
    return keys


def test_exclusions_and_name_problems_are_both_recorded(tmp_path):
    """Regression: the second ``errors`` write must not erase the first."""
    keys = _card(tmp_path / "card", ("clip.mp4", "CLIP.MP4"))
    result = hash_source(tmp_path / "bundle", keys)
    assert result.name_problem_count > 0  # the collision is really there

    fragment = to_evidence(result,
                           excluded=(".Trashes/501/loc.json", "skip/notes.txt"))
    errors = fragment["errors"]

    # The exclusion audit survives alongside the unresolved name problems.
    assert errors["declared_exclusions"] == [".Trashes/501/loc.json", "skip/notes.txt"]
    assert errors["excluded_objects"] == 2
    # ...and the release-denying counter is still present.
    assert errors["unresolved"] == result.name_problem_count
    assert errors["summaries"] == list(result.name_problems)


def test_name_problems_alone_are_still_recorded(tmp_path):
    """Control: the name-problem-only branch is unchanged."""
    keys = _card(tmp_path / "card", ("clip.mp4", "CLIP.MP4"))
    result = hash_source(tmp_path / "bundle", keys)
    fragment = to_evidence(result)
    assert fragment["errors"]["unresolved"] == result.name_problem_count
    assert "declared_exclusions" not in fragment["errors"]


def test_exclusions_alone_are_still_recorded(tmp_path):
    """Control: the exclusion-only branch on a clean card is unchanged."""
    keys = _card(tmp_path / "card", ("clip_0.mp4", "clip_1.mp4"))
    result = hash_source(tmp_path / "bundle", keys)
    assert result.name_problem_count == 0

    fragment = to_evidence(result, excluded=(".Trashes/501/loc.json",))
    assert fragment["errors"]["declared_exclusions"] == [".Trashes/501/loc.json"]
    assert fragment["errors"]["excluded_objects"] == 1
    assert "unresolved" not in fragment["errors"]


def test_no_findings_writes_no_errors_block(tmp_path):
    """Control: a clean pass with no exclusions stays silent."""
    keys = _card(tmp_path / "card", ("clip_0.mp4",))
    result = hash_source(tmp_path / "bundle", keys)
    assert result.name_problem_count == 0
    assert "errors" not in to_evidence(result)
