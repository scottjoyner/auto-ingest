#!/usr/bin/env python3
"""Read-only promotion preflight for the 2026-11-28 auto-ingest review.

Verifies the exact head/commit set that was reviewed for promotion and prints
the disposition of every commit in ``origin/main..HEAD``. No network, no
writes, no branch changes -- safe under the fleet launch HOLD. An operator may
run this against any checkout; a non-matching head is reported as a failure,
not silently accepted.

Usage::

    python3 scripts/check_promotion_20261128.py [REPO]

REPO defaults to the repository containing this script.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

EXPECTED_HEAD = "f24b26465f56bab0eadce977ed67b91a410125b7"
EXPECTED_BRANCH = "integration/round7-auto-ingest-20261126"
BASE = "origin/main"

# (full hash, disposition, reason). Order is oldest -> newest, as in
# `git log --oneline origin/main..HEAD`.
COMMITS = [
    ("2b778b63db2b4c498c09d4a3549c5e81839fc541", "GENUINE feature",
     "custody external-inventory import (pinned NAS metadata, no release authority) + docs + tests"),
    ("13383a635b743498a0094874d467ee653e02ef57", "GENUINE fix",
     "custody fail-closed on casefold-colliding source/destination paths + tests"),
    ("89d20773713b3fbd324e388aa7d4d307cc833d28", "GENUINE docs",
     "records the NAS casefold collision fail-closed rule (accurate)"),
    ("9f74cf5b92854abb543b1ab3fe220e6a7d685f34", "GENUINE docs",
     "README branch-lineage note; PR #20 = 0d7f21d9 is on main (accurate, not stale)"),
    ("b81b9e112c534f0c8a31339df3318dd44844a6e0", "GENUINE fix",
     "CsvAdapter.extract strips UTF-8 BOM; byte digest preserved; 2 regression tests"),
    ("b237e9a927e8968c3d63fc9b91e773aeb54616e8", "GENUINE fix",
     "custody inventory CSV decoded utf-8-sig; SHA-256 pin still covers BOM; regression test"),
    ("871e05fc35dc8cecdf6c8340421a27c1ba241698", "GENUINE fix",
     "ingest intermediate CSV readers decoded utf-8-sig; 2 regression tests"),
    ("7c244d8618b860a542d55c8b942865ffd35a6a38", "GENUINE merge",
     "parents 9f74cf5b + b81b9e11; forwards the CsvAdapter BOM fix into integration"),
    (EXPECTED_HEAD, "GENUINE merge",
     "parents 7c244d86 + 871e05fc; forwards custody/ingest BOM fixes; reviewed head"),
]

MERGE_PARENTS = {
    "7c244d8618b860a542d55c8b942865ffd35a6a38": [
        "9f74cf5b92854abb543b1ab3fe220e6a7d685f34",
        "b81b9e112c534f0c8a31339df3318dd44844a6e0",
    ],
    EXPECTED_HEAD: [
        "7c244d8618b860a542d55c8b942865ffd35a6a38",
        "871e05fc35dc8cecdf6c8340421a27c1ba241698",
    ],
}

REGRESSION_FILES = [
    "tests/test_adapters.py",
    "tests/test_custody_external_inventory.py",
    "tests/test_ingest_intermediate.py",
]


def git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True, capture_output=True, check=False,
    )


def main() -> int:
    repo = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
    failures: list[str] = []

    def check(label: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}{(': ' + detail) if detail else ''}")
        if not ok:
            failures.append(label)

    head = git(repo, "rev-parse", "HEAD").stdout.strip()
    branch = git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    base_exists = git(repo, "rev-parse", "--verify", BASE).returncode == 0

    print(f"repo: {repo}")
    check("exact reviewed HEAD", head == EXPECTED_HEAD, head)
    # Branch name is informational: a detached worktree at the exact head is
    # equally valid for promotion review; the head hash is the lineage anchor.
    print(f"  [INFO ] branch: {branch} (reviewed on {EXPECTED_BRANCH})")
    check(f"base ref {BASE} resolves", base_exists)

    if base_exists:
        ancestor = git(repo, "merge-base", "--is-ancestor", BASE, "HEAD").returncode == 0
        check("base is ancestor of HEAD", ancestor)
        count = git(repo, "rev-list", "--count", f"{BASE}..HEAD").stdout.strip()
        check("exactly nine commits to promote", count == "9", count)
        actual = git(repo, "rev-list", "--reverse", f"{BASE}..HEAD").stdout.split()
        expected = [h for h, _, _ in COMMITS]
        check("commit set/order matches reviewed lineage", actual == expected)

    for commit, parents in MERGE_PARENTS.items():
        line = git(repo, "rev-list", "--parents", "-n", "1", commit).stdout.split()
        check(f"merge parents {commit[:8]}", line[1:] == parents, " ".join(line[1:]))

    for relpath in REGRESSION_FILES:
        present = git(repo, "cat-file", "-e", f"HEAD:{relpath}").returncode == 0
        check(f"regression test {relpath}", present)

    # Prove the BOM fixes are not stale duplicates already on the base.
    dup = git(repo, "grep", "-l", "utf-8-sig", BASE, "--",
              "content_os/adapters.py", "auto_ingest/custody/external_inventory.py",
              "auto_ingest/ingest_import.py", "auto_ingest/ingest_write.py").stdout.strip()
    check("BOM fixes absent on base (not stale duplicates)", not dup, dup)

    print("\nDisposition of origin/main..HEAD (oldest -> newest):")
    for short, disposition, reason in COMMITS:
        print(f"  {short[:8]}  {disposition:<15} {reason}")

    ok = not failures
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'} ({len(failures)} failing check(s))")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
