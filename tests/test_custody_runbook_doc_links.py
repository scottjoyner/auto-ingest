#!/usr/bin/env python3
"""Offline checks for the R9 external-inventory custody runbook.

Reads only the committed runbook in this worktree: every cited repo-relative file
path must exist, and the document must contain no credential-shaped value. No
network, NAS mount, credential, live state or model load is used.
"""
import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNBOOK = "docs/external-inventory-custody-runbook-r9.md"

# Cited paths are backticked repo-relative source/config/doc files. Absolute
# paths, module paths, env vars, column names, SHA pins and media filenames are
# deliberately not treated as repo paths.
PATH_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*(?:/[A-Za-z0-9_.-]+)*")
DOC_EXTS = frozenset({".py", ".md", ".toml", ".yml", ".yaml", ".json", ".sh"})

CREDENTIAL_PATTERNS = (
    ("assigned secret", re.compile(
        r"(?i)\b(?:password|passwd|pwd|secret|api[_-]?key|access[_-]?key|"
        r"private[_-]?key|auth[_-]?token|bearer)\b\s*[:=]\s*[\"']?[A-Za-z0-9+/=_-]{6,}")),
    ("private key block", re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----")),
    ("github token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("aws access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("openai-style key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("url credentials", re.compile(r"[a-z][a-z0-9+.-]*://[^/\s:]+:[^/\s@]+@")),
    ("sshpass assignment", re.compile(r"(?i)\bSSHPASS\s*=\s*\S+")),
)


def cited_paths(text):
    """Sorted distinct repo-relative file paths cited in inline code."""
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    result = set()
    for token in re.findall(r"`([^`\n]+)`", text):
        candidate = token.strip()
        if candidate.startswith("/") or ".." in candidate.split("/"):
            continue
        if PATH_RE.fullmatch(candidate) and Path(candidate).suffix.lower() in DOC_EXTS:
            result.add(candidate)
    return sorted(result)


def find_credentials(text):
    """Human-readable list of credential-shaped values found in the text."""
    return [
        f"{name}: {match.group(0)}"
        for name, pattern in CREDENTIAL_PATTERNS
        for match in pattern.finditer(text)
    ]


def read_runbook():
    return (ROOT / RUNBOOK).read_text(encoding="utf-8")


class CustodyRunbookDocLinksTest(unittest.TestCase):
    def test_runbook_is_committed(self):
        self.assertTrue((ROOT / RUNBOOK).is_file(), f"runbook not present: {RUNBOOK}")
        proc = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", RUNBOOK],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertEqual(
            proc.returncode, 0, f"not committed: {RUNBOOK}: {proc.stderr.strip()}"
        )

    def test_cited_repo_paths_exist(self):
        paths = cited_paths(read_runbook())
        self.assertGreaterEqual(len(paths), 5, f"too few cited paths: {paths}")
        missing = [path for path in paths if not (ROOT / path).is_file()]
        self.assertEqual(missing, [], "cited paths missing from worktree")

    def test_extractor_ignores_non_repo_paths(self):
        sample = (
            "`/home/scott/nas-reconciliation-audit-20261016/sd.csv` "
            "`auto_ingest.custody.external_inventory.observe_metadata_csv` "
            "`e0026bdc3975f1f0f7fd5db529edb65105918c900dc89655e9c2a1ffc822d484` "
            "`Fig1.png` `EXACT_NAME_ABSENT`"
        )
        self.assertEqual(cited_paths(sample), [])

    def test_no_credential_shaped_values(self):
        self.assertEqual(find_credentials(read_runbook()), [])

    def test_sha256_pin_is_not_a_credential(self):
        # Control: a bare 64-hex pin (as in the runbook example) is not flagged.
        sample = 'expected_sha256="e0026bdc3975f1f0f7fd5db529edb65105918c900dc89655e9c2a1ffc822d484"'
        self.assertEqual(find_credentials(sample), [])

    def test_credential_detector_is_not_vacuous(self):
        sample = (
            'password = "hunter2supersecret"\n'
            "ghp_abcdefghijklmnopqrstuvwxyz0123456789\n"
            "https://user:p4ssw0rd@example.invalid/repo.git\n"
        )
        found = find_credentials(sample)
        self.assertGreaterEqual(len(found), 3, f"detector missed values: {found}")
        self.assertTrue(any("github token" in item for item in found))
        self.assertTrue(any("url credentials" in item for item in found))


if __name__ == "__main__":
    unittest.main()
