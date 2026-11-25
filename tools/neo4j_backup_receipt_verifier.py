"""General-purpose offline backup evidence verifier; NEVER certifies a restore.

Claims of successful exit and SHA-256 are untrusted until independently
checking archive bytes. All archive reads require explicit operator path
and I/O cap. No database query, model request, NAS write, SSH or backup job.
"""

import argparse
import datetime as dt
import hashlib
import hmac
import json
import os
import re
import stat
from pathlib import Path

SCHEMA = "neo4j-backup-completion-evidence-v1"
MAX_RECEIPT = 32768
DEFAULT_READ_MAX = 16 * 1024 * 1024
MAX_READ_MAX = 32 * 1024**3
RUN_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
HASH_RE = re.compile(r"^[a-f0-9]{64}$")
FIELDS = {
    "schema",
    "role",
    "runId",
    "startUtc",
    "finishUtc",
    "exitCode",
    "archiveBytes",
    "archiveSha256",
}


def report(state="unverified", reason="evidence-unverified"):
    return {
        "schema": "neo4j-backup-receipt-assessment-v1",
        "state": state,
        "reason": reason,
        "claimedExitZero": False,
        "contentDigestIndependentlyVerified": False,
        "primarySourceBytesIndependentlyVerified": False,
        "mirrorSourceByteIdentityVerified": False,
        "writerSignatureVerified": False,
        "databaseConsistencyVerified": False,
        "restoreVerified": False,
        "deleteAuthorized": False,
        "backupChangeAuthorized": False,
        "archiveReads": 0,
        "remoteWrites": 0,
        "nasWrites": 0,
        "modelRequestsSent": 0,
    }


def unique_pairs(pairs):
    result = {}
    for key, val in pairs:
        if key in result:
            raise ValueError("duplicate field")
        result[key] = val
    return result


def load(path):
    try:
        path = Path(path)
        s = path.lstat()
        if not stat.S_ISREG(s.st_mode) or not 0 < s.st_size <= MAX_RECEIPT:
            return None
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or (before.st_size, before.st_ino) != (
                s.st_size,
                s.st_ino,
            ):
                return None
            data = os.read(fd, MAX_RECEIPT + 1)
            after = os.fstat(fd)
            if len(data) != s.st_size or (before.st_size, before.st_mtime_ns) != (
                after.st_size,
                after.st_mtime_ns,
            ):
                return None
        finally:
            os.close(fd)
        x = json.loads(data.decode("utf8"), object_pairs_hook=unique_pairs)
        return x if isinstance(x, dict) else None
    except (OSError, UnicodeError, ValueError, TypeError):
        return None


def utc(value):
    if not isinstance(value, str) or len(value) > 40 or not value.endswith("Z"):
        return None
    try:
        val = dt.datetime.fromisoformat(value[:-1] + "+00:00")
        return val if val.utcoffset() == dt.timedelta(0) else None
    except ValueError:
        return None


def validate(claim):
    out = report()
    if not isinstance(claim, dict) or set(claim) != FIELDS:
        return out
    if claim.get("schema") != SCHEMA or claim.get("role") not in ("primary", "mirror"):
        return out
    identifier = claim.get("runId")
    start = utc(claim.get("startUtc"))
    end = utc(claim.get("finishUtc"))
    size = claim.get("archiveBytes")
    digest = claim.get("archiveSha256")
    if (
        not isinstance(identifier, str)
        or not RUN_RE.fullmatch(identifier)
        or start is None
        or end is None
        or not start <= end
        or (end - start).total_seconds() > 12 * 3600
        or type(claim.get("exitCode")) is not int
        or type(size) is not int
        or not 1 <= size <= MAX_READ_MAX
        or not isinstance(digest, str)
        or not HASH_RE.fullmatch(digest)
    ):
        return out
    out["claimedExitZero"] = claim["exitCode"] == 0
    if not out["claimedExitZero"]:
        out.update(state="nonzero-exit-claimed-not-certified", reason="source-claims-failure")
    else:
        out.update(
            state="unsigned-success-claim-not-certified", reason="untrusted-exit-and-sha256-claims"
        )
    return out


def compare_claims(primary, mirror):
    out = report()
    if (
        validate(primary)["state"] != "unsigned-success-claim-not-certified"
        or validate(mirror)["state"] != "unsigned-success-claim-not-certified"
    ):
        return out
    if (
        primary["role"] != "primary"
        or mirror["role"] != "mirror"
        or primary["runId"] != mirror["runId"]
        or primary["archiveBytes"] != mirror["archiveBytes"]
        or not hmac.compare_digest(primary["archiveSha256"], mirror["archiveSha256"])
        or utc(primary["finishUtc"]) > utc(mirror["finishUtc"])
    ):
        return out
    out.update(
        state="matching-unsigned-claims-not-certified",
        reason="both-writers-make-agreeing-claims-without-independent-byte-proof",
        claimedExitZero=True,
    )
    return out


def check_selected_archive(claim, path, max_bytes=DEFAULT_READ_MAX, *, _expected_identity=None):
    # Private pair-check binding. A path must still name the inode that was
    # independently identified BEFORE either source/mirror archive is read.
    out = validate(claim)
    if out["state"] != "unsigned-success-claim-not-certified":
        return out
    if type(max_bytes) is not int or not 1 <= max_bytes <= MAX_READ_MAX:
        return report(reason="invalid-io-cap")
    try:
        target = Path(path)
        if target.suffix != ".backup":
            return report(reason="file-extension-not-backup")
        meta = target.lstat()
        observed = (meta.st_dev, meta.st_ino, meta.st_size, meta.st_mtime_ns, meta.st_ctime_ns)
        if _expected_identity is not None and observed != _expected_identity:
            return report(reason="archive-identity-changed-before-read")
        if (
            not stat.S_ISREG(meta.st_mode)
            or meta.st_size != claim["archiveBytes"]
            or meta.st_size > max_bytes
        ):
            return report(reason="file-is-not-regular-matches-size-or-under-io-cap")
        fd = os.open(target, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            before = os.fstat(fd)
            if (
                not stat.S_ISREG(before.st_mode)
                or (before.st_dev, before.st_ino, before.st_size)
                != (meta.st_dev, meta.st_ino, meta.st_size)
                or (
                    _expected_identity is not None
                    and (
                        before.st_dev,
                        before.st_ino,
                        before.st_size,
                        before.st_mtime_ns,
                        before.st_ctime_ns,
                    )
                    != _expected_identity
                )
            ):
                return report(reason="archive-identity-changed")
            sha = hashlib.sha256()
            read = 0
            while read < meta.st_size:
                block = os.read(fd, min(1048576, meta.st_size - read))
                if not block:
                    break
                sha.update(block)
                read += len(block)
            after = os.fstat(fd)
            if read != meta.st_size or (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ):
                return report(reason="archive-changed-during-read")
        finally:
            os.close(fd)
        out["archiveReads"] = read
        match = hmac.compare_digest(sha.hexdigest(), claim["archiveSha256"])
        out["contentDigestIndependentlyVerified"] = match
        out["state"] = (
            "local-hash-matched-writer-and-restore-unverified"
            if match
            else "local-archive-hash-mismatch"
        )
        out["reason"] = (
            "local-bytes-match-claim-without-writer-or-restore-proof"
            if match
            else "archive-digest-differs-from-claim"
        )
        return out
    except (OSError, TypeError, ValueError):
        return report(reason="archive-file-unavailable")


def compare_independently_read_files(
    primary, primary_path, mirror, mirror_path, max_bytes=DEFAULT_READ_MAX
):
    """Independently hash both explicit files; do not infer authentication or a valid graph."""
    if compare_claims(primary, mirror)["state"] != "matching-unsigned-claims-not-certified":
        return report(reason="backup-stage-claims-do-not-align")
    # Never call two reads of the SAME inode independent primary-vs-mirror proof.
    try:
        a = Path(primary_path).lstat()
        b = Path(mirror_path).lstat()
        if (
            not stat.S_ISREG(a.st_mode)
            or not stat.S_ISREG(b.st_mode)
            or (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino)
        ):
            return report(reason="source-and-mirror-not-distinct-regular-files")
    except (OSError, TypeError, ValueError):
        return report(reason="source-or-mirror-unavailable")

    def identity(meta):
        return (meta.st_dev, meta.st_ino, meta.st_size, meta.st_mtime_ns, meta.st_ctime_ns)

    original_primary = identity(a)
    original_mirror = identity(b)
    source = check_selected_archive(
        primary, primary_path, max_bytes, _expected_identity=original_primary
    )
    if source["state"] != "local-hash-matched-writer-and-restore-unverified":
        return report(reason="primary-byte-readback-not-verified")
    copied = check_selected_archive(
        mirror, mirror_path, max_bytes, _expected_identity=original_mirror
    )
    if copied["state"] != "local-hash-matched-writer-and-restore-unverified":
        return report(reason="mirror-byte-readback-not-verified")
    # Protect the pair witness from pathname substitution after either read.
    try:
        if (
            identity(Path(primary_path).lstat()) != original_primary
            or identity(Path(mirror_path).lstat()) != original_mirror
        ):
            return report(reason="source-or-mirror-path-changed-after-readback")
    except (OSError, TypeError, ValueError):
        return report(reason="source-or-mirror-path-unavailable-after-readback")
    out = report(
        "readback-byte-identical-writer-and-restore-unverified",
        "independently-read-primary-and-mirror-sha256-match-but-writer-and-graph-not-certified",
    )
    out["claimedExitZero"] = True
    out["contentDigestIndependentlyVerified"] = True
    out["primarySourceBytesIndependentlyVerified"] = True
    out["mirrorSourceByteIdentityVerified"] = True
    out["archiveReads"] = source["archiveReads"] + copied["archiveReads"]
    return out


def main():
    cli = argparse.ArgumentParser(description="Explicit bounded read-only archive/receipt verifier")
    cli.add_argument("--receipt", required=True)
    cli.add_argument(
        "--archive", help="Optional explicitly selected backup file; default no archive reads"
    )
    cli.add_argument("--max-bytes", type=int, default=DEFAULT_READ_MAX)
    a = cli.parse_args()
    claim = load(a.receipt)
    result = check_selected_archive(claim, a.archive, a.max_bytes) if a.archive else validate(claim)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["state"] == "local-hash-matched-writer-and-restore-unverified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
