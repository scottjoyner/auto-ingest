"""Owner-only, opt-in *experimental* producer adapter for signed backup claims.

NOT deployed, not part of any cron schedule, and never a restore certificate.
This callable executes one explicitly approved local executable with shell=False
ONLY after owner_execute=True; observes the direct subprocess return code,
then hashes an explicitly selected NEW archive with a bounded byte ceiling.
A signed receipt records witnessed *claims*, not successful graph recovery.

Do not call from a production scheduler before owner security/key/access review.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import os
import stat
import subprocess
from pathlib import Path

from neo4j_backup_receipt_verifier import MAX_READ_MAX, RUN_RE, SCHEMA, utc
from neo4j_signed_completion_receipt import (
    KEY_RE,
    _open_private_dir_componentwise,
    create_signed_claim,
    publish_signed_envelope,
    verify_signed_claim,
)

_CHUNK = 1024 * 1024
_MAX_SECONDS = 12 * 3600


def report(state="unverified", reason="producer-evidence-unavailable"):
    return {
        "schema": "neo4j-owner-producer-adapter-v1",
        "state": state,
        "reason": reason,
        "ownerExecutionOptIn": False,
        "jobProcessInvoked": False,
        "directJobExitObservedZero": False,
        "archiveReadbackSha256Completed": False,
        "signedReceiptPublished": False,
        "archiveBytesRead": 0,
        "archiveWritesByAdapter": 0,
        "backupCertified": False,
        "sourceMirrorVerified": False,
        "databaseRestoreVerified": False,
        "deleteAuthorized": False,
        "automatedDeploymentAuthorized": False,
    }


def _identity(s):
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def _checked_dir(path, *, owner_private=False):
    fd = _open_private_dir_componentwise(path)
    try:
        info = os.fstat(fd)
        if (
            not stat.S_ISDIR(info.st_mode)
            or (info.st_mode & 0o002)
            or (owner_private and (info.st_uid != os.geteuid() or info.st_mode & 0o077))
        ):
            raise ValueError("unapproved directory access or ownership")
        return fd
    except BaseException:
        os.close(fd)
        raise


def _read_new_archive(path, dir_fd, *, cap):
    """Stream a regular NEW archive through the pinned parent fd.

    This does not establish that the observed job, rather than a concurrent
    actor, created these bytes. Producer isolation is a separate approval.
    """
    meta = os.stat(path.name, dir_fd=dir_fd, follow_symlinks=False)
    if not stat.S_ISREG(meta.st_mode) or meta.st_nlink != 1 or not 1 <= meta.st_size <= cap:
        raise ValueError("archive file or byte budget unverified")
    fd = os.open(
        path.name,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
        dir_fd=dir_fd,
    )
    try:
        before = os.fstat(fd)
        if _identity(before) != _identity(meta) or not stat.S_ISREG(before.st_mode):
            raise ValueError("archive identity changed before read")
        digest = hashlib.sha256()
        count = 0
        while count < meta.st_size:
            chunk = os.read(fd, min(_CHUNK, meta.st_size - count))
            if not chunk:
                raise ValueError("unexpected short archive")
            digest.update(chunk)
            count += len(chunk)
        after = os.fstat(fd)
        observed = os.stat(path.name, dir_fd=dir_fd, follow_symlinks=False)
        if (
            _identity(after) != _identity(meta)
            or _identity(observed) != _identity(meta)
            or count != meta.st_size
        ):
            raise ValueError("archive changed during independent read")
        return count, digest.hexdigest()
    finally:
        os.close(fd)


def run_owner_stage(
    *,
    owner_execute=False,
    role,
    run_id,
    executable,
    executable_root,
    archive,
    archive_root,
    receipt_directory,
    key_id,
    owner_private_key,
    max_archive_bytes,
    timeout_seconds,
    approved_primary_envelope=None,
    approved_primary_keys=None,
    runner=subprocess.run,
    clock=lambda: dt.datetime.now(dt.timezone.utc),
):
    """Observe one *newly produced* archive after a real local job exit.

    No CLI entry point. Only the approved owner should integrate this into an
    independently reviewed scheduler, after supplying role-specific keys.
    The mirror role requires a valid independent signed primary claim.
    """
    out = report()
    if owner_execute is not True:
        return out
    out["ownerExecutionOptIn"] = True

    def held(reason, state="unverified"):
        out["reason"] = reason
        out["state"] = state
        return out

    job_fd = archive_fd = receipt_fd = None
    try:
        if (
            role not in ("primary", "mirror")
            or not isinstance(run_id, str)
            or not RUN_RE.fullmatch(run_id)
            or type(max_archive_bytes) is not int
            or not 1 <= max_archive_bytes <= MAX_READ_MAX
            or type(timeout_seconds) is not int
            or not 1 <= timeout_seconds <= _MAX_SECONDS
            or not callable(getattr(owner_private_key, "sign", None))
            or not isinstance(key_id, str)
            or not KEY_RE.fullmatch(key_id)
        ):
            return held("invalid-owner-job-contract")
        script = Path(executable)
        jobroot = Path(executable_root)
        target = Path(archive)
        root = Path(archive_root)
        receipt = Path(receipt_directory)
        if (
            not all(
                p.is_absolute() and ".." not in p.parts
                for p in (script, jobroot, target, root, receipt)
            )
            or script.parent != jobroot
            or target.parent != root
            or target.suffix != ".backup"
            or not script.name
            or not target.name
        ):
            return held("owner-path-contract-invalid")
        job_fd = _checked_dir(jobroot, owner_private=True)
        archive_fd = _checked_dir(root)
        receipt_fd = _checked_dir(receipt, owner_private=True)
        script_meta = os.stat(script.name, dir_fd=job_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(script_meta.st_mode)
            or script_meta.st_nlink != 1
            or script_meta.st_uid != os.geteuid()
            or script_meta.st_mode & 0o022
            or not script_meta.st_mode & 0o100
        ):
            return held("approved-executable-identity-unverified")
        try:
            os.stat(target.name, dir_fd=archive_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            return held("refused-to-reuse-existing-archive")
        # If mirroring, authenticate the independently approved primary
        # claim before executing a mirror job or accepting its output.
        primary = None
        if role == "mirror":
            check = verify_signed_claim(approved_primary_envelope, approved_primary_keys)
            if (
                not check["approvedProducerKeyVerified"]
                or approved_primary_envelope["claim"]["role"] != "primary"
                or approved_primary_envelope["claim"]["runId"] != run_id
            ):
                return held("primary-claim-not-independently-approved")
            primary = approved_primary_envelope["claim"]
        start = clock()
        if not isinstance(start, dt.datetime) or start.tzinfo is None:
            return held("untrusted-owner-clock")
        if primary is not None and utc(primary["finishUtc"]) > start:
            return held("mirror-before-primary-completion")
        # This is the sole execution call: no shell, no command arguments,
        # no user input and no inherited stdin; do not collect job logs.
        out["jobProcessInvoked"] = True
        completed = runner(
            [str(script)],
            shell=False,
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_seconds,
        )
        finish = clock()
        if (
            not isinstance(finish, dt.datetime)
            or finish.tzinfo is None
            or not start <= finish
            or (finish - start).total_seconds() > _MAX_SECONDS
            or type(completed.returncode) is not int
            or completed.returncode != 0
        ):
            return held(
                "no-success-receipt-for-failed-job", state="held-job-exit-not-zero-or-unverified"
            )
        out["directJobExitObservedZero"] = True
        # Confirm the executable and both directory identities were not
        # replaced by pathname substitution during the job. Pinned FDs
        # preserve what this process saw even if the path changes.
        current_executable = os.stat(script.name, dir_fd=job_fd, follow_symlinks=False)
        if _identity(current_executable) != _identity(script_meta):
            return held("job-executable-identity-changed")
        for pathname, fd in ((jobroot, job_fd), (root, archive_fd), (receipt, receipt_fd)):
            if _identity(os.stat(pathname, follow_symlinks=False)) != _identity(os.fstat(fd)):
                return held("owner-directory-identity-changed")
        # A failed/interrupted stream may have read an unknown number of
        # bytes. Never report zero I/O after starting an uncertain read.
        out["archiveBytesRead"] = None
        count, digest = _read_new_archive(target, archive_fd, cap=max_archive_bytes)
        out["archiveBytesRead"] = count
        out["archiveReadbackSha256Completed"] = True
        if primary is not None and (
            count != primary["archiveBytes"] or digest != primary["archiveSha256"]
        ):
            return held(
                "mirror-bytes-differ-from-pinned-primary-claim",
                state="hold-mirror-primary-byte-claim-mismatch",
            )
        claim = {
            "schema": SCHEMA,
            "role": role,
            "runId": run_id,
            "startUtc": start.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
            "finishUtc": finish.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
            "exitCode": 0,
            "archiveBytes": count,
            "archiveSha256": digest,
        }
        envelope = create_signed_claim(claim, key_id, owner_private_key)
        publish_signed_envelope(receipt, envelope)
        out.update(
            state="owner-observed-exit-and-archive-claim-published-not-certified",
            reason="direct-subprocess-exit-and-local-archive-hash-observed-by-adapter",
            archiveReadbackSha256Completed=True,
            signedReceiptPublished=True,
        )
        return out
    except (OSError, ValueError, TypeError, AttributeError, subprocess.TimeoutExpired):
        return held("producer-evidence-or-private-publication-unverified")
    finally:
        for fd in (receipt_fd, archive_fd, job_fd):
            if fd is not None:
                os.close(fd)
