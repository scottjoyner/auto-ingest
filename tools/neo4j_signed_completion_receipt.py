"""Offline, role-pinned Ed25519 attestation for *claims* about Neo4j backups.

No production services, backup jobs, keys, databases, remote hosts or archive
bytes are accessed. Even a valid signature proves ONLY that the holder of a
previously approved private key signed these claimed fields; it does not
prove that the job actually ran, completed, wrote correct bytes, or restored.
"""

import base64
import json
import os
import re
import secrets
import stat
from pathlib import Path

from neo4j_backup_receipt_verifier import (
    DEFAULT_READ_MAX,
    FIELDS,
    MAX_READ_MAX,
    MAX_RECEIPT,
    RUN_RE,
    compare_claims,
    compare_independently_read_files,
    utc,
)
from neo4j_backup_receipt_verifier import (
    load as load_bounded_receipt,
)
from neo4j_backup_receipt_verifier import (
    validate as validate_claim,
)

SCHEMA = "neo4j-backup-signed-claim-v1"
DOMAIN = b"neo4j-backup-signed-claim-v1\x00"
KEY_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
ENVELOPE_KEYS = {"schema", "keyId", "claim", "signature"}
MAX_ATTESTATION = MAX_RECEIPT
TRUSTED_ROLES = ("primary", "mirror")


def result(state="unverified", reason="signed-claim-evidence-unverified"):
    return {
        "schema": "neo4j-backup-signed-claim-assessment-v1",
        "state": state,
        "reason": reason,
        "approvedProducerKeyVerified": False,
        "claimedExitZero": False,
        "archiveBytesIndependentlyVerified": False,
        "sourceMirrorByteIdentityVerified": False,
        "databaseConsistencyVerified": False,
        "databaseRestoreVerified": False,
        "backupJobExecutionVerified": False,
        "deleteAuthorized": False,
        "hostChangeAuthorized": False,
        "archiveBytesRead": 0,
        "nasWrites": 0,
        "remoteWrites": 0,
        "modelRequestsSent": 0,
    }


def _claim_eligible(claim):
    if not isinstance(claim, dict) or set(claim) != FIELDS:
        return False
    # A signed failure receipt would need a separate schema because a failed
    # backup may not have produced any archive/hash at all. Fail closed here.
    return validate_claim(claim)["state"] == "unsigned-success-claim-not-certified"


def _canonical(claim):
    return json.dumps(
        claim, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def _signed_bytes(key_id, claim):
    return DOMAIN + key_id.encode("ascii") + b"\x00" + _canonical(claim)


def _strict_envelope(value):
    if not isinstance(value, dict) or set(value) != ENVELOPE_KEYS:
        return False
    if value.get("schema") != SCHEMA:
        return False
    key_id = value.get("keyId")
    signature = value.get("signature")
    if not isinstance(key_id, str) or not KEY_RE.fullmatch(key_id):
        return False
    if not isinstance(signature, str) or len(signature) != 86:
        return False
    if not re.fullmatch(r"[A-Za-z0-9_-]{86}", signature):
        return False
    try:
        sig = base64.urlsafe_b64decode(signature + "==")
        if len(sig) != 64 or base64.urlsafe_b64encode(sig).rstrip(b"=").decode() != signature:
            return False
    except (ValueError, TypeError):
        return False
    return _claim_eligible(value.get("claim"))


def create_signed_claim(claim, key_id, private_key):
    """Only for an approved owner-held Ed25519 private-key object.

    Passing a key does not establish that the backup job ran or succeeded.
    A producer integration MUST bind this operation to a checked process
    exit and bounded independent archive readback, separately reviewed.
    No key generation, key loading, subprocess or disk writes here.
    """
    if not isinstance(key_id, str) or not KEY_RE.fullmatch(key_id):
        raise ValueError("invalid opaque key identifier")
    if not _claim_eligible(claim):
        raise ValueError("unsupported or unsuccessful claim")
    if not callable(getattr(private_key, "sign", None)):
        raise ValueError("a trusted in-memory signing key is required")
    sig = private_key.sign(_signed_bytes(key_id, claim))
    if not isinstance(sig, bytes) or len(sig) != 64:
        raise ValueError("invalid signing result")
    envelope = {
        "schema": SCHEMA,
        "keyId": key_id,
        "claim": dict(claim),
        "signature": base64.urlsafe_b64encode(sig).rstrip(b"=").decode("ascii"),
    }
    if not _strict_envelope(envelope):
        raise ValueError("signed envelope failed canonical validation")
    return envelope


def verify_signed_claim(envelope, approved_role_keys):
    """Role-pinned, out-of-band trusted public keys ONLY.

    approved_role_keys = {
      'primary': {'keyId': 'approved-primary', 'publicKeyRaw': <32 bytes>},
      'mirror': {'keyId': 'approved-mirror', 'publicKeyRaw': <32 bytes>},
    }
    Never accept verification keys provided by the untrusted envelope.
    """
    out = result()
    if not _strict_envelope(envelope) or not isinstance(approved_role_keys, dict):
        return out
    claim = envelope["claim"]
    role = claim["role"]
    if role not in TRUSTED_ROLES:
        return out
    configured = approved_role_keys.get(role)
    if not isinstance(configured, dict) or set(configured) != {"keyId", "publicKeyRaw"}:
        return out
    pinned_id = configured["keyId"]
    key_raw = configured["publicKeyRaw"]
    if (
        not isinstance(pinned_id, str)
        or not KEY_RE.fullmatch(pinned_id)
        or envelope["keyId"] != pinned_id
        or type(key_raw) is not bytes
        or len(key_raw) != 32
    ):
        return out
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError:
        return result(reason="trusted-crypto-dependency-unavailable")
    try:
        Ed25519PublicKey.from_public_bytes(key_raw).verify(
            base64.urlsafe_b64decode(envelope["signature"] + "=="),
            _signed_bytes(pinned_id, claim),
        )
    except (InvalidSignature, ValueError, TypeError):
        return result(reason="role-pinned-signature-invalid")
    out.update(
        state="approved-key-signed-claim-content-and-restore-unverified",
        reason="signature-matches-preapproved-role-key-but-exit-is-still-an-assertion",
        approvedProducerKeyVerified=True,
        claimedExitZero=True,
    )
    return out


def verify_signed_pair(primary_envelope, mirror_envelope, approved_role_keys):
    """Check *two* independent approved role keys and cold-stage ordering."""
    out = result()
    if not isinstance(approved_role_keys, dict):
        return out
    primary = verify_signed_claim(primary_envelope, approved_role_keys)
    mirror = verify_signed_claim(mirror_envelope, approved_role_keys)
    if not (primary["approvedProducerKeyVerified"] and mirror["approvedProducerKeyVerified"]):
        return out
    if (
        primary_envelope["claim"]["role"] != "primary"
        or mirror_envelope["claim"]["role"] != "mirror"
    ):
        return out
    if primary_envelope["keyId"] == mirror_envelope["keyId"]:
        return result(reason="primary-and-mirror-require-distinct-approved-role-key-ids")
    # Different labels pointing to the SAME public key are not two signers.
    if (
        approved_role_keys["primary"]["publicKeyRaw"]
        == approved_role_keys["mirror"]["publicKeyRaw"]
    ):
        return result(reason="primary-and-mirror-require-distinct-approved-public-keys")
    if (
        compare_claims(primary_envelope["claim"], mirror_envelope["claim"])["state"]
        != "matching-unsigned-claims-not-certified"
    ):
        return result(reason="signed-stage-claims-do-not-align")
    out.update(
        state="two-approved-keys-signed-matching-claims-not-byte-proof",
        reason="two-role-signatures-valid-but-archive-integrity-and-restore-unverified",
        approvedProducerKeyVerified=True,
        claimedExitZero=True,
    )
    return out


def verify_signed_pair_with_readback(
    primary_envelope,
    mirror_envelope,
    approved_role_keys,
    *,
    expected_run_id,
    not_before_utc,
    not_after_utc,
    primary_path,
    mirror_path,
    max_bytes=DEFAULT_READ_MAX,
):
    """Explicit bounded source AND mirror byte readback for authenticated claims.

    This only reports that two independently read archive byte sequences agree
    with two role-pinned signed claims. It does NOT authenticate the actual
    backup process exit, authoritatively bind a DB transaction generation,
    prove an isolated restore, or grant retention/deletion approval.

    Caller must specify the expected run and independent time interval, two
    distinct explicit archive paths and an I/O budget. No default path or
    automatic bulk scan exists.
    """
    out = result()
    if (
        not isinstance(expected_run_id, str)
        or not RUN_RE.fullmatch(expected_run_id)
        or type(max_bytes) is not int
        or not 1 <= max_bytes <= MAX_READ_MAX
    ):
        return result(reason="invalid-independent-run-or-io-bound")
    lower = utc(not_before_utc)
    upper = utc(not_after_utc)
    if (
        lower is None
        or upper is None
        or not lower <= upper
        or (upper - lower).total_seconds() > 24 * 3600
    ):
        return result(reason="invalid-independent-time-window")
    pair = verify_signed_pair(primary_envelope, mirror_envelope, approved_role_keys)
    if pair["state"] != "two-approved-keys-signed-matching-claims-not-byte-proof":
        return result(reason="pair-of-approved-signed-claims-required")
    primary = primary_envelope["claim"]
    mirror = mirror_envelope["claim"]
    if (
        primary["runId"] != expected_run_id
        or mirror["runId"] != expected_run_id
        or not (
            lower
            <= utc(primary["startUtc"])
            <= utc(primary["finishUtc"])
            <= utc(mirror["startUtc"])
            <= utc(mirror["finishUtc"])
            <= upper
        )
    ):
        return result(reason="claim-outside-independent-run-or-time-scope")
    # Both signatures have passed verification BEFORE this can read any file.
    # The base verifier separately checks inode identity, symlink refusal,
    # declared SHA256, bounded reads and post-read path identity stability.
    readback = compare_independently_read_files(
        primary,
        primary_path,
        mirror,
        mirror_path,
        max_bytes=max_bytes,
    )
    if (
        readback["state"] != "readback-byte-identical-writer-and-restore-unverified"
        or readback.get("primarySourceBytesIndependentlyVerified") is not True
        or readback.get("mirrorSourceByteIdentityVerified") is not True
        or type(readback.get("archiveReads")) is not int
        or not 0 < readback["archiveReads"] <= 2 * max_bytes
    ):
        return result(reason="signed-claims-match-but-independent-readback-unverified")
    out.update(
        state="role-signed-and-independent-byte-readback-verified-not-a-restore",
        reason="two-role-pinned-claims-and-bounded-source-mirror-sha256-verified-not-process-exit",
        approvedProducerKeyVerified=True,
        claimedExitZero=True,
        archiveBytesIndependentlyVerified=True,
        sourceMirrorByteIdentityVerified=True,
        archiveBytesRead=readback["archiveReads"],
    )
    return out


def load_signed_envelope(path):
    """Bounded/no-follow JSON read. Never reads a backup archive or key."""
    obj = load_bounded_receipt(path)
    return obj if _strict_envelope(obj) else None


def publish_signed_envelope(directory, envelope):
    """Owner-run, exclusive no-overwrite publication of ONE small signed claim.

    Only use an EXISTING owner-only directory with trusted local storage
    custody. Atomic hardlink publication has a brief pre-fsync visibility
    interval. If the final directory sync fails, make a best-effort
    inode-checked rollback and always raise; the caller must reconcile
    the private destination after any ambiguous storage error.
    """
    if not _strict_envelope(envelope):
        raise ValueError("invalid signed claim")
    parent = Path(directory)
    if parent.is_symlink():
        raise ValueError("symlink directory rejected")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    dir_fd = os.open(parent, flags)
    name = f"{envelope['claim']['runId']}-{envelope['claim']['role']}.signed.json"
    payload = _canonical(envelope) + b"\n"
    fd = None
    temporary_name = None
    published = False
    committed = False
    original_inode = None
    try:
        if len(payload) > MAX_ATTESTATION:
            raise ValueError("signed claim too large")
        info = os.fstat(dir_fd)
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o077:
            raise ValueError("private owner-only output directory required")
        file_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        temporary_name = f".tmp-{secrets.token_hex(16)}-{name}"
        fd = os.open(temporary_name, file_flags, 0o600, dir_fd=dir_fd)
        original = os.fstat(fd)
        original_inode = (original.st_dev, original.st_ino)
        offset = 0
        while offset < len(payload):
            written = os.write(fd, payload[offset:])
            if written <= 0:
                raise OSError("short private receipt write")
            offset += written
        os.fsync(fd)
        os.close(fd)
        fd = None
        # Atomic no-overwrite hardlink, then drop the temporary name.
        # The FINAL directory sync must happen AFTER dropping the temp
        # link, so no failing later cleanup sync can look like success.
        os.link(
            temporary_name,
            name,
            src_dir_fd=dir_fd,
            dst_dir_fd=dir_fd,
            follow_symlinks=False,
        )
        published = True
        os.unlink(temporary_name, dir_fd=dir_fd)
        temporary_name = None
        os.fsync(dir_fd)
        committed = True
    except (OSError, ValueError):
        if published and not committed:
            # A failed directory sync or temporary unlink must not
            # silently leave an apparently final receipt behind.
            try:
                target = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
                if (target.st_dev, target.st_ino) != original_inode:
                    raise OSError("published final receipt identity changed")
                os.unlink(name, dir_fd=dir_fd)
                os.fsync(dir_fd)
            except OSError as cleanup_error:
                raise OSError(
                    "receipt publication uncertain; reconcile private directory before retry"
                ) from cleanup_error
        raise
    finally:
        if fd is not None:
            os.close(fd)
        if temporary_name is not None:
            try:
                os.unlink(temporary_name, dir_fd=dir_fd)
            except FileNotFoundError:
                pass
        os.close(dir_fd)
    return parent / name
