"""Neo4j watchdog: poll the bolt URI, alert, and optionally auto-heal (O-G9).

Read-only by default. Opens a connection, runs a trivial ``RETURN 1`` and
verifies connectivity. On failure it classifies the error (service unavailable
vs memory-pool OOM vs other), logs it, and optionally POSTs a small JSON alert
to ``WATCHDOG_ALERT_URL`` if that env var is set.

Recovery (opt-in via ``--heal``): when an OOM / down / transient condition is
detected, the watchdog can restart the Neo4j service instead of only alerting.
The heal method is chosen from ``WATCHDOG_HEAL_METHOD`` (default ``auto``):

* ``systemctl`` — restart a local systemd unit
  (``WATCHDOG_SYSTEMD_UNIT``, default ``neo4j``).
* ``docker`` — ``docker restart`` a local container
  (``WATCHDOG_DOCKER_CONTAINER``, default ``neo4j``).
* ``ssh`` — SSH to the DB host and restart there
  (``WATCHDOG_SSH_HOST``, plus one of the above sub-methods).
* ``auto`` — pick a working method: local systemctl, else local docker,
  else ssh (if ``WATCHDOG_SSH_HOST`` is set).

All heal actions are guarded by a cooldown (``WATCHDOG_COOLDOWN_SECS``, default
300) tracked in a state file so a flapping Neo4j is not restarted in a tight
loop. The existing alert path is always preserved (heal is additive), and a
failed heal merely falls back to alert-only with a log message.

Usage::

    python3 scripts/neo4j_watchdog.py            # single check, exit 0/1
    python3 scripts/neo4j_watchdog.py --loop 60  # poll every 60s
    python3 scripts/neo4j_watchdog.py --heal     # auto-heal when down/oom
    python3 scripts/neo4j_watchdog.py --no-heal  # explicit opt-out (default)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import time
import urllib.request
from typing import Optional, Tuple

log = logging.getLogger("neo4j.watchdog")

OOM_MARKERS = ("MemoryPoolOutOfMemory", "OutOfMemory", "TransientError")
DOWN_MARKERS = ("ServiceUnavailable", "SessionExpired",
                "Connection refused", "defunct connection")

# Conditions that are considered recoverable via a service restart.
HEALABLE = ("oom", "down")

STATE_FILE = os.environ.get(
    "WATCHDOG_STATE_FILE", "/tmp/neo4j_watchdog_last_action.json")
DEFAULT_COOLDOWN_SECS = int(os.environ.get("WATCHDOG_COOLDOWN_SECS", "300"))


def _neo4j_conf() -> Tuple[str, str, str, str]:
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    db = os.environ.get("NEO4J_DB", "neo4j")
    pw = os.environ.get("NEO4J_PASSWORD", "")
    if not pw:
        try:
            from auto_ingest_config import get_neo4j_password
            pw = get_neo4j_password()
        except Exception:
            pw = ""
    return uri, user, pw, db


def classify_error(exc: BaseException) -> str:
    """Return 'oom', 'down', or 'error' for an exception."""
    s = f"{type(exc).__name__}: {exc}"
    if any(m in s for m in OOM_MARKERS):
        return "oom"
    if any(m in s for m in DOWN_MARKERS):
        return "down"
    return "error"


def send_alert(status: str, detail: str, url: Optional[str] = None) -> bool:
    """POST a small JSON alert to WATCHDOG_ALERT_URL (best-effort).

    Returns True if an alert was sent, False otherwise. Never raises.
    """
    url = url or os.environ.get("WATCHDOG_ALERT_URL")
    if not url:
        return False
    payload = json.dumps({"service": "neo4j", "status": status,
                          "detail": detail, "ts": int(time.time())}).encode()
    try:
        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=5).read()
        return True
    except Exception as e:  # pragma: no cover - network dependent
        log.warning("watchdog alert POST failed: %s", e)
        return False


def _load_last_action() -> dict:
    """Load the heal state file, returning {} on any problem."""
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _save_last_action(action: str, detail: str) -> None:
    """Persist the last heal attempt atomically-ish. Never raises."""
    payload = {"action": action, "detail": detail, "ts": int(time.time())}
    try:
        tmp = f"{STATE_FILE}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        os.replace(tmp, STATE_FILE)
    except Exception as e:  # pragma: no cover - fs dependent
        log.warning("could not persist heal state: %s", e)


def _cooldown_ok(cooldown_secs: int) -> bool:
    """Return True if enough time has passed since the last heal action."""
    last = _load_last_action().get("ts")
    if last is None:
        return True
    return (time.time() - float(last)) >= float(cooldown_secs)


def _run(cmd: list[str]) -> Tuple[int, str]:
    """Run a command list, returning (returncode, combined_output).

    Never raises on command failure; only on unrecoverable exec errors which
    are captured and returned as a non-zero code with the message.
    """
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        out = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode, out.strip()
    except Exception as e:  # pragma: no cover - exec dependent
        return 1, f"{type(e).__name__}: {e}"


def _heal_command(method: str) -> Optional[list[str]]:
    """Translate a heal method into a command list, or None if not actionable.

    ``ssh`` resolves to an ssh invocation wrapping the chosen remote sub-method
    (systemctl or docker). Local methods return their direct command.
    """
    unit = os.environ.get("WATCHDOG_SYSTEMD_UNIT", "neo4j")
    container = os.environ.get("WATCHDOG_DOCKER_CONTAINER", "neo4j")
    ssh_host = os.environ.get("WATCHDOG_SSH_HOST", "")

    if method == "systemctl":
        return ["systemctl", "restart", unit]
    if method == "docker":
        return ["docker", "restart", container]
    if method == "ssh":
        if not ssh_host:
            return None
        sub = os.environ.get("WATCHDOG_SSH_METHOD", "systemctl")
        if sub == "docker":
            remote = f"docker restart {container}"
        else:
            remote = f"systemctl restart {unit}"
        return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                ssh_host, remote]
    return None


def _detect_method() -> str:
    """Auto-detect a heal method; prefer local, fall back to ssh."""
    unit = os.environ.get("WATCHDOG_SYSTEMD_UNIT", "neo4j")
    code, _ = _run(["systemctl", "is-active", "--quiet", unit])
    if code == 0:
        return "systemctl"
    code, _ = _run(["docker", "inspect", os.environ.get(
        "WATCHDOG_DOCKER_CONTAINER", "neo4j")])
    if code == 0:
        return "docker"
    if os.environ.get("WATCHDOG_SSH_HOST"):
        return "ssh"
    return "none"


def attempt_heal(kind: str, *, cooldown_secs: int = DEFAULT_COOLDOWN_SECS,
                 method: str = "auto") -> bool:
    """Attempt a recovery restart for a healable ``kind``.

    Returns True if a heal command was issued, False otherwise (including when
    on cooldown, no method available, or the command failed). A failure is
    logged and is NOT raised — the caller should still alert.
    """
    if kind not in HEALABLE:
        return False
    if not _cooldown_ok(cooldown_secs):
        log.warning("heal skipped: cooldown active (<%ss since last action)",
                    cooldown_secs)
        return False

    resolved = _detect_method() if method == "auto" else method
    cmd = _heal_command(resolved)
    if not cmd:
        log.warning("heal skipped: no usable method for %r", resolved)
        return False

    log.warning("heal: restarting neo4j via %r", " ".join(cmd))
    code, out = _run(cmd)
    if code != 0:
        log.error("heal FAILED (%s): %s", " ".join(cmd), out)
        _save_last_action("failed", f"{kind}:{out}")
        return False

    log.info("heal OK: %s", " ".join(cmd))
    _save_last_action("heal", f"{kind}:{' '.join(cmd)}")
    return True


def check_neo4j(uri: Optional[str] = None, user: Optional[str] = None,
                password: Optional[str] = None, db: Optional[str] = None,
                *, alert: bool = True) -> Tuple[bool, str]:
    """Return (True, 'ok') if Neo4j answers; (False, kind) if not.

    Read-only: runs ``RETURN 1``. Import-safe — if the neo4j driver is missing
    it logs and returns (False, 'error') without raising.
    """
    c_uri, c_user, c_pw, c_db = _neo4j_conf()
    uri = uri or c_uri
    user = user or c_user
    password = password if password is not None else c_pw
    db = db or c_db
    try:
        from neo4j import GraphDatabase
    except Exception as e:
        log.error("neo4j driver unavailable: %s", e)
        return False, "error"

    driver = None
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session(database=db) as s:
            val = s.run("RETURN 1 AS ok").single()["ok"]
        if val == 1:
            log.info("neo4j OK (%s)", uri)
            return True, "ok"
        log.warning("neo4j unexpected response: %r", val)
        return False, "error"
    except Exception as e:
        kind = classify_error(e)
        log.error("neo4j check failed [%s]: %s", kind, e)
        if alert:
            send_alert(kind, f"{type(e).__name__}: {e}")
        return False, kind
    finally:
        if driver is not None:
            try:
                driver.close()
            except Exception:
                pass


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Neo4j watchdog (read-only).")
    ap.add_argument("--loop", type=int, default=0,
                    help="Poll every N seconds (0 = single check).")
    ap.add_argument("--no-alert", action="store_true",
                    help="Do not POST to WATCHDOG_ALERT_URL on failure.")
    ap.add_argument("--heal", action="store_true",
                    help="Auto-restart Neo4j on OOM/down (opt-in).")
    ap.add_argument("--no-heal", action="store_true", dest="no_heal",
                    help="Explicitly disable auto-restart (default).")
    ap.add_argument("--cooldown", type=int, default=DEFAULT_COOLDOWN_SECS,
                    help="Min seconds between heal attempts (default 300).")
    ap.add_argument("--heal-method", dest="heal_method", default="auto",
                    help="systemctl|docker|ssh|auto (default auto).")
    args = ap.parse_args()

    if args.loop > 0:
        while True:
            ok, kind = check_neo4j(alert=not args.no_alert)
            if not ok and args.heal:
                attempt_heal(kind, cooldown_secs=args.cooldown,
                             method=args.heal_method)
            time.sleep(args.loop)
        return 0  # pragma: no cover - interrupted by signal
    ok, kind = check_neo4j(alert=not args.no_alert)
    if not ok and args.heal:
        attempt_heal(kind, cooldown_secs=args.cooldown,
                     method=args.heal_method)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
