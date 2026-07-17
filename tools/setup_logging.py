"""Logging setup helper: rotating file handlers for the pipeline (O-G8).

Best-effort, import-safe. Configures a :class:`~logging.handlers.RotatingFileHandler`
(default 10 MB × 5 backups) on the root logger (and optionally named pipeline
loggers) so long-running jobs — speaker linking, dashcam render, ingest — don't
grow an unbounded log file. Existing stdout/stderr output is preserved (the
console handler is left intact / added if missing).

Usage::

    from tools.setup_logging import setup_logging
    setup_logging()                         # ./logs/auto-ingest.log
    setup_logging(name="speaker-link")      # ./logs/speaker-link.log

Never raises: any failure (unwritable dir, etc.) is swallowed and logging falls
back to whatever was already configured.
"""
from __future__ import annotations

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Iterable, Optional

DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
DEFAULT_BACKUPS = 5

# Loggers that log heavily during long runs; attaching the rotating handler to
# these (via propagation to root) keeps their output bounded too.
PIPELINE_LOGGERS = (
    "shorts", "shorts.render", "shorts.persona",
    "speakers", "diarize", "link_global_speakers",
    "ingest", "yolo",
)


def _log_dir() -> Path:
    return Path(os.environ.get("AUTO_INGEST_LOG_DIR",
                               str(Path.cwd() / "logs")))


def setup_logging(name: str = "auto-ingest", *,
                  level: Optional[int] = None,
                  log_dir: Optional[Path] = None,
                  max_bytes: int = DEFAULT_MAX_BYTES,
                  backups: int = DEFAULT_BACKUPS,
                  console: bool = True,
                  extra_loggers: Iterable[str] = ()) -> Optional[Path]:
    """Attach a rotating file handler (+ optional console) to the root logger.

    Returns the log file path on success, or ``None`` if setup was skipped
    (best-effort; never raises).
    """
    try:
        lvl = level if level is not None else getattr(
            logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO)
        d = Path(log_dir) if log_dir is not None else _log_dir()
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{name}.log"

        root = logging.getLogger()
        root.setLevel(lvl)
        fmt = logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s")

        # Avoid duplicating our rotating handler for the same file.
        existing = {getattr(h, "baseFilename", None)
                    for h in root.handlers
                    if isinstance(h, logging.handlers.RotatingFileHandler)}
        if str(path) not in existing:
            fh = logging.handlers.RotatingFileHandler(
                path, maxBytes=max_bytes, backupCount=backups,
                encoding="utf-8")
            fh.setLevel(lvl)
            fh.setFormatter(fmt)
            root.addHandler(fh)

        if console and not any(isinstance(h, logging.StreamHandler)
                               and not isinstance(h, logging.FileHandler)
                               for h in root.handlers):
            ch = logging.StreamHandler()
            ch.setLevel(lvl)
            ch.setFormatter(fmt)
            root.addHandler(ch)

        # Ensure named pipeline loggers propagate to root (so they rotate too).
        for lname in list(PIPELINE_LOGGERS) + list(extra_loggers):
            logging.getLogger(lname).propagate = True
        return path
    except Exception:  # pragma: no cover - defensive, never break the caller
        return None


if __name__ == "__main__":
    p = setup_logging()
    logging.getLogger("shorts").info("setup_logging smoke test -> %s", p)
    print(f"log file: {p}")
