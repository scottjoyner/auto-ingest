"""Explicit operator CLI: read-only lsblk observation, pure custody classification.

Separated from auto_ingest.custody because its package safety contracts
forbid process-execution imports in custody modules.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from .custody.next_card_discovery import discover, write_receipt

LSBLK_READ_ONLY_ARGS = [
    "lsblk", "-J", "-o",
    "NAME,TYPE,SIZE,FSTYPE,LABEL,UUID,RM,RO,TRAN,MOUNTPOINTS,MODEL",
]


def observe_devices() -> dict:
    """Read block metadata once. No mounting, writes, probing media contents."""
    output = subprocess.run(
        LSBLK_READ_ONLY_ARGS, capture_output=True, text=True,
        check=True, timeout=15,
    )
    return json.loads(output.stdout)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-handoff", type=Path, required=True)
    parser.add_argument("--expected-handoff-sha256", required=True)
    parser.add_argument("--out", type=Path, help="optional exclusive local OS-disk audit JSON")
    args = parser.parse_args(argv)
    try:
        result = discover(
            args.prior_handoff, args.expected_handoff_sha256,
            observe_devices(),
        )
        if args.out:
            write_receipt(args.out, result)
    except (OSError, ValueError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
