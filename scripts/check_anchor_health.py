#!/usr/bin/env python3
"""Read-only health check for the Scott ("me") speaker anchor.

Connects to Neo4j, counts ``GlobalSpeaker`` nodes with ``is_me=true``, and
asserts exactly ONE exists (the canonical Scott identity). Prints the count and
exits non-zero if it differs. READ-ONLY — performs no writes.

Usage:
    python3 scripts/check_anchor_health.py [--expected N] [--json]
"""
import argparse
import sys

from auto_ingest.speaker_health import check_anchor_health, count_is_me_global_speakers


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--expected", type=int, default=1,
                    help="Expected number of is_me GlobalSpeakers (default 1).")
    ap.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    args = ap.parse_args()

    try:
        count = check_anchor_health(expected=args.expected)
    except AssertionError as ex:
        count = None
        try:
            count = count_is_me_global_speakers()
        except Exception:
            count = None
        if args.json:
            print(f'{{"ok": false, "expected": {args.expected}, "actual": {count if count is not None else "null"}, "error": {ex!r}}}')
        else:
            print(f"ANCHOR UNHEALTHY: {ex}")
            if count is not None:
                print(f"is_me GlobalSpeaker count = {count} (expected {args.expected})")
        return 1
    except Exception as ex:  # connection / query failure
        if args.json:
            print(f'{{"ok": false, "error": {ex!r}}}')
        else:
            print(f"ANCHOR CHECK FAILED: {ex}")
        return 2

    if args.json:
        print(f'{{"ok": true, "expected": {args.expected}, "actual": {count}}}')
    else:
        print(f"ANCHOR HEALTHY: exactly {count} 'is_me' GlobalSpeaker node(s) (expected {args.expected}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
