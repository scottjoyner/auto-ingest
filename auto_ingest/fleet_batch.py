"""Generate fleet Tasks from a batch / folder so idle nodes can run them.

This is the missing link between auto-ingest and the unified fleet: it fans a
folder (or explicit item list) into capability-tagged `READY` Tasks on AssistX.
The fleet node-agent (`assistx.fleet_node_agent`) then polls by capability and
executes them — e.g. a Mac with ``yolo`` runs YOLO over dashcam frames, a weak
box runs a shell/ffmpeg job, etc.

Why HTTP and not Neo4j-direct: the canonical task-creation path is now
``POST /api/tasks`` on AssistX, which calls ``Neo4jClient.create_task_with_context``
internally and notifies auto-assign. This keeps auto-ingest decoupled from the
graph driver and auth surface.

Examples
--------
    # YOLO over every .mp4 in a folder (Macs with yolo capability pick these up)
    python -m auto_ingest.fleet_batch \\
        --assistx-url http://assistx:8000 --auth-pass fuck-you \\
        --input /nas/dashcam/2026-07 --glob '*.mp4' \\
        --capabilities yolo,vision,media \\
        --command 'python auto_ingest/dashcam/yolo_embeddings.py --input "{item}" --out /nas/drop/yolo' \\
        --title-prefix 'YOLO detect'

    # Generic shell batch (any script-capable node runs it)
    python -m auto_ingest.fleet_batch \\
        --assistx-url http://assistx:8000 --auth-pass fuck-you \\
        --items a.txt b.txt c.txt --capabilities script \\
        --command 'process.sh {item}'
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import uuid
from typing import Any, Optional

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None


def build_items(args: argparse.Namespace) -> list[str]:
    items: list[str] = list(args.items or [])
    if args.input and args.glob:
        pattern = os.path.join(args.input, args.glob)
        items.extend(sorted(glob.glob(pattern)))
    # de-dup, preserve order
    seen = set()
    out = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


def create_task(
    assistx_url: str,
    auth: Optional[tuple[str, str]],
    title: str,
    capabilities: list[str],
    command: Optional[str] = None,
    yolo_command: Optional[str] = None,
    payload: Optional[dict] = None,
    correlation_id: Optional[str] = None,
    priority: str = "background",
    task_type: str = "swarm_task",
) -> dict:
    if requests is None:
        raise RuntimeError("requests is required (pip install requests)")
    body: dict[str, Any] = {
        "title": title,
        "task_type": task_type,
        "status": "READY",
        "required_capabilities": capabilities,
        "priority": priority,
        "payload": payload or {},
    }
    if command is not None:
        body["payload"]["command"] = command
    if yolo_command is not None:
        body["payload"]["yolo_command"] = yolo_command
    if correlation_id:
        body["correlation_id"] = correlation_id
    resp = requests.post(
        f"{assistx_url}/api/tasks",
        json=body,
        auth=auth,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def run_batch(args: argparse.Namespace) -> int:
    auth = (args.auth_user, args.auth_pass) if args.auth_pass else None
    caps = [c.strip() for c in args.capabilities.split(",") if c.strip()]
    items = build_items(args)
    if not items:
        print("[fleet-batch] no items found; nothing to do", file=sys.stderr)
        return 1
    print(f"[fleet-batch] {len(items)} item(s), caps={caps}", file=sys.stderr)
    created = 0
    for item in items:
        correlation_id = str(uuid.uuid4())
        command = args.command.format(item=item) if args.command else None
        yolo_command = args.yolo_command.format(item=item) if args.yolo_command else None
        title = f"{args.title_prefix} {os.path.basename(item)}"
        payload = {
            "source": "auto-ingest",
            "item": item,
            "batch_id": args.batch_id or correlation_id,
            "correlation_id": correlation_id,
        }
        try:
            res = create_task(
                args.assistx_url, auth, title, caps,
                command=command, yolo_command=yolo_command,
                payload=payload, correlation_id=correlation_id,
                priority=args.priority,
            )
            created += 1
            print(f"[fleet-batch] created {res.get('task_id')} <- {item}", file=sys.stderr)
        except Exception as e:
            print(f"[fleet-batch] FAILED {item}: {e}", file=sys.stderr)
    print(f"[fleet-batch] done: {created}/{len(items)} tasks created", file=sys.stderr)
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Fan a folder/items into fleet Tasks")
    p.add_argument("--assistx-url", default=os.getenv("FLEET_ASSISTX_URL", "http://assistx:8000"))
    p.add_argument("--auth-user", default=os.getenv("FLEET_AUTH_USER", "admin"))
    p.add_argument("--auth-pass", default=os.getenv("FLEET_AUTH_PASS", "fuck-you"))
    p.add_argument("--input", help="Directory to scan")
    p.add_argument("--glob", default="*", help="Glob inside --input (default *)")
    p.add_argument("--items", nargs="*", help="Explicit item list")
    p.add_argument("--capabilities", required=True, help="comma-separated, e.g. yolo,vision")
    p.add_argument("--command", help="Shell command; {item} substituted per item")
    p.add_argument("--yolo-command", help="Vision command; {item} substituted per item")
    p.add_argument("--title-prefix", default="batch")
    p.add_argument("--batch-id", default=None)
    p.add_argument("--priority", default="background")
    args = p.parse_args()
    sys.exit(run_batch(args))


if __name__ == "__main__":
    main()
