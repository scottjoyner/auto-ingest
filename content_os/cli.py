from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import core


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="content-os", description="Local-first approval-gated Content OS"
    )
    p.add_argument(
        "--root",
        default=None,
        help="Content OS root (default: ./content-os or current Content OS directory)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("init")
    i.add_argument("--root", default=None)
    i.add_argument("--force", action="store_true")
    c = sub.add_parser("capture")
    c.add_argument("--title", required=True)
    c.add_argument("--text", required=True)
    c.add_argument("--source")
    c.add_argument(
        "--route",
        default="auto",
        choices=["auto", "ORIGINAL", "REPURPOSE", "REWRITE", "RESEARCH_IDEATE"],
    )
    c.add_argument(
        "--no-run",
        action="store_true",
        help="Only append to inbox.md; do not create a run folder.",
    )
    n = sub.add_parser("new-run")
    n.add_argument("--title", required=True)
    n.add_argument(
        "--route",
        default="auto",
        choices=["auto", "ORIGINAL", "REPURPOSE", "REWRITE", "RESEARCH_IDEATE"],
    )
    n.add_argument(
        "--format",
        default="x_post",
        choices=["x_post", "x_thread", "linkedin", "blog", "newsletter", "custom"],
    )
    for name in ["route", "brief", "draft", "verify", "archive"]:
        sp = sub.add_parser(name)
        sp.add_argument("run_id")
        if name == "archive":
            sp.add_argument("--force", action="store_true")
    a = sub.add_parser("approve")
    a.add_argument("run_id")
    a.add_argument("--force", action="store_true")
    a.add_argument("--reason", default="")
    sh = sub.add_parser("scheduler-handoff")
    sh.add_argument("run_id")
    sh.add_argument(
        "--platform",
        default="custom",
        choices=["x", "linkedin", "threads", "blog", "newsletter", "custom"],
    )
    pe = sub.add_parser("postiz-export")
    pe.add_argument("run_id")
    pe.add_argument("--out", required=True)
    pe.add_argument(
        "--send-to-postiz",
        action="store_true",
        help="Reserved for explicit draft creation; auto-publish is never performed.",
    )
    f = sub.add_parser("feedback")
    f.add_argument("run_id")
    for x in ["views", "bookmarks", "likes", "replies", "reposts"]:
        f.add_argument(f"--{x}", required=True, type=int)
    f.add_argument("--notes", default="")
    scan = sub.add_parser("scan-inputs")
    scan.add_argument("path")
    scan.add_argument("--out")
    scan.add_argument("--limit", type=int)
    ingest = sub.add_parser("ingest-source")
    ingest.add_argument("path")
    ingest.add_argument("--title")
    ingest.add_argument(
        "--route",
        default="auto",
        choices=["auto", "ORIGINAL", "REPURPOSE", "REWRITE", "RESEARCH_IDEATE"],
    )
    ingest.add_argument(
        "--format",
        default="custom",
        choices=["x_post", "x_thread", "linkedin", "blog", "newsletter", "custom"],
    )
    ingest.add_argument(
        "--no-run",
        action="store_true",
        help="Only write the extracted source into stores/proof/.",
    )
    ingest.add_argument(
        "--no-proof",
        action="store_true",
        help="Create the run without writing a stores/proof entry.",
    )
    sub.add_parser("list-adapters")
    sub.add_parser("status")
    sub.add_parser("doctor")
    return p


def print_table(rows, headers):
    widths = [len(h) for h in headers]
    data = [
        [str(getattr(r, h.lower().replace(" ", "_"), "")) for h in headers]
        for r in rows
    ]
    for row in data:
        for i, v in enumerate(row):
            widths[i] = max(widths[i], len(v))
    print(" | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print("-+-".join("-" * w for w in widths))
    for row in data:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    root = core.resolve_root(getattr(args, "root", None))
    try:
        if args.cmd == "init":
            root = core.resolve_root(args.root)
            made = core.init(root, args.force)
            print(f"Initialized Content OS at {root} ({len(made)} files written)")
        elif args.cmd == "capture":
            path = core.capture(
                root,
                args.title,
                args.text,
                args.source,
                args.route,
                create_run_folder=not args.no_run,
            )
            print(path.name if path else "captured_to_inbox")
        elif args.cmd == "new-run":
            path = core.create_run(root, args.title, args.route, args.format)
            print(path.name)
        elif args.cmd == "route":
            print(core.route_run(root, args.run_id))
        elif args.cmd == "brief":
            print(core.make_brief(root, args.run_id))
        elif args.cmd == "draft":
            print(core.draft(root, args.run_id))
        elif args.cmd == "verify":
            print(core.verify(root, args.run_id))
        elif args.cmd == "approve":
            core.approve(root, args.run_id, args.force, args.reason)
            print("approved")
        elif args.cmd == "scheduler-handoff":
            print(core.scheduler_handoff(root, args.run_id, args.platform))
        elif args.cmd == "postiz-export":
            if args.send_to_postiz:
                print(
                    "--send-to-postiz acknowledged: this implementation exports only; no auto-publish or API send is performed."
                )
            core.postiz_export(root, args.run_id, Path(args.out))
            print(args.out)
        elif args.cmd == "feedback":
            print(
                core.feedback(
                    root,
                    args.run_id,
                    args.views,
                    args.bookmarks,
                    args.likes,
                    args.replies,
                    args.reposts,
                    args.notes,
                )
            )
        elif args.cmd == "archive":
            print(core.archive(root, args.run_id, args.force))
        elif args.cmd == "scan-inputs":
            docs, report_path = core.scan_inputs(
                root, Path(args.path), Path(args.out) if args.out else None, args.limit
            )
            if report_path:
                print(report_path)
            else:
                for doc in docs:
                    print(
                        f"{doc.path}	{doc.source_type}	{doc.suggested_route}	{len(doc.text)} chars"
                    )
        elif args.cmd == "ingest-source":
            run_path, proof_path, source = core.import_source(
                root,
                Path(args.path),
                title=args.title,
                route=args.route,
                fmt=args.format,
                as_proof=not args.no_proof,
                create_run_folder=not args.no_run,
            )
            print(f"source_type={source.source_type}")
            if proof_path:
                print(f"proof={proof_path}")
            if run_path:
                print(f"run={run_path.name}")
        elif args.cmd == "list-adapters":
            for item in core.adapter_inventory(root):
                print(
                    f"{item['kind']}\t{item['name']}\t{','.join(item.get('extensions', []))}"
                )
        elif args.cmd == "status":
            print_table(
                core.status(root),
                ["id", "title", "route", "state", "score", "next_action"],
            )
        elif args.cmd == "doctor":
            issues = core.doctor(root)
            if issues:
                print("Doctor found issues:")
                [print(f"- {i}") for i in issues]
                return (
                    1
                    if any(not i.startswith("LLM not configured") for i in issues)
                    else 0
                )
            print("Doctor passed.")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
