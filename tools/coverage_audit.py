#!/usr/bin/env python3
"""Audit repository and per-function coverage from coverage.py JSON.

The aggregate number can hide completely untested functions inside otherwise
well-covered modules. This tool intersects coverage.py executed/missing line and
branch data with Python AST function spans and can enforce both:

* repository-wide branch-aware coverage >= --min-total
* every non-empty function's branch-aware coverage >= --min-function

Branch coverage is counted using individual coverage.py branch arcs, not merely
the number of source lines containing branches. Only executable lines reported
by coverage.py participate. Functions with no measurable executable lines are
reported as N/A rather than counted as failures.
"""
from __future__ import annotations

import argparse
import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class FunctionCoverage:
    file: str
    qualname: str
    start: int
    end: int
    statements: int
    covered_statements: int
    branches: int
    covered_branches: int
    percent: float | None


def _qualname(stack: list[str], name: str) -> str:
    return ".".join([*stack, name]) if stack else name


def _function_nodes(tree: ast.AST) -> Iterable[tuple[str, ast.AST]]:
    def walk(node: ast.AST, stack: list[str]):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qn = _qualname(stack, child.name)
                yield qn, child
                yield from walk(child, [*stack, child.name])
            elif isinstance(child, ast.ClassDef):
                yield from walk(child, [*stack, child.name])
            else:
                yield from walk(child, stack)

    yield from walk(tree, [])


def _branch_arcs(file_data: dict) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    executed = {
        (int(src), int(dst))
        for src, dst in file_data.get("executed_branches", [])
    }
    missing = {
        (int(src), int(dst))
        for src, dst in file_data.get("missing_branches", [])
    }
    return executed, missing


def audit(coverage_json: str | Path, repo_root: str | Path = ".") -> dict:
    root = Path(repo_root)
    payload = json.loads(Path(coverage_json).read_text(encoding="utf-8"))
    functions: list[FunctionCoverage] = []

    for rel, data in payload.get("files", {}).items():
        path = root / rel
        if not path.exists() or path.suffix != ".py":
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        except (OSError, SyntaxError):
            continue
        executed = set(map(int, data.get("executed_lines", [])))
        missing = set(map(int, data.get("missing_lines", [])))
        executable = executed | missing
        executed_arcs, missing_arcs = _branch_arcs(data)

        for qualname, node in _function_nodes(tree):
            start = int(getattr(node, "lineno", 0))
            end = int(getattr(node, "end_lineno", start))
            fn_lines = {line for line in executable if start <= line <= end}
            fn_covered = fn_lines & executed
            fn_executed_arcs = {
                arc for arc in executed_arcs if start <= arc[0] <= end
            }
            fn_missing_arcs = {
                arc for arc in missing_arcs if start <= arc[0] <= end
            }
            all_arcs = fn_executed_arcs | fn_missing_arcs
            denominator = len(fn_lines) + len(all_arcs)
            numerator = len(fn_covered) + len(fn_executed_arcs)
            percent = (100.0 * numerator / denominator) if denominator else None
            functions.append(
                FunctionCoverage(
                    file=rel,
                    qualname=qualname,
                    start=start,
                    end=end,
                    statements=len(fn_lines),
                    covered_statements=len(fn_covered),
                    branches=len(all_arcs),
                    covered_branches=len(fn_executed_arcs),
                    percent=percent,
                )
            )

    functions.sort(
        key=lambda row: (
            101.0 if row.percent is None else row.percent,
            -(row.statements + row.branches),
            row.file,
            row.qualname,
        )
    )
    return {
        "totals": payload.get("totals", {}),
        "functions": [asdict(row) for row in functions],
    }


def _total_percent(report: dict) -> float:
    totals = report.get("totals", {})
    return float(totals.get("percent_covered", 0.0))


def failures(report: dict, *, min_total: float, min_function: float) -> list[str]:
    problems: list[str] = []
    total = _total_percent(report)
    if total < min_total:
        problems.append(f"repository coverage {total:.2f}% < {min_total:.2f}%")
    for row in report.get("functions", []):
        pct = row.get("percent")
        if pct is not None and float(pct) < min_function:
            problems.append(
                f"{row['file']}:{row['start']} {row['qualname']} "
                f"{float(pct):.2f}% < {min_function:.2f}%"
            )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("coverage_json", nargs="?", default="coverage.json")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output", default="coverage-functions.json")
    parser.add_argument("--min-total", type=float, default=0.0)
    parser.add_argument("--min-function", type=float, default=0.0)
    parser.add_argument("--top", type=int, default=50)
    args = parser.parse_args()

    report = audit(args.coverage_json, args.repo_root)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    total = _total_percent(report)
    print(f"repository branch-aware coverage: {total:.2f}%")
    measurable = [r for r in report["functions"] if r["percent"] is not None]
    print(f"measurable functions: {len(measurable)}")
    print("lowest-covered functions:")
    for row in measurable[: max(0, args.top)]:
        print(
            f"  {row['percent']:6.2f}%  {row['file']}:{row['start']}  "
            f"{row['qualname']}  "
            f"stmts {row['covered_statements']}/{row['statements']}  "
            f"branches {row['covered_branches']}/{row['branches']}"
        )

    problems = failures(
        report,
        min_total=args.min_total,
        min_function=args.min_function,
    )
    if problems:
        print(f"coverage gate failed with {len(problems)} violation(s):")
        for problem in problems[:200]:
            print("  " + problem)
        if len(problems) > 200:
            print(f"  ... {len(problems) - 200} more")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
