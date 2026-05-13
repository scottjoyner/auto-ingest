from __future__ import annotations

import json
import os
import re
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .adapters import (
    AdapterRule,
    SourceDocument,
    extract_source,
    scan_path,
    scan_report,
)
from .io import append, atomic_write, dump_yaml, load_yaml, read_text, write_if_missing
from .models import ContentObject, State, now_iso

DEFAULT_SCORE_THRESHOLD = 8
WIN_BOOKMARK_RATE = 0.05
LOSE_BOOKMARK_RATE = 0.01
VALID_ROUTES = {"auto", "ORIGINAL", "REPURPOSE", "REWRITE", "RESEARCH_IDEATE"}
VALID_FORMATS = {"x_post", "x_thread", "linkedin", "blog", "newsletter", "custom"}

STRUCTURE_DIRS = [
    "strategy",
    "voice",
    "stores/ideas",
    "stores/hooks",
    "stores/proof",
    "stores/feedback",
    "stores/winners",
    "stores/losers",
    "runs/active",
    "runs/archive",
    "modules/writer/references",
    "modules/writer/templates",
    "workflows",
    "integrations",
]

TEMPLATES: dict[str, str] = {
    "strategy/positioning.md": "# Positioning\n\nWho you help, what change you create, and why your viewpoint is credible.\n",
    "strategy/audience.md": "# Audience\n\n## Primary reader\n- Role:\n- Pain:\n- Job-to-be-done:\n- What they save/bookmark:\n",
    "strategy/pillars.md": "# Content Pillars\n\n1. Systems\n2. Proof and build notes\n3. Opinionated lessons\n",
    "strategy/source-watchlist.md": "# Source Watchlist\n\nUser-approved sources, APIs, exports, or manual inputs only. Do not scrape platforms.\n",
    "voice/voice-profile.md": "# Voice Profile\n\n## Voice anchors\n- Clear, specific, and useful.\n- Short sentences with concrete nouns.\n- Shows the work; does not posture.\n",
    "voice/master-avoid-slop.md": '# Master Avoid-Slop Rules\n\nFlag or remove these patterns unless intentionally justified:\n\n- promotional language: "groundbreaking", "game-changing"\n- significance inflation: "pivotal moment", "testament to"\n- vague attribution: "experts believe", "studies show"\n- false agency: "the system compounds", "the data tells us"\n- rhetorical setups: "the question is whether"\n- staccato cliché: "No X. No Y. No Z."\n- em dash overuse\n- filler adverbs: "actually", "literally", "quietly"\n- generic AI phrasing\n- unsupported superlatives\n',
    "stores/inbox.md": "# Inbox\n\nCaptured ideas and raw inputs.\n",
    "stores/workboard.md": "# Workboard\n\nUse `content-os status` for generated active run state.\n",
    "modules/writer/SKILL.md": "# Writer Module\n\nUse compact writer packets. Do not invent proof, facts, clients, screenshots, metrics, or attribution. Preserve route constraints and human approval gates.\n",
    "workflows/idea-to-published-post.md": "# Idea to Published Post\n\nCapture → route → brief → draft → verify → human review → approve → scheduler handoff → manual scheduling/export → feedback → learn.\n",
    "workflows/verifier-checklist.md": "# Verifier Checklist\n\nScore bookmarkability, run avoid-slop checks, require human review, and never approve failed verification without an explicit force reason.\n",
    "workflows/scheduler-handoff.md": "# Scheduler Handoff\n\nExport only after approval. Postiz payloads are handoff/draft data and require manual review. No auto-publishing.\n",
    "workflows/feedback-loop.md": "# Feedback Loop\n\nRecord views, bookmarks, likes, replies, reposts, and notes. Propose voice/hook changes; never overwrite voice rules automatically.\n",
    "integrations/autoingest.md": "# Autoingest Integration\n\nUse this repo's local transcripts, summaries, metadata exports, dashcam/audio artifacts, and analysis notes as user-owned inputs. Content OS never scrapes platforms and never transcribes binary media automatically; point it at approved local outputs with `scan-inputs` and `ingest-source`.\n",
    "integrations/source-adapters.json": '[\n  {\n    "name": "autoingest-cypher-note",\n    "extensions": [".cy"],\n    "source_type": "graph_query_note",\n    "route": "RESEARCH_IDEATE",\n    "format": "custom",\n    "treat_as_text": true\n  },\n  {\n    "name": "custom-json-note-example",\n    "extensions": [".notejson"],\n    "source_type": "custom_json_note",\n    "text_keys": ["text", "body", "summary"],\n    "title_keys": ["title", "name"],\n    "route": "REPURPOSE",\n    "format": "custom"\n  }\n]\n',
    "config.yaml": "score_threshold: 8\nwinning_bookmark_rate: 0.05\nlosing_bookmark_rate: 0.01\ndefault_publish_window: 'Manual choice required'\ninput_ingest:\n  max_scan_files: 100\n  preferred_proof_store: stores/proof\n  autoingest_patterns:\n    transcripts: ['**/*transcript*.txt', '**/*transcription*.json', '**/*.srt', '**/*.vtt']\n    summaries: ['**/*summar*.md', '**/*summar*.txt']\n    metadata: ['**/*metadata*.json', '**/*metadata*.csv']\n",
}

SCORE_ITEMS = [
    "Saves the reader future work",
    "Contains concrete proof",
    "Provides reusable takeaway: template, checklist, framework, workflow, or mental model",
    "Has a specific reader and job-to-be-done",
    "Can be applied without the author present",
    "Has a strong screenshot, visual, example, or artifact opportunity",
]

BANNED_PATTERNS = [
    r"groundbreaking",
    r"game-changing",
    r"pivotal moment",
    r"testament to",
    r"experts believe",
    r"studies show",
    r"the system compounds",
    r"the data tells us",
    r"the question is whether",
    r"\bNo [^.]+\. No [^.]+\. No [^.]+\.",
    r"actually",
    r"literally",
    r"quietly",
    r"unlock (?:the )?power",
    r"delve",
    r"in today's (?:fast-paced|digital)",
    r"revolutionary",
    r"best",
    r"ultimate",
    r"unparalleled",
]


def resolve_root(root: str | Path | None = None) -> Path:
    if root:
        return Path(root).expanduser().resolve()
    cwd = Path.cwd().resolve()
    if (cwd / "config.yaml").exists() and (cwd / "runs").exists():
        return cwd
    return cwd / "content-os"


def slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug[:64].strip("-") or "untitled"


def run_id_for(title: str) -> str:
    return f"{datetime.now(timezone.utc):%Y-%m}-{slugify(title)}"


def init(root: Path, force: bool = False) -> list[Path]:
    created: list[Path] = []
    for d in STRUCTURE_DIRS:
        (root / d).mkdir(parents=True, exist_ok=True)
    for rel, text in TEMPLATES.items():
        if write_if_missing(root / rel, text, force=force):
            created.append(root / rel)
    example = root / "runs" / "archive" / "example-bookmarkable-content-system"
    example.mkdir(parents=True, exist_ok=True)
    write_if_missing(
        example / "content-object.yaml",
        dump_yaml(
            ContentObject(
                id="example-bookmarkable-content-system",
                title="Example: bookmarkable content system",
                slug="example-bookmarkable-content-system",
                route="ORIGINAL",
                state="archived",
                files={"idea": "idea.md"},
            ).model_dump()
        ),
        force=force,
    )
    write_if_missing(
        example / "idea.md",
        "# Idea\n\nA local-first workflow for turning ideas into approval-gated content.\n",
        force=force,
    )
    return created


def load_config(root: Path) -> dict[str, Any]:
    cfg = load_yaml(root / "config.yaml")
    return cfg if isinstance(cfg, dict) else {}


def run_dir(root: Path, run_id: str) -> Path:
    for base in (root / "runs" / "active", root / "runs" / "archive"):
        p = base / run_id
        if p.exists():
            return p
    return root / "runs" / "active" / run_id


def load_object(path: Path) -> ContentObject:
    return ContentObject(**load_yaml(path / "content-object.yaml"))


def save_object(path: Path, obj: ContentObject) -> None:
    obj.updated_at = now_iso()
    atomic_write(path / "content-object.yaml", dump_yaml(obj.model_dump()), backup=True)


def create_run(
    root: Path,
    title: str,
    route: str = "auto",
    fmt: str = "x_post",
    source: str | None = None,
    text: str = "",
) -> Path:
    rid = run_id_for(title)
    path = root / "runs" / "active" / rid
    i = 2
    while path.exists():
        path = root / "runs" / "active" / f"{rid}-{i}"
        i += 1
    path.mkdir(parents=True)
    obj = ContentObject(
        id=path.name,
        title=title,
        slug=path.name,
        route=route,
        format=fmt,
        state=State.idea_review.value,
        source=source,
        files={
            "idea": "idea.md",
            "brief": "brief.md",
            "draft": "draft-package.md",
            "verification": "verification.md",
            "review": "review.md",
            "scheduler_handoff": "scheduler-handoff.md",
            "feedback": "feedback.md",
        },
    )
    atomic_write(path / "content-object.yaml", dump_yaml(obj.model_dump()))
    atomic_write(
        path / "idea.md",
        f"# Idea\n\nTitle: {title}\n\n{text or 'TODO: Describe the idea, source material, proof, and intended reader.'}\n",
    )
    return path


def adapter_rules_path(root: Path) -> Path:
    return root / "integrations" / "source-adapters.json"


def _adapter_rule_items(root: Path) -> tuple[list[Any], list[str]]:
    path = adapter_rules_path(root)
    if not path.exists():
        return [], []
    try:
        data = json.loads(read_text(path) or "[]")
    except json.JSONDecodeError as exc:
        return [], [f"Malformed adapter registry JSON: {path.relative_to(root)}: {exc}"]
    if not isinstance(data, list):
        return [], ["Adapter registry must be a JSON list of rule objects."]
    return data, []


def validate_adapter_rules(root: Path) -> list[str]:
    data, issues = _adapter_rule_items(root)
    seen_names: set[str] = set()
    for index, item in enumerate(data):
        prefix = f"Adapter rule #{index + 1}"
        if not isinstance(item, dict):
            issues.append(f"{prefix} must be an object.")
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            issues.append(f"{prefix} is missing name.")
        elif name in seen_names:
            issues.append(f"{prefix} duplicates adapter rule name: {name}")
        seen_names.add(name)
        extensions = item.get("extensions")
        if not isinstance(extensions, list) or not extensions:
            issues.append(f"{prefix} must declare a non-empty extensions list.")
        else:
            bad_ext = [
                ext
                for ext in extensions
                if not isinstance(ext, str) or not ext.startswith(".")
            ]
            if bad_ext:
                issues.append(f"{prefix} has invalid extensions: {bad_ext}")
        if not str(item.get("source_type") or "").strip():
            issues.append(f"{prefix} is missing source_type.")
        route = str(item.get("route") or "REPURPOSE")
        if route not in VALID_ROUTES:
            issues.append(f"{prefix} has invalid route: {route}")
        fmt = str(item.get("format") or "custom")
        if fmt not in VALID_FORMATS:
            issues.append(f"{prefix} has invalid format: {fmt}")
        for key in ("text_keys", "title_keys"):
            if key in item and not isinstance(item[key], list):
                issues.append(f"{prefix} {key} must be a list when provided.")
    return issues


def load_adapter_rules(root: Path) -> list[AdapterRule]:
    data, _issues = _adapter_rule_items(root)
    rules: list[AdapterRule] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        extensions = {
            str(ext).lower()
            for ext in item.get("extensions", [])
            if isinstance(ext, str) and ext.startswith(".")
        }
        if (
            not extensions
            or str(item.get("route") or "REPURPOSE") not in VALID_ROUTES
            or str(item.get("format") or "custom") not in VALID_FORMATS
        ):
            continue
        rules.append(
            AdapterRule(
                name=str(item.get("name") or "custom-rule"),
                extensions=extensions,
                source_type=str(item.get("source_type") or "custom_source"),
                text_keys=(
                    [str(k) for k in item.get("text_keys", [])]
                    if isinstance(item.get("text_keys", []), list)
                    else []
                ),
                title_keys=(
                    [str(k) for k in item.get("title_keys", [])]
                    if isinstance(item.get("title_keys", []), list)
                    else []
                ),
                route=str(item.get("route") or "REPURPOSE"),
                format=str(item.get("format") or "custom"),
                treat_as_text=bool(item.get("treat_as_text", True)),
            )
        )
    return rules


def adapter_inventory(root: Path) -> list[dict[str, Any]]:
    builtins = [
        {"name": "json", "kind": "built-in", "extensions": [".json", ".jsonl"]},
        {"name": "csv", "kind": "built-in", "extensions": [".csv"]},
        {
            "name": "text",
            "kind": "built-in",
            "extensions": sorted([".txt", ".md", ".markdown", ".log", ".srt", ".vtt"]),
        },
        {"name": "pdf", "kind": "built-in", "extensions": [".pdf"]},
        {
            "name": "media_manifest",
            "kind": "built-in",
            "extensions": sorted(
                [
                    ".mp3",
                    ".wav",
                    ".m4a",
                    ".flac",
                    ".mp4",
                    ".mov",
                    ".mkv",
                    ".avi",
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".webp",
                ]
            ),
        },
    ]
    custom = [
        {
            "name": rule.name,
            "kind": "rule",
            "extensions": sorted(rule.extensions),
            "source_type": rule.source_type,
            "route": rule.route,
            "format": rule.format,
        }
        for rule in load_adapter_rules(root)
    ]
    return custom + builtins


def source_manifest_path(root: Path) -> Path:
    return root / "stores" / "proof" / "source-manifest.json"


def record_source_manifest(
    root: Path, source: SourceDocument, proof_path: Path | None, run_path: Path | None
) -> None:
    manifest_path = source_manifest_path(root)
    existing = load_yaml(manifest_path)
    records = existing.get("sources", []) if isinstance(existing, dict) else []
    records = [
        record for record in records if record.get("source_id") != source.source_id
    ]
    records.append(
        {
            "source_id": source.source_id,
            "sha256": source.sha256,
            "path": str(source.path),
            "title": source.title,
            "source_type": source.source_type,
            "adapter": source.adapter,
            "proof_path": str(proof_path.relative_to(root)) if proof_path else None,
            "run_id": run_path.name if run_path else None,
            "imported_at": now_iso(),
            "warnings": source.warnings,
        }
    )
    atomic_write(
        manifest_path,
        json.dumps({"sources": records}, indent=2) + "\n",
        backup=manifest_path.exists(),
    )


def proof_path_for_source(root: Path, source: SourceDocument) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    base = root / "stores" / "proof" / f"{stamp}-{slugify(source.title)}.md"
    if not base.exists():
        return base
    suffix = 2
    while True:
        candidate = base.with_name(f"{base.stem}-{suffix}{base.suffix}")
        if not candidate.exists():
            return candidate
        suffix += 1


def store_source_as_proof(root: Path, source: SourceDocument) -> Path:
    proof_path = proof_path_for_source(root, source)
    atomic_write(proof_path, source.to_markdown(), backup=False)
    return proof_path


def import_source(
    root: Path,
    source_path: Path,
    title: str | None = None,
    route: str = "auto",
    fmt: str = "custom",
    as_proof: bool = True,
    create_run_folder: bool = True,
) -> tuple[Path | None, Path | None, SourceDocument]:
    source = extract_source(source_path, rules=load_adapter_rules(root))
    if title:
        source.title = title
    proof_path = store_source_as_proof(root, source) if as_proof else None
    selected_route = source.suggested_route if route == "auto" else route
    idea_text = f"""Imported local source: `{source.path}`

Detected source type: {source.source_type}
Suggested route: {source.suggested_route}

Use only the extracted proof below, the proof store entry, and other user-approved context. Do not invent missing details.

## Extracted source
{source.text[:12000]}
"""
    if proof_path:
        idea_text += f"\n\nProof store entry: `{proof_path.relative_to(root)}`\n"
    run_path = None
    if create_run_folder:
        run_path = create_run(
            root,
            source.title,
            route=selected_route,
            fmt=fmt,
            source=str(source.path),
            text=idea_text,
        )
        obj = load_object(run_path)
        obj.attribution = {
            "source_id": source.source_id,
            "source_path": str(source.path),
            "source_type": source.source_type,
            "sha256": source.sha256,
            "adapter": source.adapter,
            "imported_at": now_iso(),
        }
        if proof_path:
            obj.files["imported_proof"] = str(proof_path.relative_to(root))
        obj.assumptions.extend(source.warnings)
        obj.next_action = (
            "Review imported source, attribution, and route before briefing."
        )
        save_object(run_path, obj)
    if proof_path or run_path:
        record_source_manifest(root, source, proof_path, run_path)
    return run_path, proof_path, source


def scan_inputs(
    root: Path, path: Path, out: Path | None = None, limit: int | None = None
) -> tuple[list[SourceDocument], Path | None]:
    cfg = load_config(root).get("input_ingest", {})
    max_files = int(cfg.get("max_scan_files", 100)) if isinstance(cfg, dict) else 100
    docs = scan_path(path, limit=limit or max_files, rules=load_adapter_rules(root))
    report = scan_report(path, docs)
    report_path = out
    if report_path:
        atomic_write(report_path, report, backup=report_path.exists())
    return docs, report_path


def capture(
    root: Path,
    title: str,
    text: str,
    source: str | None,
    route: str,
    create_run_folder: bool = True,
) -> Path | None:
    ts = now_iso()
    rid = run_id_for(title)
    entry = f"\n## {title}\n\n- id: {rid}\n- captured_at: {ts}\n- source: {source or 'manual'}\n- route_hint: {route}\n- status: captured\n\n{text}\n"
    append(root / "stores" / "inbox.md", entry)
    if not create_run_folder:
        return None
    path = create_run(root, title, route=route, source=source, text=text)
    obj = load_object(path)
    obj.state = State.captured.value
    obj.next_action = "Move to idea_review with new-run or edit content-object.yaml."
    save_object(path, obj)
    return path


def classify_route(text: str) -> tuple[str, str]:
    t = text.lower()
    if any(w in t for w in ["research", "explore", "ideas about", "candidate ideas"]):
        return "RESEARCH_IDEATE", "Heuristic saw research/ideation language."
    if any(w in t for w in ["rewrite", "external", "source:", "attribution", "link:"]):
        return "REWRITE", "Heuristic saw external source or rewrite language."
    if any(
        w in t
        for w in [
            "repurpose",
            "transcript",
            "article",
            "newsletter",
            "old post",
            "convert",
        ]
    ):
        return (
            "REPURPOSE",
            "Heuristic saw user-owned source material conversion language.",
        )
    return (
        "ORIGINAL",
        "Heuristic defaulted to original user viewpoint. Edit content-object.yaml if wrong.",
    )


def route_run(root: Path, rid: str) -> str:
    path = run_dir(root, rid)
    obj = load_object(path)
    idea = read_text(path / "idea.md")
    if obj.route == "auto":
        obj.route, reason = classify_route(idea)
        if not llm_configured():
            reason += " No LLM is configured, so deterministic heuristics were used; edit content-object.yaml manually if needed."
    else:
        reason = "Route was already set manually."
    if obj.route == "REWRITE" and not obj.attribution:
        obj.assumptions.append(
            "REWRITE route needs attribution metadata before approval."
        )
    if obj.route == "RESEARCH_IDEATE":
        ideas_path = root / "stores" / "ideas" / f"{rid}.md"
        atomic_write(
            ideas_path,
            f"# Candidate ideas from {obj.title}\n\nSource run: {rid}\n\n{idea}\n",
            backup=ideas_path.exists(),
        )
        obj.next_action = (
            "Review candidate ideas in stores/ideas before creating a final-post run."
        )
    else:
        obj.next_action = "Create writer brief."
    save_object(path, obj)
    return reason


def context_snippet(root: Path, rel: str, limit: int = 1200) -> str:
    return read_text(root / rel).strip()[:limit]


def make_brief(root: Path, rid: str) -> Path:
    path = run_dir(root, rid)
    obj = load_object(path)
    idea = read_text(path / "idea.md")
    proof_files = sorted((root / "stores" / "proof").glob("*.md"))[:5]
    hooks_files = sorted((root / "stores" / "hooks").glob("*.md"))[:5]
    proof = (
        "\n".join(f"## {p.name}\n{read_text(p)[:800]}" for p in proof_files)
        or "No proof files supplied yet."
    )
    hooks = (
        "\n".join(f"- {read_text(p)[:160].strip()}" for p in hooks_files)
        or "No hook files supplied yet."
    )
    body = f"""# Writer Context Packet: {obj.title}

## thesis
TODO: One sentence the post must prove. Start from the idea below; do not invent facts.

## reader
{context_snippet(root, 'strategy/audience.md')}

## proof
Allowed proof only. Do not invent facts, metrics, clients, screenshots, or examples.

{proof}

## angle
TODO: Unexpected framing based on the idea and positioning.

## constraints
- format: {obj.format}
- route: {obj.route}
- banned phrases and slop checks: see voice/master-avoid-slop.md
- attribution required for REWRITE: {'yes' if obj.route == 'REWRITE' else 'no unless a source is used'}

## voice anchors
{context_snippet(root, 'voice/voice-profile.md')}

## risks
{context_snippet(root, 'voice/master-avoid-slop.md')}

## open loops
- Confirm thesis, proof, attribution, and any missing source details.
- LLM-disabled drafts must remain placeholders until a human fills them.

## source idea
{idea}

## positioning and pillars
{context_snippet(root, 'strategy/positioning.md')}

{context_snippet(root, 'strategy/pillars.md')}

## hook references
{hooks}
"""
    atomic_write(path / "brief.md", body, backup=True)
    obj.transition(State.brief_ready.value, force=True)
    obj.next_action = "Draft from brief."
    save_object(path, obj)
    return path / "brief.md"


def llm_configured() -> bool:
    return bool(
        os.getenv("CONTENT_OS_LLM_BASE_URL") and os.getenv("CONTENT_OS_LLM_MODEL")
    )


def call_llm(prompt: str) -> str:
    base = os.getenv("CONTENT_OS_LLM_BASE_URL", "").rstrip("/")
    key = os.getenv("CONTENT_OS_LLM_API_KEY", "")
    payload = {
        "model": os.getenv("CONTENT_OS_LLM_MODEL"),
        "temperature": float(os.getenv("CONTENT_OS_LLM_TEMPERATURE", "0.4")),
        "max_tokens": int(os.getenv("CONTENT_OS_LLM_MAX_TOKENS", "1200")),
        "messages": [
            {
                "role": "system",
                "content": "You draft approval-gated creator content. Do not invent proof.",
            },
            {"role": "user", "content": prompt},
        ],
    }
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode())
    return data["choices"][0]["message"]["content"]


def draft(root: Path, rid: str) -> Path:
    path = run_dir(root, rid)
    obj = load_object(path)
    brief = read_text(path / "brief.md")
    if llm_configured():
        candidate = call_llm(
            f"Create draft-package.md with required sections from this brief:\n\n{brief}"
        )
    else:
        candidate = f"""# Draft Package: {obj.title}

## final draft candidate
TODO: Write the final {obj.format} after reviewing the brief. No LLM is configured, so this placeholder avoids inventing proof.

## alternate hooks
- TODO: Hook tied to the reader's job-to-be-done.
- TODO: Hook using a concrete artifact or proof point.
- TODO: Hook that frames the cost of the current workflow.

## source/proof checklist
- [ ] Every claim maps to proof in brief.md or user knowledge.
- [ ] Attribution is present for external signals.
- [ ] No invented metrics, clients, screenshots, or outcomes.

## known assumptions
- Draft copy is incomplete until a human fills TODOs.

## suggested visuals
- Screenshot, checklist, workflow diagram, or annotated artifact from approved proof.

## platform format notes
- Target format: {obj.format}
- Keep the opening specific and useful.
"""
    atomic_write(
        path / "draft-package.md",
        candidate if candidate.startswith("#") else "# Draft Package\n\n" + candidate,
        backup=True,
    )
    obj.transition(State.drafting.value, force=True)
    obj.next_action = "Run verification."
    save_object(path, obj)
    return path / "draft-package.md"


def avoid_slop(text: str) -> list[str]:
    issues = []
    for pat in BANNED_PATTERNS:
        if re.search(pat, text, flags=re.IGNORECASE):
            issues.append(pat)
    if text.count("—") > 3:
        issues.append("em dash overuse")
    return issues


def score_draft(text: str) -> list[tuple[str, int, str]]:
    lower = text.lower()
    heuristics = [
        any(w in lower for w in ["save", "checklist", "workflow", "template", "reuse"]),
        any(
            w in lower
            for w in ["proof", "example", "screenshot", "number", "metric", "case"]
        ),
        any(
            w in lower
            for w in ["template", "checklist", "framework", "workflow", "mental model"]
        ),
        any(
            w in lower
            for w in ["reader", "founder", "operator", "creator", "job-to-be-done"]
        ),
        any(w in lower for w in ["step", "apply", "copy", "use this", "without"]),
        any(
            w in lower
            for w in ["visual", "screenshot", "artifact", "diagram", "example"]
        ),
    ]
    rows = []
    for item, ok in zip(SCORE_ITEMS, heuristics):
        score = 2 if ok else 0
        rows.append(
            (
                item,
                score,
                "Heuristic evidence found." if ok else "Missing or too vague.",
            )
        )
    return rows


def verify(root: Path, rid: str) -> Path:
    path = run_dir(root, rid)
    obj = load_object(path)
    text = read_text(path / "draft-package.md")
    rows = score_draft(text)
    total = sum(r[1] for r in rows)
    threshold = int(load_config(root).get("score_threshold", DEFAULT_SCORE_THRESHOLD))
    banned = avoid_slop(text)
    passed = total >= threshold and not banned and "TODO" not in text
    md = [
        f"# Verification: {obj.title}",
        "",
        "## score breakdown",
        "",
        "| Item | Score | Notes |",
        "|---|---:|---|",
    ]
    md += [f"| {item} | {score} | {note} |" for item, score, note in rows]
    md += [
        "",
        f"## total\n{total}/12",
        "",
        f"## pass/fail\n{'PASS' if passed else 'FAIL'} (threshold: {threshold}/12)",
        "",
        "## exact issues",
    ]
    if not passed:
        if total < threshold:
            md.append(f"- Score below threshold by {threshold-total} point(s).")
        if "TODO" in text:
            md.append("- Draft still contains TODO placeholders.")
        for b in banned:
            md.append(f"- Banned phrase/pattern matched: `{b}`")
    else:
        md.append("- None detected by deterministic checks.")
    md += [
        "",
        "## suggested fixes",
        "- Add concrete proof and reusable artifacts where scores are low.",
        "- Remove banned phrases and unsupported claims.",
        "",
        "## banned phrase matches",
    ]
    md += [f"- `{b}`" for b in banned] or ["- None"]
    md += [
        "",
        "## required human review note",
        "Human approval is required before any scheduler handoff or Postiz export can be used for publishing.",
    ]
    atomic_write(path / "verification.md", "\n".join(md) + "\n", backup=True)
    obj.score = total
    obj.transition(State.verification.value, force=True)
    obj.next_action = "Human review and approval."
    save_object(path, obj)
    return path / "verification.md"


def verification_passed(path: Path) -> bool:
    return "## pass/fail\nPASS" in read_text(path / "verification.md")


def approve(root: Path, rid: str, force: bool = False, reason: str = "") -> None:
    path = run_dir(root, rid)
    obj = load_object(path)
    passed = verification_passed(path)
    if not passed and not force:
        raise ValueError(
            "Verification failed; use --force with --reason to approve anyway."
        )
    if not passed and force and not reason.strip():
        raise ValueError("Forced approval of failed verification requires --reason.")
    obj.transition(State.human_review.value, force=True)
    obj.transition(State.approved.value)
    obj.human_approved = True
    obj.next_action = "Create scheduler handoff."
    append(
        path / "review.md",
        f"\n## Approval {now_iso()}\n- forced: {force}\n- reason: {reason or 'Verification passed'}\n",
    )
    save_object(path, obj)


def make_payload(obj: ContentObject, draft_text: str, platform: str) -> dict[str, Any]:
    return {
        "requires_manual_review": True,
        "approval_status": "approved" if obj.human_approved else "not_approved",
        "platform": platform,
        "title": obj.title,
        "content": draft_text,
        "attribution": obj.attribution,
        "utm": {"campaign": obj.slug, "source": platform, "medium": "social"},
        "postiz_note": "Create a draft/queue item only; do not auto-publish.",
    }


def scheduler_handoff(root: Path, rid: str, platform: str) -> Path:
    path = run_dir(root, rid)
    obj = load_object(path)
    if not obj.human_approved or obj.state not in {
        State.approved.value,
        State.scheduler_ready.value,
    }:
        raise ValueError("Scheduler handoff requires approved run.")
    draft_text = read_text(path / "draft-package.md")
    payload = make_payload(obj, draft_text, platform)
    md = f"""# Scheduler Handoff: {obj.title}

## final approved copy
{draft_text}

## platform
{platform}

## suggested publish window
{load_config(root).get('default_publish_window', 'Manual choice required')}

## media/assets needed
TODO: Attach approved assets manually.

## attribution
```json
{json.dumps(obj.attribution, indent=2)}
```

## UTM/campaign fields
- campaign: {obj.slug}
- source: {platform}
- medium: social

## Postiz-ready payload JSON
```json
{json.dumps(payload, indent=2)}
```
"""
    atomic_write(path / "scheduler-handoff.md", md, backup=True)
    if obj.state != State.scheduler_ready.value:
        obj.transition(State.scheduler_ready.value)
    obj.next_action = "Export payload or schedule manually."
    save_object(path, obj)
    return path / "scheduler-handoff.md"


def postiz_export(root: Path, rid: str, out: Path, platform: str = "custom") -> None:
    path = run_dir(root, rid)
    obj = load_object(path)
    if not obj.human_approved:
        raise ValueError("Postiz export requires approved run.")
    payload = make_payload(obj, read_text(path / "draft-package.md"), platform)
    atomic_write(out, json.dumps(payload, indent=2) + "\n", backup=out.exists())


def feedback(
    root: Path,
    rid: str,
    views: int,
    bookmarks: int,
    likes: int,
    replies: int,
    reposts: int,
    notes: str = "",
) -> dict[str, float]:
    path = run_dir(root, rid)
    obj = load_object(path)
    br = bookmarks / views if views else 0
    rr = replies / views if views else 0
    er = (likes + replies + reposts + bookmarks) / views if views else 0
    md = f"# Feedback: {obj.title}\n\n- views: {views}\n- bookmarks: {bookmarks}\n- likes: {likes}\n- replies: {replies}\n- reposts: {reposts}\n- bookmark_rate: {br:.6f}\n- reply_rate: {rr:.6f}\n- engagement_rate: {er:.6f}\n\n## qualitative notes\n{notes}\n\n## proposed lessons\n- Review hooks, proof density, and avoid-slop matches. Proposed voice changes must be manually copied into voice rules.\n"
    atomic_write(path / "feedback.md", md, backup=True)
    cfg = load_config(root)
    dest_base = (
        root
        / "stores"
        / (
            "winners"
            if br >= float(cfg.get("winning_bookmark_rate", WIN_BOOKMARK_RATE))
            else (
                "losers"
                if br <= float(cfg.get("losing_bookmark_rate", LOSE_BOOKMARK_RATE))
                else "feedback"
            )
        )
    )
    atomic_write(dest_base / f"{rid}.md", md, backup=True)
    obj.next_action = "Archive after lessons are reviewed."
    save_object(path, obj)
    return {"bookmark_rate": br, "reply_rate": rr, "engagement_rate": er}


def archive(root: Path, rid: str, force: bool = False) -> Path:
    src = root / "runs" / "active" / rid
    obj = load_object(src)
    if (
        obj.state
        not in [
            State.approved.value,
            State.published.value,
            State.learned.value,
            State.scheduler_ready.value,
        ]
        and not force
    ):
        raise ValueError("Refusing to archive unfinished work without --force.")
    if force:
        append(
            src / "review.md",
            f"\n## Forced archive {now_iso()}\n- previous_state: {obj.state}\n",
        )
    dest = root / "runs" / "archive" / rid
    if dest.exists():
        raise ValueError("Archive destination already exists.")
    obj.transition(State.archived.value, force=True)
    save_object(src, obj)
    shutil.move(str(src), str(dest))
    return dest


def status(root: Path) -> list[ContentObject]:
    return [
        load_object(p)
        for p in sorted((root / "runs" / "active").glob("*/"))
        if (p / "content-object.yaml").exists()
    ]


def doctor(root: Path) -> list[str]:
    issues = []
    for d in STRUCTURE_DIRS:
        if not (root / d).is_dir():
            issues.append(f"Missing directory: {d}")
    for rel in TEMPLATES:
        if not (root / rel).exists():
            issues.append(f"Missing file: {rel}")
    for y in list(root.glob("**/*.yaml")) + list(root.glob("**/*.yml")):
        try:
            load_yaml(y)
        except Exception as e:
            issues.append(f"Malformed YAML: {y.relative_to(root)}: {e}")
    issues.extend(validate_adapter_rules(root))
    if not llm_configured():
        issues.append("LLM not configured; placeholder drafting mode is active.")
    return issues
