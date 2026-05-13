from __future__ import annotations

import csv
import importlib.util
import json
import re
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

MAX_TEXT_BYTES = 250_000
TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".log", ".srt", ".vtt"}
MEDIA_EXTENSIONS = {
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
}
IGNORED_PARTS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "build",
    "dist",
    ".venv",
    "venv",
}


@dataclass
class SourceDocument:
    path: Path
    title: str
    source_type: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    suggested_route: str = "REPURPOSE"
    suggested_format: str = "custom"
    warnings: list[str] = field(default_factory=list)
    adapter: str = "unknown"
    sha256: str = ""

    @property
    def source_id(self) -> str:
        digest = self.sha256[:12] if self.sha256 else "unhashed"
        return f"{self.source_type}:{digest}"

    def to_markdown(self) -> str:
        warnings = "\n".join(f"- {w}" for w in self.warnings) or "- None"
        metadata = json.dumps(self.metadata, indent=2, sort_keys=True)
        return f"""# Source: {self.title}

- source_id: `{self.source_id}`
- path: `{self.path}`
- type: {self.source_type}
- adapter: {self.adapter}
- sha256: `{self.sha256 or 'not_computed'}`
- suggested_route: {self.suggested_route}
- suggested_format: {self.suggested_format}

## warnings
{warnings}

## metadata
```json
{metadata}
```

## extracted text
{self.text}
"""


@dataclass
class AdapterRule:
    name: str
    extensions: set[str]
    source_type: str
    text_keys: list[str] = field(default_factory=list)
    title_keys: list[str] = field(default_factory=list)
    route: str = "REPURPOSE"
    format: str = "custom"
    treat_as_text: bool = True


class SourceAdapter(Protocol):
    name: str

    def can_handle(self, path: Path) -> bool: ...

    def extract(self, path: Path) -> SourceDocument: ...


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_limited(path: Path, limit: int = MAX_TEXT_BYTES) -> tuple[str, bool]:
    data = path.read_bytes()[: limit + 1]
    truncated = len(data) > limit
    if truncated:
        data = data[:limit]
    text = data.decode("utf-8", errors="replace")
    return text, truncated


def title_from_path(path: Path) -> str:
    return re.sub(r"[_\-]+", " ", path.stem).strip().title() or path.name


def strip_subtitle_noise(text: str) -> str:
    lines = []
    for line in text.splitlines():
        clean = line.strip()
        if not clean or clean.isdigit() or "-->" in clean or clean.upper() == "WEBVTT":
            continue
        lines.append(clean)
    return "\n".join(lines)


def _segment_text(data: Any) -> str:
    if isinstance(data, dict):
        if isinstance(data.get("segments"), list):
            parts = []
            for segment in data["segments"]:
                if isinstance(segment, dict) and segment.get("text"):
                    start = segment.get("start")
                    prefix = f"[{start}] " if start is not None else ""
                    parts.append(f"{prefix}{segment['text']}")
            return "\n".join(parts)
        for key in ["transcript", "text", "summary", "content", "description"]:
            if isinstance(data.get(key), str):
                return data[key]
    if isinstance(data, list):
        parts = []
        for item in data:
            if isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return ""


def value_from_keys(data: Any, keys: list[str]) -> str:
    if not isinstance(data, dict):
        return ""
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


class RuleBasedAdapter:
    def __init__(self, rule: AdapterRule):
        self.rule = rule
        self.name = f"rule:{rule.name}"

    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() in self.rule.extensions

    def extract(self, path: Path) -> SourceDocument:
        raw, truncated = read_limited(path)
        warnings = ["File was truncated for local ingest."] if truncated else []
        text = raw
        title = title_from_path(path)
        metadata: dict[str, Any] = {
            "extension": path.suffix.lower(),
            "bytes": path.stat().st_size,
            "rule": self.rule.name,
        }
        if self.rule.text_keys and (
            path.suffix.lower() == ".jsonl" or raw.lstrip().startswith(("{", "["))
        ):
            if path.suffix.lower() == ".jsonl":
                rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
                text = "\n".join(
                    value_from_keys(row, self.rule.text_keys)
                    for row in rows
                    if value_from_keys(row, self.rule.text_keys)
                )
                metadata["records"] = len(rows)
                if rows:
                    title = value_from_keys(rows[0], self.rule.title_keys) or title
            else:
                data = json.loads(raw)
                text = value_from_keys(data, self.rule.text_keys) or _segment_text(data)
                title = value_from_keys(data, self.rule.title_keys) or title
                metadata["json_shape"] = "object" if isinstance(data, dict) else "list"
            if not text:
                warnings.append(
                    "Rule did not find configured text keys; preserved raw text excerpt."
                )
                text = raw[:MAX_TEXT_BYTES]
        return SourceDocument(
            path=path,
            title=title,
            source_type=self.rule.source_type,
            text=text,
            metadata=metadata,
            suggested_route=self.rule.route,
            suggested_format=self.rule.format,
            warnings=warnings,
            adapter=self.name,
            sha256=file_sha256(path),
        )


class JsonAdapter:
    name = "json"

    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() in {".json", ".jsonl"}

    def extract(self, path: Path) -> SourceDocument:
        raw, truncated = read_limited(path)
        warnings = ["File was truncated for local ingest."] if truncated else []
        if path.suffix.lower() == ".jsonl":
            rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
            text = _segment_text(rows)
            metadata: dict[str, Any] = {"records": len(rows)}
        else:
            data = json.loads(raw)
            text = _segment_text(data)
            metadata = (
                data
                if isinstance(data, dict)
                else {"records": len(data) if isinstance(data, list) else 1}
            )
        source_type = (
            "transcript_json" if "segment" in raw[:2000].lower() else "metadata_json"
        )
        if not text:
            text = json.dumps(metadata, indent=2, sort_keys=True)[:MAX_TEXT_BYTES]
            warnings.append(
                "No transcript-like text field found; preserved metadata excerpt instead."
            )
        return SourceDocument(
            path=path,
            title=str(metadata.get("title") or title_from_path(path)),
            source_type=source_type,
            text=text,
            metadata=(
                {k: v for k, v in metadata.items() if k != "segments"}
                if isinstance(metadata, dict)
                else metadata
            ),
            suggested_route="REPURPOSE",
            warnings=warnings,
            adapter=self.name,
            sha256=file_sha256(path),
        )


class CsvAdapter:
    name = "csv"

    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() == ".csv"

    def extract(self, path: Path) -> SourceDocument:
        raw, truncated = read_limited(path)
        reader = csv.DictReader(raw.splitlines())
        rows = list(reader)
        sample = rows[:10]
        columns = reader.fieldnames or []
        text_fields = [
            c
            for c in columns
            if c
            and c.lower()
            in {"text", "transcript", "summary", "content", "note", "notes"}
        ]
        if text_fields:
            extracted = "\n".join(
                str(row.get(text_fields[0], ""))
                for row in rows
                if row.get(text_fields[0])
            )
        else:
            extracted = "\n".join(json.dumps(row, sort_keys=True) for row in sample)
        warnings = ["File was truncated for local ingest."] if truncated else []
        return SourceDocument(
            path=path,
            title=title_from_path(path),
            source_type="tabular_export",
            text=extracted or "No rows found.",
            metadata={
                "columns": columns,
                "rows_read": len(rows),
                "sample_rows": sample,
            },
            warnings=warnings,
            adapter=self.name,
            sha256=file_sha256(path),
        )


class TextAdapter:
    name = "text"

    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() in TEXT_EXTENSIONS

    def extract(self, path: Path) -> SourceDocument:
        raw, truncated = read_limited(path)
        ext = path.suffix.lower()
        text = strip_subtitle_noise(raw) if ext in {".srt", ".vtt"} else raw
        lowered = path.name.lower() + "\n" + text[:1000].lower()
        if "summary" in lowered:
            source_type = "summary"
        elif "transcript" in lowered or ext in {".srt", ".vtt"}:
            source_type = "transcript"
        elif ext in {".md", ".markdown"}:
            source_type = "markdown_note"
        else:
            source_type = "text_note"
        warnings = ["File was truncated for local ingest."] if truncated else []
        return SourceDocument(
            path=path,
            title=title_from_path(path),
            source_type=source_type,
            text=text,
            metadata={"extension": ext, "bytes": path.stat().st_size},
            warnings=warnings,
            adapter=self.name,
            sha256=file_sha256(path),
        )


class PdfAdapter:
    name = "pdf"

    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() == ".pdf"

    def extract(self, path: Path) -> SourceDocument:
        warnings = []
        text = ""
        if importlib.util.find_spec("pypdf"):
            import pypdf  # type: ignore

            reader = pypdf.PdfReader(str(path))
            text = "\n".join(page.extract_text() or "" for page in reader.pages[:20])
        else:
            warnings.append(
                "pypdf is not installed; captured PDF as a file manifest only."
            )
        return SourceDocument(
            path=path,
            title=title_from_path(path),
            source_type="pdf_document",
            text=text
            or f"PDF source available at {path}. Extract text manually or install pypdf for local extraction.",
            metadata={"extension": ".pdf", "bytes": path.stat().st_size},
            warnings=warnings,
            adapter=self.name,
            sha256=file_sha256(path),
        )


class MediaManifestAdapter:
    name = "media_manifest"

    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() in MEDIA_EXTENSIONS

    def extract(self, path: Path) -> SourceDocument:
        return SourceDocument(
            path=path,
            title=title_from_path(path),
            source_type="media_asset",
            text=(
                f"Media asset captured as a manifest only: {path.name}. "
                "Use approved local transcription, metadata export, screenshots, or notes as proof before drafting."
            ),
            metadata={"extension": path.suffix.lower(), "bytes": path.stat().st_size},
            suggested_route="REPURPOSE",
            warnings=["Binary media is not transcribed automatically by Content OS."],
            adapter=self.name,
            sha256=file_sha256(path),
        )


ADAPTERS: list[SourceAdapter] = [
    JsonAdapter(),
    CsvAdapter(),
    TextAdapter(),
    PdfAdapter(),
    MediaManifestAdapter(),
]


def adapters_for_rules(rules: list[AdapterRule] | None = None) -> list[SourceAdapter]:
    rule_adapters = [RuleBasedAdapter(rule) for rule in (rules or [])]
    return rule_adapters + ADAPTERS


def detect_adapter(
    path: Path, rules: list[AdapterRule] | None = None
) -> SourceAdapter | None:
    for adapter in adapters_for_rules(rules):
        if adapter.can_handle(path):
            return adapter
    return None


def extract_source(
    path: Path, rules: list[AdapterRule] | None = None
) -> SourceDocument:
    adapter = detect_adapter(path, rules=rules)
    if adapter:
        return adapter.extract(path)
    raw, truncated = read_limited(path)
    if "\x00" in raw[:1000]:
        return SourceDocument(
            path=path,
            title=title_from_path(path),
            source_type="unknown_binary",
            text=f"Unsupported binary source captured as manifest only: {path.name}",
            metadata={"extension": path.suffix.lower(), "bytes": path.stat().st_size},
            warnings=[
                "Unsupported binary format; provide a manual export or approved API output."
            ],
            adapter="unknown_binary",
            sha256=file_sha256(path),
        )
    return SourceDocument(
        path=path,
        title=title_from_path(path),
        source_type="unknown_text",
        text=raw,
        metadata={"extension": path.suffix.lower(), "bytes": path.stat().st_size},
        warnings=["Unknown extension treated as text."]
        + (["File was truncated for local ingest."] if truncated else []),
        adapter="unknown_text",
        sha256=file_sha256(path),
    )


def iter_supported_files(base: Path, limit: int = 100) -> list[Path]:
    if base.is_file():
        return [base]
    files: list[Path] = []
    for path in sorted(base.rglob("*")):
        if len(files) >= limit:
            break
        if not path.is_file() or any(part in IGNORED_PARTS for part in path.parts):
            continue
        files.append(path)
    return files


def scan_path(
    base: Path, limit: int = 100, rules: list[AdapterRule] | None = None
) -> list[SourceDocument]:
    docs: list[SourceDocument] = []
    for path in iter_supported_files(base, limit=limit):
        try:
            docs.append(extract_source(path, rules=rules))
        except (
            OSError,
            UnicodeError,
            ValueError,
            json.JSONDecodeError,
            csv.Error,
        ) as exc:
            docs.append(
                SourceDocument(
                    path=path,
                    title=title_from_path(path),
                    source_type="unreadable",
                    text="",
                    metadata={"extension": path.suffix.lower()},
                    warnings=[f"Could not read source: {exc}"],
                    adapter="scan_error",
                )
            )
    return docs


def scan_report(base: Path, docs: list[SourceDocument]) -> str:
    rows = "\n".join(
        f"| `{doc.path}` | `{doc.source_id}` | {doc.adapter} | {doc.source_type} | {doc.suggested_route} | {len(doc.text)} | {', '.join(doc.warnings) or 'None'} |"
        for doc in docs
    )
    return f"""# Input Scan Report

Scanned: `{base}`

| Path | Source ID | Adapter | Detected type | Suggested route | Extracted chars | Warnings |
|---|---|---|---|---|---:|---|
{rows}

No external platforms were scraped. Files were read from local/user-provided inputs only.
"""
