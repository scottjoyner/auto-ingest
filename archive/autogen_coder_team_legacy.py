#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AutoGen Coder Team (Ollama + Neo4j vector index + Git logger + Sidecar replay)

Key features:
- Model routing across your local Ollama models (Gemma3, Qwen2.5 Coder, Qwen3, Reader-LM, Nomic-Embed)
- Neo4j memory with **native vector index** for :Document(embedding)
- **Offline-first sidecar** JSONL logs for everything (files/runs/docs/artifacts/commits/cypher)
- **Replay/ingest** sidecars to Neo4j later
- **Git logger** with auto init + auto-commits on write/patch
- Tools for agents:
  write_file / read_file / list_dir / run / neo4j_query / neo4j_write
  kb_ingest_text / kb_ingest_url / kb_similar_local / kb_vector_search_neo4j
  git(cmd) / apply_patch(unified_diff)
- CLI:
  - `run` (default): start multi-agent workflow
  - `ingest-sidecar`: replay sidecar JSONL into Neo4j
  - `cheatsheet`: print common commands

ENV:
  SANDBOX=./_coder_sandbox
  OLLAMA_CHAT_BASE=http://localhost:11434/v1
  OLLAMA_EMBED_BASE=http://localhost:11434
  NEO4J_URI=bolt://localhost:7687
  NEO4J_USER=neo4j
  NEO4J_PASSWORD=yourpassword
  NEO4J_DB=neo4j
"""

import os, sys, json, time, argparse, shutil, subprocess, textwrap, math, uuid
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import requests

# -------- Optional Neo4j driver (operate offline if missing/unreachable) -----
NEO4J_AVAILABLE = True
try:
    from neo4j import GraphDatabase
except Exception:
    NEO4J_AVAILABLE = False

# -------- AutoGen ------------------------------------------------------------
from autogen import AssistantAgent, UserProxyAgent
from autogen.agentchat.groupchat import GroupChat, GroupChatManager

# =========================
# Config
# =========================
SANDBOX = Path(os.getenv("SANDBOX", "./_coder_sandbox")).resolve()
SIDE_DIR = SANDBOX / "_sidecar"
SIDE_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_CHAT_BASE = os.getenv("OLLAMA_CHAT_BASE", "http://localhost:11434/v1")
OLLAMA_EMBED_BASE = os.getenv("OLLAMA_EMBED_BASE", "http://localhost:11434")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "changeme")
NEO4J_DB = os.getenv("NEO4J_DB", "neo4j")

# Models (editable)
MODELS = {
    "planner": "qwen3:4b",
    "architect": "qwen3:4b",
    "researcher": "gemma3:4b",
    "summarizer": "gemma3:1b",
    "coder": "qwen2.5-coder:7b",
    "reviewer": "qwen2.5-coder:1.5b",
    "tester": "qwen2.5-coder:1.5b",
    "builder": "qwen2.5-coder:1.5b",
    "reader": "reader-lm:1.5b",
    "embed": "nomic-embed-text:latest",
}

# Vector index naming
DOC_EMBED_INDEX = os.getenv("DOC_EMBED_INDEX", "doc_embedding_index")

# ================ Utilities: sidecar logging (offline-first) =================

def _sidecar_log(kind: str, payload: Dict[str, Any]) -> None:
    """Write an event to JSONL for later replay. One file per 'kind'."""
    f = SIDE_DIR / f"{kind}.jsonl"
    payload = {"ts": int(time.time()*1000), "kind": kind, **payload}
    with f.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(payload, ensure_ascii=False) + "\n")

def _sidecar_iter(kind: str):
    f = SIDE_DIR / f"{kind}.jsonl"
    if not f.exists():
        return
    with f.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line: continue
            try:
                yield json.loads(line)
            except Exception:
                continue

# =========================
# Git helpers
# =========================
def _git(cmd: str) -> str:
    """Run git inside SANDBOX; init repo if needed."""
    if not (SANDBOX / ".git").exists():
        subprocess.run("git init", shell=True, cwd=str(SANDBOX), check=False,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        # minimal identity to avoid global dependency
        subprocess.run('git config user.email "autogen@deathstar.local"', shell=True, cwd=str(SANDBOX))
        subprocess.run('git config user.name "AutoGen Bot"', shell=True, cwd=str(SANDBOX))
        _sidecar_log("git_init", {"cwd": str(SANDBOX)})
    p = subprocess.run(f"git {cmd}", shell=True, cwd=str(SANDBOX),
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = f"$ git {cmd}\n{p.stdout}"
    _sidecar_log("git", {"cmd": cmd, "output": out})
    return out

def git_auto_commit(message: str, add_paths: Optional[List[str]] = None) -> str:
    """Stage changed files and commit."""
    if add_paths:
        for ap in add_paths:
            _git(f'add "{ap}"')
    else:
        _git("add -A")

    # Escape only for shell safety
    safe_msg = message.replace('"', '\\"')
    res = _git(f'commit -m "{safe_msg}" || true')

    # log commit hash (if any)
    log = _git("rev-parse --short HEAD || true")
    _sidecar_log("commit", {"message": message, "head": log.strip()})
    return res + "\n" + log


# =========================
# Sandbox tools
# =========================
def _safe_path(rel: str) -> Path:
    p = (SANDBOX / rel).resolve()
    if SANDBOX not in p.parents and p != SANDBOX:
        raise ValueError(f"Path escapes sandbox: {p}")
    return p

def tool_write_file(path: str, content: str, mkparents: bool = True, mode: str = "w", commit: bool = True) -> str:
    p = _safe_path(path)
    if mkparents:
        p.parent.mkdir(parents=True, exist_ok=True)
    with p.open(mode, encoding="utf-8") as f:
        f.write(content)
    bytesz = p.stat().st_size if p.exists() else len(content)
    _sidecar_log("file_write", {"path": str(p), "bytes": bytesz})
    if commit:
        git_auto_commit(f"write_file: {path}", add_paths=[str(p.relative_to(SANDBOX))])
    # Mirror to Neo4j if online
    neo4j_write("""
        MERGE (f:File {path:$path})
        ON CREATE SET f.createdAt=timestamp()
        SET f.updatedAt=timestamp(), f.bytes=$bytes
    """, {"path": str(p), "bytes": bytesz})
    return f"WROTE {p}"

def tool_read_file(path: str, max_chars: int = 120_000) -> str:
    p = _safe_path(path)
    data = p.read_text(encoding="utf-8")
    return data if len(data) <= max_chars else (data[:max_chars] + f"\n[...TRUNCATED {len(data)-max_chars} chars]")

def tool_list_dir(path: str = ".") -> str:
    p = _safe_path(path)
    if not p.exists():
        return f"{p} (does not exist)"
    out = []
    for item in sorted(p.rglob("*")):
        rel = item.relative_to(SANDBOX)
        out.append(str(rel) + ("/" if item.is_dir() else ""))
    return "\n".join(out) if out else "(empty)"

def tool_run(cmd: str, timeout: int = 600, env: Dict[str, str] | None = None, workdir: str = ".") -> str:
    wd = _safe_path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    full_env = os.environ.copy()
    if env: full_env.update(env)
    try:
        p = subprocess.run(cmd, shell=True, cwd=str(wd), env=full_env,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout, text=True)
        out = f"$ {cmd}\n(exit={p.returncode})\n{p.stdout}"
    except subprocess.TimeoutExpired as e:
        out = f"$ {cmd}\n(TIMEOUT after {timeout}s)\n{e.output or ''}"
    _sidecar_log("run", {"cmd": cmd, "workdir": str(wd), "output": out[:200_000]})
    neo4j_write("CREATE (r:Run {cmd:$cmd, output:$out, ts:timestamp()})", {"cmd": cmd, "out": out[:200_000]})
    return out

def tool_git(cmd: str) -> str:
    return _git(cmd)

def tool_apply_patch(unified_diff: str, commit_message: str = "apply_patch") -> str:
    """Apply a unified diff using `git apply`."""
    patch_path = SANDBOX / f"_patch_{uuid.uuid4().hex}.diff"
    patch_path.write_text(unified_diff, encoding="utf-8")
    out = _git(f'apply "{patch_path.name}" || true')
    _sidecar_log("patch", {"file": str(patch_path), "output": out})
    # commit
    out += "\n" + git_auto_commit(commit_message)
    return out

# =========================
# Neo4j online/offline helpers
# =========================
_driver = None
def _neo4j_online() -> bool:
    global _driver
    if not NEO4J_AVAILABLE:
        return False
    try:
        if _driver is None:
            _driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with _driver.session(database=NEO4J_DB) as s:
            s.run("RETURN 1 AS ok").single()
        return True
    except Exception:
        return False

ONLINE = _neo4j_online()

def neo4j_write(cypher: str, params: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    """If offline, store Cypher to sidecar; if online, execute."""
    payload = {"cypher": cypher, "params": params or {}}
    if ONLINE:
        with _driver.session(database=NEO4J_DB) as s:
            res = s.run(cypher, params or {})
            data = [r.data() for r in res]
        return data
    else:
        _sidecar_log("cypher_write", payload)
        return []

def neo4j_query(cypher: str, params: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    if ONLINE:
        with _driver.session(database=NEO4J_DB) as s:
            res = s.run(cypher, params or {})
            return [r.data() for r in res]
    else:
        # Offline: log and return empty (agents still proceed using local files)
        _sidecar_log("cypher_query", {"cypher": cypher, "params": params or {}})
        return []

def _infer_dim(default_dim: int = 768) -> int:
    try:
        v = requests.post(f"{OLLAMA_EMBED_BASE}/api/embeddings",
                          json={"model": MODELS["embed"], "prompt": "dimension check"},
                          timeout=30).json().get("embedding", [])
        return len(v) or default_dim
    except Exception:
        return default_dim

def neo4j_bootstrap_schema(dim: Optional[int] = None) -> None:
    """Create helpful constraints & a vector index for :Document(embedding)."""
    if dim is None:
        dim = _infer_dim(768)
    stmts = [
        "CREATE INDEX IF NOT EXISTS FOR (p:Project) ON (p.id)",
        "CREATE INDEX IF NOT EXISTS FOR (t:Task) ON (t.id)",
        "CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.id)",
        "CREATE INDEX IF NOT EXISTS FOR (f:File) ON (f.path)",
        # Vector index (Neo4j 5)
        f"CREATE VECTOR INDEX IF NOT EXISTS {DOC_EMBED_INDEX} IF NOT EXISTS FOR (d:Document) ON (d.embedding) "
        f"OPTIONS {{ indexConfig: {{ 'vector.dimensions': {dim}, 'vector.similarity_function': 'cosine' }} }}",
    ]
    for c in stmts:
        try:
            neo4j_write(c)
        except Exception:
            pass

# =========================
# Knowledge base (Documents + embeddings)
# =========================
def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b): return 0.0
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a)); nb = math.sqrt(sum(y*y for y in b))
    return 0.0 if na==0 or nb==0 else dot/(na*nb)

def embed_text(text: str) -> List[float]:
    url = f"{OLLAMA_EMBED_BASE}/api/embeddings"
    r = requests.post(url, json={"model": MODELS["embed"], "prompt": text}, timeout=90)
    r.raise_for_status()
    return r.json()["embedding"]

def kb_ingest_text(title: str, text: str, tags: Optional[List[str]] = None) -> str:
    vec = embed_text(text[:8000])
    doc_id = f"doc_{int(time.time()*1000)}"
    params = {"id": doc_id, "title": title, "text": text, "tags": tags or [], "embedding": vec, "ts": int(time.time()*1000)}
    # Sidecar copy
    _sidecar_log("document", params)
    # Files snapshot
    tool_write_file(f"kb/markdown/{doc_id}.md", f"# {title}\n\n{text}\n", commit=False)
    # Neo4j
    neo4j_write("""
        MERGE (d:Document {id:$id})
        SET d.title=$title, d.text=$text, d.tags=$tags, d.embedding=$embedding, d.ts=$ts
    """, params)
    return f"INGESTED {doc_id} (dim={len(vec)})"

def _reader_convert_html_to_md(html: str) -> str:
    url = f"{OLLAMA_CHAT_BASE}/chat/completions"
    payload = {
        "model": MODELS["reader"],
        "messages": [
            {"role": "system", "content": "Convert the following HTML to clean, readable Markdown."},
            {"role": "user", "content": html}
        ],
        "temperature": 0.0,
    }
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def kb_ingest_url(url: str, tags: Optional[List[str]] = None) -> str:
    import urllib.request
    with urllib.request.urlopen(url, timeout=60) as resp:
        html = resp.read().decode("utf-8", errors="ignore")
    md = _reader_convert_html_to_md(html)
    tool_write_file(f"kb/raw/{Path(url).name or 'page'}.html", html, commit=False)
    tool_write_file(f"kb/markdown/{Path(url).name or 'page'}.md", md, commit=False)
    return kb_ingest_text(f"URL:{url}", md, tags=(tags or []) + ["url"])

def _load_sidecar_documents() -> List[Dict[str, Any]]:
    docs = []
    for rec in _sidecar_iter("document"):
        docs.append(rec)
    return docs

def kb_similar_local(query_text: str, top_k: int = 5) -> str:
    """Local cosine over available Documents (fallback). Uses Neo4j if online; else sidecar docs."""
    qv = embed_text(query_text[:8000])
    rows: List[Tuple[float, str, str]] = []
    if ONLINE:
        fetched = neo4j_query("MATCH (d:Document) RETURN d.id AS id, d.title AS title, d.embedding AS emb LIMIT 2000")
        for r in fetched:
            emb = r.get("emb") or r.get("d.embedding")
            if not emb: continue
            rows.append((_cosine(qv, emb), r["id"], r.get("title","")))
    else:
        for d in _load_sidecar_documents():
            emb = d.get("embedding")
            if not emb: continue
            rows.append((_cosine(qv, emb), d.get("id",""), d.get("title","")))
    rows.sort(reverse=True, key=lambda x: x[0])
    top = rows[:max(1, min(top_k, len(rows)))]
    return "\n".join([f"{i+1}. {doc_id} :: {title} :: score={score:.4f}" for i,(score,doc_id,title) in enumerate(top)]) or "(no matches)"

def kb_vector_search_neo4j(query_text: str, top_k: int = 5) -> str:
    """
    Deep vector search using Neo4j native vector index.
    Falls back to kb_similar_local if offline.
    """
    if not ONLINE:
        return "[offline] " + kb_similar_local(query_text, top_k)
    qv = embed_text(query_text[:8000])
    # Neo4j 5 vector query:
    cypher = (
        f"CALL db.index.vector.queryNodes('{DOC_EMBED_INDEX}', $k, $qv) "
        "YIELD node, score "
        "RETURN node.id AS id, node.title AS title, score "
        "ORDER BY score DESC LIMIT $k"
    )
    rows = neo4j_query(cypher, {"k": int(top_k), "qv": qv})
    if not rows:
        return "(no matches)"
    return "\n".join([f"{i+1}. {r['id']} :: {r.get('title','')} :: score={float(r['score']):.4f}" for i,r in enumerate(rows)])

# =========================
# Tool registration
# =========================
TOOLS_DOC = textwrap.dedent("""
You have these sandboxed tools:

- write_file(path, content, mkparents=True, mode="w", commit=True) -> "WROTE <path>"
- read_file(path, max_chars=120000) -> str
- list_dir(path=".") -> str
- run(cmd, timeout=600, env=None, workdir=".") -> str
- git(cmd) -> str
- apply_patch(unified_diff, commit_message="apply_patch") -> str

Neo4j / KB tools (operate offline via sidecars if DB is down):
- neo4j_query(cypher, params={}) -> rows
- neo4j_write(cypher, params={}) -> rows
- kb_ingest_text(title, text, tags=[]) -> "INGESTED Document <id> ..."
- kb_ingest_url(url, tags=[])       -> fetch HTML, convert to Markdown via Reader-LM, embed+store
- kb_similar_local(query_text, top_k=5) -> cosine over stored docs (fallback)
- kb_vector_search_neo4j(query_text, top_k=5) -> Neo4j native vector index (preferred when online)

Always ingest key outputs (HLD, README, API, test reports, build logs) as Documents.
""").strip()

def register_tools(agent: AssistantAgent):
    agent.register_function(
        function_map={
            "write_file": tool_write_file,
            "read_file": tool_read_file,
            "list_dir": tool_list_dir,
            "run": tool_run,
            "git": tool_git,
            "apply_patch": tool_apply_patch,
            "neo4j_query": neo4j_query,
            "neo4j_write": neo4j_write,
            "kb_ingest_text": kb_ingest_text,
            "kb_ingest_url": kb_ingest_url,
            "kb_similar_local": kb_similar_local,
            "kb_vector_search_neo4j": kb_vector_search_neo4j,
        }
    )

# =========================
# LLM config
# =========================
def make_llm_config(model_name: str, temperature: float = 0.3, timeout: int = 120) -> Dict[str, Any]:
    return {
        "config_list": [
            {
                "model": model_name,
                "base_url": OLLAMA_CHAT_BASE,
                "api_key": "ollama",
                "price": [0, 0],
            }
        ],
        "temperature": temperature,
        "timeout": timeout,
    }

# =========================
# Agents
# =========================
Planner = UserProxyAgent(
    name="Planner",
    human_input_mode="NEVER",
    llm_config=make_llm_config(MODELS["planner"], temperature=0.25),
    system_message=(
        "You are the Project Planner and orchestrator.\n"
        "- Restate acceptance criteria and record them as (:Project)-[:HAS_TASK]->(:Task).\n"
        "- Use kb_vector_search_neo4j or kb_similar_local to reference prior work.\n"
        "- Delegate: Architect → Coder ↔ Reviewer ↔ Tester → Builder; iterate until criteria pass.\n"
        "- Ensure HLD/README/USAGE/CHANGELOG/test reports/build logs are ingested to KB.\n"
        "- End with 'TERMINATE: <bulleted summary + artifact paths>'.\n"
        f"{TOOLS_DOC}"
    ),
    code_execution_config=False,
    is_termination_msg=lambda m: m.get("content","").strip().startswith("TERMINATE:")
)

Architect = AssistantAgent(
    name="Architect",
    llm_config=make_llm_config(MODELS["architect"], temperature=0.3),
    system_message=(
        "Software Architect: produce HLD, file layout, dependencies, risks; create scaffolding via write_file.\n"
        "Ingest HLD via kb_ingest_text(tags=['design']).\n"
        f"{TOOLS_DOC}"
    ),
)

Researcher = AssistantAgent(
    name="Researcher",
    llm_config=make_llm_config(MODELS["researcher"], temperature=0.3),
    system_message=(
        "Concise researcher: gather facts/options/tradeoffs; store via kb_ingest_text(tags=['research']).\n"
        "End major research outputs with 'RESEARCH COMPLETE'.\n"
        f"{TOOLS_DOC}"
    ),
)

Summarizer = AssistantAgent(
    name="Summarizer",
    llm_config=make_llm_config(MODELS["summarizer"], temperature=0.2),
    system_message=(
        "Summarizer: crisp summaries + next steps; ingest via kb_ingest_text(tags=['summary']).\n"
        "When Planner requests final summary, end with 'TERMINATE'.\n"
        f"{TOOLS_DOC}"
    ),
)

Coder = AssistantAgent(
    name="Coder",
    llm_config=make_llm_config(MODELS["coder"], temperature=0.2),
    system_message=(
        "Senior engineer: implement & refactor with tests; prefer TDD.\n"
        "Use write_file / apply_patch / run('pytest -q'); commit changes automatically.\n"
        "Ingest API/USAGE docs via kb_ingest_text.\n"
        f"{TOOLS_DOC}"
    ),
)

Reviewer = AssistantAgent(
    name="Reviewer",
    llm_config=make_llm_config(MODELS["reviewer"], temperature=0.2),
    system_message=(
        "Strict reviewer: correctness, security, style; provide diffs/patches; say 'REVIEW OK' when satisfied.\n"
        f"{TOOLS_DOC}"
    ),
)

Tester = AssistantAgent(
    name="Tester",
    llm_config=make_llm_config(MODELS["tester"], temperature=0.2),
    system_message=(
        "QA engineer: add/adjust pytest tests; run with run('pytest -q'); iterate until green.\n"
        f"{TOOLS_DOC}"
    ),
)

Builder = AssistantAgent(
    name="Builder",
    llm_config=make_llm_config(MODELS["builder"], temperature=0.2),
    system_message=(
        "Builder: create reproducible builds. For Python: pyproject.toml, 'pip install -e .', 'pytest -q', 'python -m build'.\n"
        "Log artifacts to Neo4j as (:Artifact {path, kind, ts}). Respond 'BUILD OK' on success.\n"
        f"{TOOLS_DOC}"
    ),
)

Collector = AssistantAgent(
    name="Collector",
    llm_config=make_llm_config(MODELS["researcher"], temperature=0.2),
    system_message=(
        "Collector: inventory repo, summarize state, ingest snapshots to KB. Use list_dir/read_file/kb_ingest_text.\n"
        f"{TOOLS_DOC}"
    ),
)

for a in (Architect, Researcher, Summarizer, Coder, Reviewer, Tester, Builder, Collector):
    register_tools(a)

# =========================
# GroupChat wiring
# =========================
def make_group():
    agents = [Planner, Architect, Researcher, Summarizer, Coder, Reviewer, Tester, Builder, Collector]
    gc = GroupChat(
        agents=agents,
        messages=[],
        max_round=40,
        speaker_selection_method="auto",
    )
    return GroupChatManager(groupchat=gc, llm_config=make_llm_config(MODELS["planner"]))

# =========================
# Bootstrap helpers
# =========================
def ensure_bootstrap(project_id: str, task: str) -> None:
    if ONLINE:
        neo4j_bootstrap_schema()
    # Project + Task
    neo4j_write("""
        MERGE (p:Project {id:$pid})
        ON CREATE SET p.createdAt=timestamp()
        SET p.updatedAt=timestamp(), p.title=$title
    """, {"pid": project_id, "title": task[:200]})
    neo4j_write("""
        MERGE (t:Task {id:$tid})
        SET t.text=$text, t.ts=timestamp()
        WITH t
        MATCH (p:Project {id:$pid})
        MERGE (p)-[:HAS_TASK]->(t)
    """, {"tid": f"task_{int(time.time()*1000)}", "text": task, "pid": project_id})
    # README
    readme = SANDBOX / "README.autogen.md"
    if not readme.exists():
        tool_write_file("README.autogen.md", f"# Project {project_id}\n\n{task}\n", commit=False)
        git_auto_commit("bootstrap: add README.autogen.md")

# =========================
# Sidecar ingestion (replay into Neo4j)
# =========================
def ingest_sidecar() -> None:
    if not ONLINE:
        print("Neo4j is offline; cannot ingest sidecars now.")
        return
    neo4j_bootstrap_schema()
    # Documents
    for rec in _sidecar_iter("document"):
        neo4j_write("""
            MERGE (d:Document {id:$id})
            SET d.title=$title, d.text=$text, d.tags=$tags, d.embedding=$embedding, d.ts=$ts
        """, {k: rec[k] for k in ["id","title","text","tags","embedding","ts"] if k in rec})
    # Files
    for rec in _sidecar_iter("file_write"):
        neo4j_write("""
            MERGE (f:File {path:$path})
            ON CREATE SET f.createdAt=$ts
            SET f.updatedAt=$ts, f.bytes=$bytes
        """, {"path": rec.get("path",""), "bytes": rec.get("bytes",0), "ts": rec.get("ts", int(time.time()*1000))})
    # Runs
    for rec in _sidecar_iter("run"):
        neo4j_write("""
            CREATE (r:Run {cmd:$cmd, output:$out, ts:$ts})
        """, {"cmd": rec.get("cmd",""), "out": rec.get("output","")[:200_000], "ts": rec.get("ts", int(time.time()*1000))})
    # Commits
    for rec in _sidecar_iter("commit"):
        neo4j_write("""
            MERGE (c:Commit {hash:$head})
            SET c.message=$msg, c.ts=$ts
        """, {"head": (rec.get("head","") or "").strip(), "msg": rec.get("message",""), "ts": rec.get("ts", int(time.time()*1000))})
    # Raw cypher writes captured while offline
    for rec in _sidecar_iter("cypher_write"):
        try:
            neo4j_write(rec.get("cypher",""), rec.get("params",{}))
        except Exception as e:
            print("Failed cypher from sidecar:", e)
    print("Sidecar ingestion complete.")

# =========================
# Cheatsheet
# =========================
def print_cheatsheet():
    print(textwrap.dedent(f"""
    AutoGen Coder Team — Cheatsheet
    --------------------------------
    # Start Ollama
    ollama serve

    # Pull models (examples)
    ollama pull {MODELS['planner']}
    ollama pull {MODELS['coder']}
    ollama pull {MODELS['reader']}
    ollama pull {MODELS['embed']}

    # Run with a task
    python {Path(__file__).name} run --project-id proj_local --task "Build a FastAPI service with /health and /slugify, tests, and a wheel + Dockerfile."

    # If Neo4j was offline, later ingest sidecars:
    python {Path(__file__).name} ingest-sidecar

    # Show sandbox tree
    tree {SANDBOX}

    # Common agent commands (inside the run loop via tools):
    - run("pytest -q")
    - write_file("pkg/__init__.py", "...source...")
    - apply_patch(\"\"\"<unified diff>\"\"\", commit_message="fix: apply patch")
    - kb_ingest_text("Design v1", "...", ["design"])
    - kb_vector_search_neo4j("slugify cli", 5)
    """).strip())

# =========================
# CLI
# =========================
def main():
    parser = argparse.ArgumentParser(description="AutoGen coder team with Neo4j vector index, git, and sidecars")
    subparsers = parser.add_subparsers(dest="cmd", required=False)

    p_run = subparsers.add_parser("run", help="Start multi-agent workflow (default)")
    p_run.add_argument("--task", required=True, help="What should the team build?")
    p_run.add_argument("--project-id", default="proj_local", help="Logical project id for Neo4j")
    p_run.add_argument("--reset", action="store_true", help="Delete sandbox before starting")

    subparsers.add_parser("ingest-sidecar", help="Replay sidecar JSONL into Neo4j")
    subparsers.add_parser("cheatsheet", help="Print common commands")

    # default to run if no subcommand
    if len(sys.argv) == 1 or (len(sys.argv) > 1 and sys.argv[1] not in {"run", "ingest-sidecar", "cheatsheet"}):
        sys.argv.insert(1, "run")

    args = parser.parse_args()

    if args.cmd == "cheatsheet":
        print_cheatsheet()
        return

    if args.cmd == "ingest-sidecar":
        if ONLINE:
            neo4j_bootstrap_schema()
        ingest_sidecar()
        return

    # run
    if args.reset and SANDBOX.exists():
        shutil.rmtree(SANDBOX)
        SANDBOX.mkdir(parents=True, exist_ok=True)

    ensure_bootstrap(args.project_id, args.task)

    kickoff = textwrap.dedent(f"""
    PROJECT BRIEF
    -------------
    {args.task}

    Acceptance Criteria
    -------------------
    - Clear HLD and file layout recorded in Neo4j (or sidecars if offline).
    - Code is documented and passes 'pytest -q'.
    - Build artifacts produced (e.g., dist/*.whl or Docker image) and recorded as (:Artifact) or sidecar.
    - Key documentation (README/USAGE/CHANGELOG) ingested into KB :Document with embeddings.
    - Final summary lists artifact paths and references to prior similar documents.
    - All file operations are committed to git automatically.
    """)

    print(f"Sandbox: {SANDBOX}")
    print("Neo4j online:", ONLINE)
    print("Starting multi-agent development...")
    print("-"*70)

    manager = make_group()
    Planner.initiate_chat(manager, message=kickoff)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
