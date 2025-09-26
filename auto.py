#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AutoGen Coder Team (Ollama + Neo4j memory & model routing)

What’s new vs. the base template:
- Per-agent MODEL ROUTING to local Ollama models you listed:
  - gemma3:270m,1b,4b  -> research/summarize/general
  - qwen2.5-coder:0.5b,1.5b,7b -> coding/editing/testing
  - qwen3:0.6b,1.7b,4b (MoE) -> planning/complex generalization
  - reader-lm:0.5b,1.5b -> HTML → Markdown ingestion
  - nomic-embed-text:latest -> embeddings for Neo4j vector properties
- Neo4j-backed project memory:
  - Nodes: Project, Task, Message, File, Run, Artifact, Document
  - Relationships: (:Project)-[:HAS_TASK]->(:Task), (:Task)-[:HAS_FILE]->(:File), etc.
  - Embeddings stored on :Document(embedding: List<Float>) for vector search
- Tools exposed to agents:
  - write_file/read_file/list_dir/run (sandboxed)
  - neo4j_query/neo4j_write (Cypher)
  - kb_ingest_text(title, text, tags) -> stores as :Document with embedding
  - kb_ingest_url(url) -> fetch, convert HTML→MD via Reader-LM, then embed+store
  - kb_similar(text_or_title, top_k) -> cosine search (client-side) over stored embeddings
- Planner and Builder now **log** acceptance criteria, build results, and artifacts to Neo4j.

Env vars you’ll likely set:
  SANDBOX=./_coder_sandbox
  OLLAMA_BASE_URL=http://localhost:11434
  OLLAMA_CHAT_BASE=http://localhost:11434/v1   (AutoGen OpenAI-compatible)
  OLLAMA_EMBED_BASE=http://localhost:11434
  NEO4J_URI=bolt://localhost:7687
  NEO4J_USER=neo4j
  NEO4J_PASSWORD=yourpassword
  NEO4J_DB=neo4j
"""

import os, sys, json, time, argparse, shutil, subprocess, textwrap, math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import requests

from autogen import AssistantAgent, UserProxyAgent
from autogen.agentchat.groupchat import GroupChat, GroupChatManager

# -------------------------
# Config
# -------------------------
SANDBOX = Path(os.getenv("SANDBOX", "./_coder_sandbox")).resolve()
SANDBOX.mkdir(parents=True, exist_ok=True)

# Ollama endpoints
OLLAMA_CHAT_BASE = os.getenv("OLLAMA_CHAT_BASE", "http://localhost:11434/v1")
OLLAMA_EMBED_BASE = os.getenv("OLLAMA_EMBED_BASE", "http://localhost:11434")

# Neo4j
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "changeme")
NEO4J_DB = os.getenv("NEO4J_DB", "neo4j")

# Models (Ollama model names you said you have)
MODELS = {
    "planner": "qwen3:4b",                      # orchestrator / complex planning
    "architect": "qwen3:4b",                    # high-level design & rationale
    "researcher": "gemma3:4b",                  # concise research
    "summarizer": "gemma3:1b",                  # shorter summaries
    "coder": "qwen2.5-coder:7b",                # best coding ability available
    "reviewer": "qwen2.5-coder:1.5b",           # faster lint/review
    "tester": "qwen2.5-coder:1.5b",             # test writing & edits
    "builder": "qwen2.5-coder:1.5b",            # packaging/Docker
    "reader": "reader-lm:1.5b",                 # HTML -> Markdown conversion
    "embed": "nomic-embed-text:latest",         # text embeddings
}

# LLM config helper (AutoGen OpenAI-compatible adapter)
def make_llm_config(model_name: str, temperature: float = 0.3, timeout: int = 120) -> Dict[str, Any]:
    return {
        "config_list": [
            {
                "model": model_name,
                "base_url": OLLAMA_CHAT_BASE,
                "api_key": "ollama",  # dummy
                "price": [0, 0],
            }
        ],
        "temperature": temperature,
        "timeout": timeout,
    }

# -------------------------
# Sandbox tools
# -------------------------
def _safe_path(rel: str) -> Path:
    p = (SANDBOX / rel).resolve()
    if SANDBOX not in p.parents and p != SANDBOX:
        raise ValueError(f"Path escapes sandbox: {p}")
    return p

def tool_write_file(path: str, content: str, mkparents: bool = True, mode: str = "w") -> str:
    p = _safe_path(path)
    if mkparents:
        p.parent.mkdir(parents=True, exist_ok=True)
    with p.open(mode, encoding="utf-8") as f:
        f.write(content)
    # log into Neo4j as File node
    neo4j_write("""
        MERGE (f:File {path:$path})
        ON CREATE SET f.createdAt=timestamp()
        SET f.updatedAt=timestamp(), f.bytes=$bytes
    """, {"path": str(p), "bytes": p.stat().st_size if p.exists() else len(content)})
    return f"WROTE {p}"

def tool_read_file(path: str, max_chars: int = 120_000) -> str:
    p = _safe_path(path)
    data = p.read_text(encoding="utf-8")
    if len(data) > max_chars:
        return data[:max_chars] + f"\n[...TRUNCATED {len(data)-max_chars} chars]"
    return data

def tool_list_dir(path: str = ".") -> str:
    p = _safe_path(path)
    if not p.exists():
        return f"{p} (does not exist)"
    out = []
    for item in sorted(p.rglob("*")):
        rel = item.relative_to(SANDBOX)
        out.append(str(rel) + ("/" if item.is_dir() else ""))
    return "\n".join(out) if out else "(empty)"

def tool_run(cmd: str, timeout: int = 480, env: Dict[str, str] | None = None, workdir: str = ".") -> str:
    wd = _safe_path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    try:
        p = subprocess.run(
            cmd, shell=True, cwd=str(wd), env=full_env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=timeout, text=True
        )
        out = f"$ {cmd}\n(exit={p.returncode})\n{p.stdout}"
        neo4j_write("""
            CREATE (r:Run {cmd:$cmd, exit:$exit, output:$out, ts:timestamp()})
        """, {"cmd": cmd, "exit": p.returncode, "out": out[:200_000]})
        return out
    except subprocess.TimeoutExpired as e:
        neo4j_write("""
            CREATE (r:Run {cmd:$cmd, exit:999, output:$out, ts:timestamp()})
        """, {"cmd": cmd, "out": f"TIMEOUT after {timeout}s\n{e.output or ''}"})
        return f"$ {cmd}\n(TIMEOUT after {timeout}s)\n{e.output or ''}"

# -------------------------
# Neo4j client (requests → Neo4j Python driver isn’t required, we use HTTP via neo4j’s transactional HTTP API if available;
# but the Bolt driver is more common. To avoid extra deps, we’ll use the official python neo4j driver if present, else fallback.)
# -------------------------
_driver = None
def _get_driver():
    global _driver
    if _driver is None:
        try:
            from neo4j import GraphDatabase
            _driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        except Exception as e:
            raise RuntimeError(f"Neo4j driver init failed: {e}")
    return _driver

def neo4j_write(cypher: str, params: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    d = _get_driver()
    with d.session(database=NEO4J_DB) as s:
        res = s.run(cypher, params or {})
        return [r.data() for r in res]

def neo4j_query(cypher: str, params: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    d = _get_driver()
    with d.session(database=NEO4J_DB) as s:
        res = s.run(cypher, params or {})
        return [r.data() for r in res]

def neo4j_bootstrap_schema():
    # Minimal indexes and helpful constraints (idempotent)
    stmts = [
        "CREATE INDEX IF NOT EXISTS FOR (p:Project) ON (p.id)",
        "CREATE INDEX IF NOT EXISTS FOR (t:Task) ON (t.id)",
        "CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.id)",
        "CREATE INDEX IF NOT EXISTS FOR (f:File) ON (f.path)",
    ]
    for c in stmts:
        try:
            neo4j_write(c)
        except Exception:
            pass

# -------------------------
# Knowledge base (Documents + embeddings)
# -------------------------
def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    if na == 0 or nb == 0: return 0.0
    return dot / (na * nb)

def embed_text(text: str) -> List[float]:
    # Use Ollama embeddings API: POST /api/embeddings {model, prompt}
    url = f"{OLLAMA_EMBED_BASE}/api/embeddings"
    r = requests.post(url, json={"model": MODELS["embed"], "prompt": text}, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data["embedding"]

def kb_ingest_text(title: str, text: str, tags: Optional[List[str]] = None) -> str:
    vec = embed_text(text[:8000])  # limit tokens for speed
    params = {
        "id": f"doc_{int(time.time()*1000)}",
        "title": title, "text": text, "tags": tags or [],
        "embedding": vec, "ts": int(time.time()*1000)
    }
    neo4j_write("""
        MERGE (d:Document {id:$id})
        SET d.title=$title, d.text=$text, d.tags=$tags, d.embedding=$embedding, d.ts=$ts
    """, params)
    return f"INGESTED Document {params['id']} with {len(vec)}-d embedding"

def _reader_convert_html_to_md(html: str) -> str:
    # Let Reader-LM do the heavy lifting via chat completion
    # System prompt instructs conversion to Markdown
    url = f"{OLLAMA_CHAT_BASE}/chat/completions"
    sys_prompt = (
        "You convert raw HTML into clean, readable Markdown. "
        "Preserve headings, links, lists, tables, and code blocks."
    )
    payload = {
        "model": MODELS["reader"],
        "messages": [
            {"role":"system","content":sys_prompt},
            {"role":"user","content":f"Convert this HTML to Markdown:\n\n{html}"}
        ],
        "temperature": 0.0,
    }
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    md = r.json()["choices"][0]["message"]["content"]
    return md

def kb_ingest_url(url: str, tags: Optional[List[str]] = None) -> str:
    import urllib.request
    with urllib.request.urlopen(url, timeout=60) as resp:
        html = resp.read().decode("utf-8", errors="ignore")
    md = _reader_convert_html_to_md(html)
    title = f"URL:{url}"
    # Save raw + md as files too
    tool_write_file(f"kb/raw/{Path(url).name or 'page'}.html", html)
    tool_write_file(f"kb/markdown/{Path(url).name or 'page'}.md", md)
    return kb_ingest_text(title, md, tags=tags or ["url"])

def kb_similar(query_text: str, top_k: int = 5) -> str:
    qv = embed_text(query_text[:8000])
    rows = neo4j_query("MATCH (d:Document) RETURN d.id AS id, d.title AS title, d.embedding AS emb LIMIT 500")
    scored = []
    for r in rows:
        emb = r.get("emb") or r.get("d.embedding")  # defensive
        if not emb: continue
        score = _cosine(qv, emb)
        scored.append((score, r["id"], r.get("title","")))
    scored.sort(reverse=True)
    top = scored[:max(1, min(top_k, len(scored)))]
    return "\n".join([f"{i+1}. {doc_id} :: {title} :: score={score:.4f}" for i,(score,doc_id,title) in enumerate(top)]) or "(no matches)"

# -------------------------
# Register functions on agents
# -------------------------
TOOLS_DOC = textwrap.dedent(f"""
You have these sandboxed tools:

- write_file(path, content, mkparents=True, mode="w") -> "WROTE <path>"
- read_file(path, max_chars=120000) -> str
- list_dir(path=".") -> str
- run(cmd, timeout=480, env=None, workdir=".") -> str

And these Neo4j / KB tools:

- neo4j_query(cypher, params={{}}) -> rows
- neo4j_write(cypher, params={{}}) -> rows
- kb_ingest_text(title, text, tags=[]) -> "INGESTED Document <id> ..."
- kb_ingest_url(url, tags=[])       -> fetch HTML, convert to Markdown via Reader-LM, embed+store
- kb_similar(query_text, top_k=5)   -> top matches by cosine similarity

Always prefer to ingest project docs (plans, designs, READMEs, test reports) into Neo4j.
When you complete meaningful steps (designs, build logs, artifacts), record them in Neo4j.
""").strip()

def register_tools(agent: AssistantAgent):
    agent.register_function(
        function_map={
            "write_file": tool_write_file,
            "read_file": tool_read_file,
            "list_dir": tool_list_dir,
            "run": tool_run,
            "neo4j_query": neo4j_query,
            "neo4j_write": neo4j_write,
            "kb_ingest_text": kb_ingest_text,
            "kb_ingest_url": kb_ingest_url,
            "kb_similar": kb_similar,
        }
    )

# -------------------------
# Agents (with routed models)
# -------------------------
Planner = UserProxyAgent(
    name="Planner",
    human_input_mode="NEVER",
    llm_config=make_llm_config(MODELS["planner"], temperature=0.25),
    system_message=(
        "You are the Project Planner and orchestrator.\n"
        "- Gather or restate acceptance criteria and persist them to Neo4j as (:Project)-[:HAS_TASK]->(:Task).\n"
        "- Query KB for similar prior work using kb_similar; reference it.\n"
        "- Delegate to Architect → Coder ↔ Reviewer ↔ Tester → Builder, iterate until all criteria pass.\n"
        "- Ensure important outputs (HLD, README, CHANGELOG, test reports, build logs) are ingested to Neo4j.\n"
        "- Finish with 'TERMINATE: <bulleted summary + artifact paths>'.\n"
        f"{TOOLS_DOC}"
    ),
    code_execution_config=False,
    is_termination_msg=lambda m: m.get("content","").strip().startswith("TERMINATE:")
)

Architect = AssistantAgent(
    name="Architect",
    llm_config=make_llm_config(MODELS["architect"], temperature=0.3),
    system_message=(
        "You are the Software Architect. Provide HLD, file layout, dependencies, risks.\n"
        "- Write initial scaffolding files using write_file.\n"
        "- Ingest your HLD into Neo4j using kb_ingest_text with tags=['design'].\n"
        f"{TOOLS_DOC}"
    ),
)

Researcher = AssistantAgent(
    name="Researcher",
    llm_config=make_llm_config(MODELS["researcher"], temperature=0.3),
    system_message=(
        "You are a concise researcher. Collect facts, options, tradeoffs.\n"
        "- Store findings via kb_ingest_text(tags=['research']).\n"
        "End major research sections with 'RESEARCH COMPLETE'.\n"
        f"{TOOLS_DOC}"
    ),
)

Summarizer = AssistantAgent(
    name="Summarizer",
    llm_config=make_llm_config(MODELS["summarizer"], temperature=0.2),
    system_message=(
        "You are a summarizer. Produce crisp summaries with key bullets and next steps.\n"
        "- Ingest summaries via kb_ingest_text(tags=['summary']).\n"
        "End your final message with 'TERMINATE' when the Planner asks for final summary.\n"
        f"{TOOLS_DOC}"
    ),
)

Coder = AssistantAgent(
    name="Coder",
    llm_config=make_llm_config(MODELS["coder"], temperature=0.2),
    system_message=(
        "You are a senior software engineer. Implement and refactor code with tests.\n"
        "- Use write_file, run('pytest -q'), and keep code well-documented.\n"
        "- Ingest important docs (API docs/README/USAGE) into Neo4j via kb_ingest_text.\n"
        f"{TOOLS_DOC}"
    ),
)

Reviewer = AssistantAgent(
    name="Reviewer",
    llm_config=make_llm_config(MODELS["reviewer"], temperature=0.2),
    system_message=(
        "You are a strict code reviewer; focus on correctness, security, style.\n"
        "- Provide concrete diffs or patch suggestions; request fixes.\n"
        "- When satisfied, respond with 'REVIEW OK'.\n"
        f"{TOOLS_DOC}"
    ),
)

Tester = AssistantAgent(
    name="Tester",
    llm_config=make_llm_config(MODELS["tester"], temperature=0.2),
    system_message=(
        "You are QA. Add/adjust pytest tests for coverage and edge cases.\n"
        "- Run tests with run('pytest -q'); iterate until green.\n"
        f"{TOOLS_DOC}"
    ),
)

Builder = AssistantAgent(
    name="Builder",
    llm_config=make_llm_config(MODELS["builder"], temperature=0.2),
    system_message=(
        "You build packages and images. Ensure reproducible builds and log outputs.\n"
        "- For Python: create pyproject.toml, run 'pip install -e .', 'pytest -q', 'python -m build'.\n"
        "- Log build results to Neo4j as (:Artifact {path, kind}).\n"
        "- When done and artifacts exist, say 'BUILD OK'.\n"
        f"{TOOLS_DOC}"
    ),
)

Collector = AssistantAgent(
    name="Collector",
    llm_config=make_llm_config(MODELS["researcher"], temperature=0.2),
    system_message=(
        "You collect project state: read repo files, summarize, and ingest into Neo4j.\n"
        "- Use list_dir/read_file to inventory code, then kb_ingest_text for documentation snapshots.\n"
        f"{TOOLS_DOC}"
    ),
)

# Register tools for all build-capable / KB-capable agents
for a in (Architect, Researcher, Summarizer, Coder, Reviewer, Tester, Builder, Collector):
    register_tools(a)

# -------------------------
# GroupChat wiring
# -------------------------
def make_group():
    agents = [Planner, Architect, Researcher, Summarizer, Coder, Reviewer, Tester, Builder, Collector]
    gc = GroupChat(
        agents=agents,
        messages=[],
        max_round=40,
        speaker_selection_method="auto",
    )
    return GroupChatManager(groupchat=gc, llm_config=make_llm_config(MODELS["planner"]))

# -------------------------
# Bootstrap helpers
# -------------------------
def ensure_bootstrap(project_id: str, task: str) -> None:
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
        tool_write_file("README.autogen.md", f"# Project {project_id}\n\n{task}\n")

# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="AutoGen coder team with Neo4j + model routing")
    parser.add_argument("--task", required=True, help="What should the team build?")
    parser.add_argument("--project-id", default="proj_local", help="Logical project id for Neo4j")
    parser.add_argument("--reset", action="store_true", help="Delete sandbox before starting")
    args = parser.parse_args()

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
    - Clear HLD and file layout recorded in Neo4j.
    - Code is documented and passes 'pytest -q'.
    - Build artifacts produced (e.g., dist/*.whl or Docker image) and recorded as (:Artifact).
    - Key documentation (README/USAGE/CHANGELOG) ingested into Neo4j as :Document.
    - Final summary lists artifact paths and references to prior similar documents (if found).
    """)

    print(f"Sandbox: {SANDBOX}")
    print("Starting multi-agent development with Neo4j memory...")
    print("-"*70)

    manager = make_group()
    Planner.initiate_chat(manager, message=kickoff)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")

"""
# 1) Start Ollama
ollama serve

# 2) Make sure the models are pulled (examples)
ollama pull gemma3:4b
ollama pull qwen2.5-coder:7b
ollama pull qwen3:4b
ollama pull reader-lm:1.5b
ollama pull nomic-embed-text:latest

# 3) Neo4j running locally (change creds as needed)
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=yourpassword
export NEO4J_DB=neo4j

# 4) Optional: set sandbox
export SANDBOX="$PWD/devspace"

# 5) Install deps
pip install autogen neo4j requests

# 6) Run a task
python autogen_coder_team_neo4j.py \
  --project-id proj_ollama_dev \
  --task "Create a FastAPI microservice with two endpoints, tests (pytest), wheel build, and a minimal Dockerfile; document everything. Ingest docs into Neo4j."
"""