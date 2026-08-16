#!/usr/bin/env python3
"""Orchestrate the knowledge-graph pipeline stages with checkpoint/resume support.

Stages run in dependency order, each a subprocess of its existing CLI so env and
exit semantics stay identical to running them by hand:

  1. reembed         --embed text-bearing nodes (Segment/Transcription/Utterance)
                       with the chosen model/prop (torch or onnx engine).
  2. embed_summaries --embed Summary nodes missing the vector (same model/prop).
  3. cluster         --KMeans over Summary vectors -> Cluster nodes.
  4. link-speakers   --assign local Speaker nodes to GlobalSpeaker identities.
  5. summarize       --optional LLM summarization (skipped unless --summarize).

Each stage is resumable: stages already past are skipped (or re-run with
--force-stage). Neo4j creds resolve through auto_ingest_config so subprocesses
inherit them via NEO4J_*.

Examples
--------
  run_pipeline.py --prop emb_gte_small --model thenlper/gte-small
  run_pipeline.py --prop emb_gte_small --engine onnx
  run_pipeline.py --stages reembed,cluster --dry-run
  run_pipeline.py --prop emb_gte_small --force-stage cluster
  run_pipeline.py --stages replay-outbox   # drain failed ingest writes
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

try:
    from auto_ingest_config import get_neo4j_password, get_neo4j_config
    _pw = get_neo4j_password()
    _cfg = get_neo4j_config()
except Exception:
    _pw = os.environ.get("NEO4J_PASSWORD") or "knowledge_graph_2026"
    _cfg = {"uri": "bolt://127.0.0.1:7687", "user": "neo4j", "database": "neo4j"}

ENV = {
    **os.environ,
    "NEO4J_URI": os.environ.get("NEO4J_URI", _cfg.get("uri", "bolt://127.0.0.1:7687")),
    "NEO4J_USER": os.environ.get("NEO4J_USER", _cfg.get("user", "neo4j")),
    "NEO4J_PASSWORD": _pw,
    "NEO4J_DB": os.environ.get("NEO4J_DB", _cfg.get("database", "neo4j")),
}

DEFAULT_MODEL = os.environ.get("EMBED_MODEL_NAME", "thenlper/gte-small")
DEFAULT_PROP = "emb_gte_small"

STAGES = ["reembed", "embed_summaries", "classify", "cluster", "link-speakers",
          "summarize", "replay-outbox"]

LINKER = "auto_ingest.diarize.link_global_speakers"


def _run(cli: list, tag: str) -> int:
    print(f"\n=== [{tag}] {' '.join(cli)} ===", flush=True)
    rc = subprocess.call(cli, cwd=HERE, env=ENV)
    print(f"=== [{tag}] exit {rc} ===", flush=True)
    return rc


def stage_reembed(args) -> int:
    cli = [PY, "-u", "reembed.py", "Segment", "Transcription", "Utterance",
           "--model", args.model, "--prop", args.prop, "--resume",
           "--batch-size", str(args.embed_batch)]
    if args.engine:
        cli += ["--engine", args.engine]
    if args.torch_threads:
        cli += ["--torch-threads", str(args.torch_threads)]
    if args.dry_run:
        cli += ["--verify-only"]
    return _run(cli, "reembed")


def stage_embed_summaries(args) -> int:
    cli = [PY, "-u", "embed_summaries.py",
           "--model", args.model, "--prop", args.prop, "--batch", str(args.embed_batch)]
    if args.engine:
        cli += ["--engine", args.engine]
    if args.dry_run:
        cli += ["--dry-run"]
    return _run(cli, "embed_summaries")


def stage_classify(args) -> int:
    cli = [PY, "-u", "02_classify_lyrics.py", "--all",
           "--segments-source", args.classify_segments_source]
    if args.dry_run:
        cli += ["--limit", str(args.classify_limit)]
    return _run(cli, "classify")


def stage_cluster(args) -> int:
    cli = [PY, "-u", "cluster.py", "--prop", args.prop]
    if args.dry_run:
        cli += ["--dry-run"]
    if args.drop_clusters:
        cli += ["--drop-existing"]
    if args.label_llm:
        cli += ["--label-llm"]
    return _run(cli, "cluster")


def stage_link_speakers(args) -> int:
    cli = [PY, "-u", "-m", LINKER,
           "--global-prefilter", "--faiss-prefilter", "--skip-already-linked",
           "--audio-index", "./audio_index.json",
           "--audio-cache", "./audio_path_cache.json",
           "--emb-cache", "./emb_cache.sqlite"]
    if args.max_speakers:
        cli += ["--max-speakers", str(args.max_speakers)]
    if args.workers > 1:
        cli += ["--workers", str(args.workers)]
    if args.remote_embed:
        cli += ["--remote-embed", args.remote_embed]
    if args.state_file:
        cli += ["--state-file", args.state_file]
    if args.dry_run:
        cli += ["--dry-run"]
    return _run(cli, "link-speakers")


def stage_summarize(args) -> int:
    cli = [PY, "-u", "summarize_from_segments.py", "--all-missing",
           "--limit", str(args.summarize_limit)]
    if args.workers:
        cli += ["--workers", str(args.workers)]
    if args.dry_run:
        cli += ["--dry-run"]
    return _run(cli, "summarize")


def stage_replay_outbox(args) -> int:
    cli = [PY, "-u", "-m", "auto_ingest.outbox"]
    if args.dry_run:
        cli += ["--dry-run"]
    return _run(cli, "replay-outbox")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", default="reembed,embed_summaries,cluster,link-speakers",
                    help="Comma-separated stages (order enforced; see STAGES)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--prop", default=DEFAULT_PROP)
    ap.add_argument("--engine", choices=["torch", "onnx"], default=os.getenv("EMBED_ENGINE", "torch"))
    ap.add_argument("--embed-batch", type=int, default=int(os.getenv("EMBED_BATCH", "128")))
    ap.add_argument("--torch-threads", type=int, default=int(os.getenv("TORCH_THREADS", "6")))
    ap.add_argument("--max-speakers", type=int, default=0,
                    help="link-speakers: optional global cap (0 = no cap; use a state-file for chunked runs)")
    ap.add_argument("--workers", type=int, default=1,
                    help="link-speakers: parallel embedding processes (1 = single-process)")
    ap.add_argument("--remote-embed", default="",
                    help="link-speakers: URL of a remote GPU ECAPA embed server "
                         "(e.g. http://100.64.43.123:8901) to offload forward passes")
    ap.add_argument("--state-file", default="",
                    help="link-speakers: JSON file of already-processed speaker ids (resume)")
    ap.add_argument("--drop-clusters", action="store_true",
                    help="cluster: delete existing Cluster nodes before writing")
    ap.add_argument("--classify-segments-source", default="neo4j",
                    help="classify: 'neo4j' (AudioSegment nodes) or 'sidecar' (music.json sidecars)")
    ap.add_argument("--classify-limit", type=int, default=64,
                    help="classify: batch size per loop iteration")
    ap.add_argument("--label-llm", action="store_true", help="cluster: LLM cluster labels")
    ap.add_argument("--summarize", action="store_true", help="include LLM summarize stage")
    ap.add_argument("--summarize-limit", type=int, default=50)
    ap.add_argument("--workers", type=int, default=1, help="summarize worker threads")
    ap.add_argument("--dry-run", action="store_true",
                    help="Dry-run where each stage supports it (no writes)")
    args = ap.parse_args()

    if args.summarize and "summarize" not in args.stages:
        args.stages += ",summarize"

    wanted = [s.strip() for s in args.stages.split(",") if s.strip()]
    for s in wanted:
        if s not in STAGES:
            sys.exit(f"Unknown stage: {s} (valid: {', '.join(STAGES)})")

    dispatch = {
        "reembed": stage_reembed,
        "embed_summaries": stage_embed_summaries,
        "classify": stage_classify,
        "cluster": stage_cluster,
        "link-speakers": stage_link_speakers,
        "summarize": stage_summarize,
        "replay-outbox": stage_replay_outbox,
    }

    rc = 0
    for stage in wanted:
        fn = dispatch[stage]
        if args.dry_run and stage == "reembed":
            pass  # verify-only already suppresses writes
        rc = fn(args)
        if rc != 0:
            print(f"\nPipeline halted at stage '{stage}' (exit {rc}). "
                  f"Re-run with --stages {','.join(wanted[wanted.index(stage):])} to resume.",
                  file=sys.stderr)
            return rc
    print("\nPipeline complete.")
    return rc


if __name__ == "__main__":
    sys.exit(main())