"""LongMemEval evaluation runner for Patha.

Usage (once dataset + env are set up):

    python -m eval.runner --data path/to/longmemeval_oracle.json \\
                          --config configs/default.yaml \\
                          --output runs/<timestamp>/results.json

This module handles:
1. Loading LongMemEval JSON (oracle or S/M variants).
2. Ingesting all haystack sessions into a Patha store.
3. Running each question through the retrieval pipeline.
4. Computing R@K metrics via eval.metrics.
5. Writing per-question traces + aggregate results.

For now (stub-embedder phase), it runs against a synthetic fixture to
validate the harness wiring. Real LongMemEval runs require the full
env (uv sync, GPU, Qwen3 models).
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from patha.chunking.views import VIEW_NAMES
from patha.indexing.bm25_index import SimpleBM25
from patha.indexing.ingest import ingest_session
from patha.indexing.songline_graph import build_songline_graph
from patha.indexing.store import InMemoryStore
from patha.models.embedder import Embedder, StubEmbedder
from patha.retrieval.pipeline import PipelineConfig, RetrievalResult, retrieve

from eval.metrics import (
    EvalReport,
    QuestionResult,
    _classify_stratum,
)


def load_longmemeval(path: str | Path) -> list[dict]:
    """Load a LongMemEval JSON file.

    Expected format: a list of dicts, each with at least:
    - question_id: str
    - question: str
    - question_type: str
    - answer_session_ids: list[str]
    - haystack_sessions: list[list[dict]]  (each session is a list of
      {role, content} turns)
    - haystack_session_ids: list[str]
    """
    path = Path(path)
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list, got {type(data).__name__}")
    return data


def _ingest_haystack(
    entry: dict,
    store: InMemoryStore,
    embedder: Embedder,
    bm25: SimpleBM25,
) -> None:
    """Ingest all haystack sessions for one LongMemEval entry.

    LongMemEval format: haystack_sessions is a list of sessions, each
    session is a list of {role, content, [has_answer]} turn dicts.
    haystack_session_ids is a parallel list of session ID strings.
    """
    session_ids = entry["haystack_session_ids"]
    sessions = entry["haystack_sessions"]

    for sid, session_turns in zip(session_ids, sessions):
        turns = []
        for turn in session_turns:
            turns.append({
                "text": turn.get("content", ""),
                "speaker": turn.get("role", None),
            })
        ingest_session(turns, session_id=sid, store=store, embedder=embedder)

    # Populate BM25 from all ingested rows
    for row in store.all_rows():
        bm25.add(row["chunk_id"], row["text"])


def run_single_question(
    entry: dict,
    store: InMemoryStore,
    embedder: Embedder,
    bm25: SimpleBM25,
    config: PipelineConfig,
    songline_graph=None,
) -> QuestionResult:
    """Run the Patha pipeline on one LongMemEval question and score it."""
    question = entry["question"]
    question_id = entry["question_id"]
    question_type = entry["question_type"]
    gold_session_ids = entry.get("answer_session_ids", [])

    result = retrieve(
        question,
        store=store,
        embedder=embedder,
        bm25=bm25,
        songline_graph=songline_graph,
        config=config,
    )

    return QuestionResult(
        question_id=question_id,
        question_type=question_type,
        stratum=_classify_stratum(question_type),
        gold_session_ids=gold_session_ids,
        retrieved_chunk_ids=result.top_ids,
    )


def run_evaluation(
    data: list[dict],
    *,
    embedder: Embedder | None = None,
    config: PipelineConfig | None = None,
    use_songline: bool = True,
    verbose: bool = False,
) -> EvalReport:
    """Run the full LongMemEval evaluation.

    Each entry gets its own fresh store (matching the official protocol
    where each question has its own haystack). This is the correct
    approach: LongMemEval provides per-question haystacks, not a single
    shared corpus.

    Parameters
    ----------
    data
        Loaded LongMemEval JSON (list of entry dicts).
    embedder
        Embedder to use. Defaults to StubEmbedder(dim=64).
    config
        Pipeline config. Defaults to PipelineConfig().
    use_songline
        Whether to build and use the songline graph. Default True.
    verbose
        Print per-question progress. Default False.

    Returns
    -------
    EvalReport
        Aggregate metrics over all questions.
    """
    if embedder is None:
        embedder = StubEmbedder(dim=64)
    if config is None:
        config = PipelineConfig()

    report = EvalReport()
    total = len(data)

    for i, entry in enumerate(data):
        qid = entry["question_id"]
        if verbose:
            print(f"  [{i + 1}/{total}] {qid} ({entry['question_type']})")

        # Fresh store per question (per LongMemEval protocol)
        store = InMemoryStore()
        bm25 = SimpleBM25()

        # Ingest this question's haystack
        _ingest_haystack(entry, store, embedder, bm25)

        # Optionally build songline graph
        songline_graph = None
        if use_songline:
            songline_graph = build_songline_graph(list(store.all_rows()))

        # Run retrieval
        qr = run_single_question(
            entry, store, embedder, bm25, config, songline_graph,
        )
        report.results.append(qr)

    return report


def save_results(
    report: EvalReport,
    output_path: str | Path,
    ks: list[int] | None = None,
) -> None:
    """Write evaluation results to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": report.summary(ks=ks),
        "per_question": [
            {
                "question_id": r.question_id,
                "question_type": r.question_type,
                "stratum": r.stratum,
                "gold_session_ids": r.gold_session_ids,
                "retrieved_chunk_ids": r.retrieved_chunk_ids,
                "recall_any@5": r.recall_any_at_k(5),
                "recall_any@10": r.recall_any_at_k(10),
            }
            for r in report.results
        ],
    }

    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)


# ─── CLI entrypoint ──────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    """Minimal CLI for running the eval harness.

    Usage:
        python -m eval.runner --data longmemeval_oracle.json [--output results.json]
    """
    import argparse

    parser = argparse.ArgumentParser(description="Patha LongMemEval runner")
    parser.add_argument("--data", required=True, help="Path to LongMemEval JSON")
    parser.add_argument("--output", default="runs/latest/results.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--embedder", default="stub",
                        choices=["stub", "minilm", "stella"],
                        help="Embedder to use (default: stub)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit to first N questions (0 = all)")
    parser.add_argument("--no-songline", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    print(f"Loading {args.data}...")
    data = load_longmemeval(args.data)
    if args.limit > 0:
        data = data[:args.limit]
    print(f"Loaded {len(data)} questions")

    embedder: Embedder | None = None
    if args.embedder == "minilm":
        from patha.models.embedder_st import SentenceTransformerEmbedder
        embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
        print(f"Using MiniLM embedder (dim={embedder.dim})")
    elif args.embedder == "stella":
        from patha.models.embedder_st import SentenceTransformerEmbedder
        embedder = SentenceTransformerEmbedder("dunzhang/stella_en_400M_v5")
        print(f"Using Stella-400M embedder (dim={embedder.dim})")

    config = PipelineConfig(top_k=args.top_k)
    report = run_evaluation(
        data,
        embedder=embedder,
        config=config,
        use_songline=not args.no_songline,
        verbose=args.verbose,
    )

    summary = report.summary()
    print("\n=== RESULTS ===")
    for k, v in summary.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for sk, sv in v.items():
                print(f"    {sk}: {sv}")
        else:
            print(f"  {k}: {v}")

    save_results(report, args.output)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
