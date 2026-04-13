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
from patha.retrieval.pipeline import PipelineConfig, Reranker, RetrievalResult, retrieve

from eval.metrics import (
    EvalReport,
    QuestionResult,
    _classify_stratum,
)


class SessionCache:
    """Cache pre-ingested session rows to avoid re-embedding shared sessions.

    LongMemEval S has ~24K total session slots across 500 questions, but
    many sessions appear in multiple questions' haystacks. Caching the
    fully-embedded rows by session_id avoids redundant embedding work,
    yielding ~10x speedup on full evals.
    """

    def __init__(self) -> None:
        self._cache: dict[str, list[dict]] = {}  # session_id -> rows
        self.hits = 0
        self.misses = 0

    def get(self, session_id: str) -> list[dict] | None:
        rows = self._cache.get(session_id)
        if rows is not None:
            self.hits += 1
        else:
            self.misses += 1
        return rows

    def put(self, session_id: str, rows: list[dict]) -> None:
        self._cache[session_id] = rows

    def __len__(self) -> int:
        return len(self._cache)


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
    session_cache: SessionCache | None = None,
    entity_enricher: object | None = None,
) -> None:
    """Ingest all haystack sessions for one LongMemEval entry.

    LongMemEval format: haystack_sessions is a list of sessions, each
    session is a list of {role, content, [has_answer]} turn dicts.
    haystack_session_ids is a parallel list of session ID strings.

    When a ``session_cache`` is provided, already-embedded session rows
    are reused from the cache instead of re-embedding. This is a major
    speedup since LongMemEval questions share many haystack sessions.
    """
    session_ids = entry["haystack_session_ids"]
    sessions = entry["haystack_sessions"]

    # LongMemEval provides per-session dates (optional)
    haystack_dates = entry.get("haystack_dates", [])

    for idx, (sid, session_turns) in enumerate(zip(session_ids, sessions)):
        # Try cache first
        if session_cache is not None:
            cached = session_cache.get(sid)
            if cached is not None:
                store.upsert(cached)
                continue

        # Extract session date for temporal anchoring (v7)
        session_date = haystack_dates[idx] if idx < len(haystack_dates) else None

        turns = []
        for turn in session_turns:
            turns.append({
                "text": turn.get("content", ""),
                "speaker": turn.get("role", None),
                "timestamp": session_date,
            })
        ingest_session(
            turns,
            session_id=sid,
            store=store,
            embedder=embedder,
            entity_enricher=entity_enricher,
        )

        # Cache the rows for this session
        if session_cache is not None:
            session_rows = [
                row for row in store.all_rows()
                if row["session_id"] == sid
            ]
            session_cache.put(sid, session_rows)

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
    reranker: Reranker | None = None,
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
        reranker=reranker,
    )

    return QuestionResult(
        question_id=question_id,
        question_type=question_type,
        stratum=_classify_stratum(question_type, question_id),
        gold_session_ids=gold_session_ids,
        retrieved_chunk_ids=result.top_ids,
    )


def pre_warm_session_cache(
    data: list[dict],
    *,
    embedder: Embedder,
    entity_enricher: object | None = None,
    verbose: bool = False,
) -> SessionCache:
    """Pre-embed all unique sessions across all questions.

    This is a massive speedup for multi-question evals: instead of
    embedding sessions on-demand (and hoping for cache hits), we scan
    all questions for unique sessions and embed them upfront. Each
    question then gets instant ingestion via cache lookup.

    For LongMemEval S (500 questions, ~24K session slots), this typically
    reduces ~24K embeddings to ~5K unique sessions, saving 5-15x wall time.
    """
    cache = SessionCache()
    seen_sessions: dict[str, list[dict]] = {}

    # Phase 1: Collect unique sessions across all questions
    for entry in data:
        for sid, session_turns in zip(
            entry["haystack_session_ids"],
            entry["haystack_sessions"],
        ):
            if sid not in seen_sessions:
                seen_sessions[sid] = session_turns

    total = len(seen_sessions)
    if verbose:
        print(f"  Pre-warming cache: {total} unique sessions to embed...", flush=True)

    # Phase 2: Embed each unique session once
    t0 = time.time()
    for i, (sid, session_turns) in enumerate(seen_sessions.items()):
        if verbose and (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = elapsed / (i + 1)
            eta = rate * (total - i - 1)
            print(f"    [{i+1}/{total}] {elapsed:.0f}s elapsed, ETA {eta:.0f}s", flush=True)

        store = InMemoryStore()
        turns = []
        for turn in session_turns:
            turns.append({
                "text": turn.get("content", ""),
                "speaker": turn.get("role", None),
            })
        ingest_session(
            turns,
            session_id=sid,
            store=store,
            embedder=embedder,
            entity_enricher=entity_enricher,
        )
        # Extract and cache the rows
        rows = list(store.all_rows())
        cache.put(sid, rows)

    if verbose:
        elapsed = time.time() - t0
        print(f"  Cache warmed: {total} sessions in {elapsed:.0f}s "
              f"({elapsed/max(total,1):.1f}s/session)", flush=True)

    return cache


def run_evaluation(
    data: list[dict],
    *,
    embedder: Embedder | None = None,
    config: PipelineConfig | None = None,
    reranker: Reranker | None = None,
    use_songline: bool = True,
    use_session_cache: bool = True,
    use_ner: bool = False,
    pre_warmed_cache: SessionCache | None = None,
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
    reranker
        Optional cross-encoder reranker. When provided, replaces the
        identity stub in the pipeline.
    use_songline
        Whether to build and use the songline graph. Default True.
    use_session_cache
        Whether to cache session embeddings across questions for speed.
        Default True. Disable for strict per-question isolation testing.
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

    session_cache: SessionCache | None = None
    if use_session_cache:
        session_cache = pre_warmed_cache if pre_warmed_cache is not None else SessionCache()
    entity_enricher = None
    if use_ner:
        from patha.query.entities import EntityEnricher
        entity_enricher = EntityEnricher()
        if verbose:
            print("  NER enrichment enabled (spaCy en_core_web_sm)", flush=True)

    report = EvalReport()
    total = len(data)
    t0 = time.time()

    for i, entry in enumerate(data):
        qid = entry["question_id"]
        qt = entry["question_type"]
        if verbose:
            elapsed = time.time() - t0
            rate = elapsed / (i + 1) if i > 0 else 0
            eta = rate * (total - i - 1) if i > 0 else 0
            cache_info = ""
            if session_cache is not None:
                cache_info = f" [cache: {len(session_cache)} sessions, {session_cache.hits}H/{session_cache.misses}M]"
            print(f"  [{i + 1}/{total}] {qid} ({qt}) {elapsed:.0f}s elapsed, ETA {eta:.0f}s{cache_info}", flush=True)

        # Fresh store per question (per LongMemEval protocol)
        store = InMemoryStore()
        bm25 = SimpleBM25()

        # Ingest this question's haystack (with optional cache + NER)
        _ingest_haystack(entry, store, embedder, bm25, session_cache, entity_enricher)

        # Optionally build songline graph
        songline_graph = None
        if use_songline:
            songline_graph = build_songline_graph(list(store.all_rows()))

        # Run retrieval
        qr = run_single_question(
            entry, store, embedder, bm25, config, songline_graph,
            reranker=reranker,
        )
        report.results.append(qr)

    if verbose and session_cache is not None:
        print(f"  Session cache final: {len(session_cache)} unique sessions, "
              f"{session_cache.hits} hits / {session_cache.misses} misses")

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
    parser.add_argument("--reranker", default="none",
                        choices=["none", "ce-mini", "ce-12"],
                        help="Reranker: none, ce-mini (MiniLM-L-6), ce-12 (MiniLM-L-12)")
    parser.add_argument("--device", default=None,
                        help="Device for models: cpu, cuda, mps, or auto (default: auto)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit to first N questions (0 = all)")
    parser.add_argument("--strata", default=None,
                        help="Comma-separated strata to include (e.g. single_session,multi_session)")
    parser.add_argument("--no-songline", action="store_true")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable session embedding cache")
    parser.add_argument("--ner", action="store_true",
                        help="Enable spaCy NER for entity-anchored views (v5/v6)")
    parser.add_argument("--pre-warm", action="store_true",
                        help="Pre-embed all unique sessions before eval (faster for large evals)")
    parser.add_argument("--views", default=None,
                        help="Comma-separated views to use (e.g. v1 or v1,v4). Default: all 7")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    print(f"Loading {args.data}...")
    data = load_longmemeval(args.data)

    # Filter by strata if requested
    if args.strata:
        wanted = set(args.strata.split(","))
        data = [e for e in data
                if _classify_stratum(e["question_type"], e.get("question_id", "")) in wanted]
        print(f"Filtered to strata {args.strata}: {len(data)} questions")

    if args.limit > 0:
        data = data[:args.limit]
    print(f"Running on {len(data)} questions")

    embedder: Embedder | None = None
    if args.embedder == "minilm":
        from patha.models.embedder_st import SentenceTransformerEmbedder
        embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2", device=args.device)
        print(f"Using MiniLM embedder (dim={embedder.dim})")
    elif args.embedder == "stella":
        from patha.models.embedder_st import SentenceTransformerEmbedder
        embedder = SentenceTransformerEmbedder("dunzhang/stella_en_400M_v5", device=args.device)
        print(f"Using Stella-400M embedder (dim={embedder.dim})")

    reranker_fn: Reranker | None = None
    if args.reranker == "ce-mini":
        from patha.retrieval.reranker import CrossEncoderReranker
        reranker_fn = CrossEncoderReranker(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            device=args.device,
        )
        print("Using cross-encoder reranker: ms-marco-MiniLM-L-6-v2")
    elif args.reranker == "ce-12":
        from patha.retrieval.reranker import CrossEncoderReranker
        reranker_fn = CrossEncoderReranker(
            "cross-encoder/ms-marco-MiniLM-L-12-v2",
            device=args.device,
        )
        print("Using cross-encoder reranker: ms-marco-MiniLM-L-12-v2")

    # Resolve embedder for pre-warming
    actual_embedder = embedder if embedder is not None else StubEmbedder(dim=64)

    # Pre-warm session cache if requested
    pre_warmed = None
    if args.pre_warm and not args.no_cache:
        entity_enricher = None
        if args.ner:
            from patha.query.entities import EntityEnricher
            entity_enricher = EntityEnricher()
        pre_warmed = pre_warm_session_cache(
            data,
            embedder=actual_embedder,
            entity_enricher=entity_enricher,
            verbose=args.verbose,
        )

    views = args.views.split(",") if args.views else None
    config = PipelineConfig(top_k=args.top_k, views=views)
    t_start = time.time()
    report = run_evaluation(
        data,
        embedder=embedder,
        config=config,
        reranker=reranker_fn,
        use_songline=not args.no_songline,
        use_session_cache=not args.no_cache,
        use_ner=args.ner,
        pre_warmed_cache=pre_warmed,
        verbose=args.verbose,
    )
    elapsed = time.time() - t_start

    summary = report.summary()
    print(f"\n=== RESULTS ({len(data)} questions, {elapsed:.0f}s) ===")
    for k, v in summary.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for sk, sv in v.items():
                print(f"    {sk}: {sv}")
        else:
            print(f"  {k}: {v}")
    print(f"\n  Avg time per question: {elapsed / max(len(data), 1):.1f}s")

    save_results(report, args.output)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
