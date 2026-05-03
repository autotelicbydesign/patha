"""Phase 3 runner — end-to-end answer evaluation.

Wires the `eval/answer_eval.py` engine (LLM + prompt + scorer) to the
LongMemEval haystack-ingest pattern from `eval/longmemeval_belief.py`.

For each question in the input JSON:

    1. Build a fresh `patha.Memory`
    2. Ingest the haystack USER turns (chronological, with session dates)
    3. `memory.recall(question)` → render prompt → call LLM → score

Aggregates one accuracy number per (LLM × scorer) pair, plus per-question
outcomes (gold, candidate, correct, strategy used by recall, summary
token count). Optionally writes a JSON results file.

Usage::

    python -m eval.run_answer_eval \\
        --data data/longmemeval_ku_78.json \\
        --llm null \\
        --scorer numeric \\
        --max-questions 10 \\
        --output runs/answer_eval/ku-null-numeric.json

LLM choices:
    null     — deterministic NullTemplateLLM (no network, free)
    claude   — Anthropic Messages API (needs ANTHROPIC_API_KEY env)
    ollama   — local Ollama (default model llama3.2:3b, override --ollama-model)

Scorer choices:
    normalised — case+punct insensitive
    numeric    — numeric within 5%, falls back to normalised on non-numeric gold
    overlap    — token overlap ≥ 60% (LongMemEval-style)
    embedding  — cosine ≥ 0.85 (lazy MiniLM)
    judge      — LLM-as-judge (uses --judge-llm, default same model as --llm)

This module reuses the haystack-ingest helpers from `longmemeval_belief.py`
rather than duplicating them.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import patha
from patha.chunking.propositionizer import propositionize

from eval.answer_eval import (
    AnswerEvalConfig,
    ClaudeLLM,
    NullTemplateLLM,
    OllamaLLM,
    embedding_cosine_match,
    llm_judge_match,
    normalised_match,
    numeric_match,
    run_answer_eval,
    token_overlap_match,
)
from eval.longmemeval_belief import (
    _parse_lme_date,
    _relevant_to_question,
    _tokens,
)


# ─── Defaults ────────────────────────────────────────────────────────


DEFAULT_TEMPLATE = (
    "Answer the user's question using only the memory context below.\n"
    "Be concise — one short sentence or a single value. If a computed\n"
    "answer is in the 'Computed' line, use it verbatim.\n"
    "\n"
    "Question: {question}\n"
    "\n"
    "Memory:\n"
    "{summary}\n"
    "\n"
    "Computed: {ganita}\n"
    "\n"
    "Answer:"
)


# ─── Memory factory ─────────────────────────────────────────────────


def _build_memory_for_question(
    q: dict, *, ingest_full: bool, store_path: Path | None = None,
) -> patha.Memory:
    """Build a fresh `patha.Memory`, ingest the question's haystack
    USER turns chronologically, and return it ready for `recall()`.

    Mirrors the ingest protocol used by `eval/longmemeval_belief.py`
    so the two evaluators see equivalent stores. Phase 1 is disabled
    (this is a recall-quality measurement, not a retrieval-pipeline
    measurement — the haystack is small enough per-question that the
    belief layer covers it directly).
    """
    keywords = set(_tokens(q["question"])) | set(_tokens(q["answer"]))
    keywords = {k for k in keywords if len(k) >= 3}

    mem = patha.Memory(
        path=store_path or Path(f"/tmp/patha_phase3_{q['question_id']}.jsonl"),
        enable_phase1=False,
    )

    session_ids = q["haystack_session_ids"]
    session_dates = [_parse_lme_date(d) for d in q["haystack_dates"]]
    sessions = q["haystack_sessions"]
    order = sorted(range(len(session_ids)), key=lambda i: session_dates[i])

    for idx in order:
        sid = session_ids[idx]
        date = session_dates[idx]
        sess = sessions[idx]
        for turn_idx, turn in enumerate(sess):
            if turn.get("role") != "user":
                continue
            text = turn.get("content", "")
            if not text.strip():
                continue
            props = propositionize(text, session_id=sid, turn_idx=turn_idx)
            for p in props:
                if not ingest_full and not _relevant_to_question(p.text, keywords):
                    continue
                mem.remember(p.text, asserted_at=date)

    return mem


# ─── LLM / scorer factories ─────────────────────────────────────────


def _build_llm(name: str, *, ollama_model: str, claude_model: str):
    if name == "null":
        return NullTemplateLLM()
    if name == "claude":
        return ClaudeLLM(model=claude_model)
    if name == "ollama":
        return OllamaLLM(model=ollama_model)
    raise ValueError(f"unknown --llm: {name!r}")


def _build_scorer(name: str, *, judge_llm=None):
    if name == "normalised":
        return normalised_match
    if name == "numeric":
        return numeric_match()
    if name == "overlap":
        return token_overlap_match()
    if name == "embedding":
        return embedding_cosine_match()
    if name == "judge":
        if judge_llm is None:
            raise ValueError("--scorer judge requires --judge-llm")
        return llm_judge_match(judge_llm)
    raise ValueError(f"unknown --scorer: {name!r}")


# ─── Main ────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Phase 3 end-to-end answer evaluation.",
    )
    p.add_argument("--data", required=True, type=Path,
                   help="Path to LongMemEval-shaped JSON (list of questions).")
    p.add_argument("--llm", default="null",
                   choices=["null", "claude", "ollama"],
                   help="Which LLM to use as the answer model.")
    p.add_argument("--claude-model", default="claude-sonnet-4-20250514",
                   help="Anthropic model id when --llm=claude.")
    p.add_argument("--ollama-model", default="llama3.2:3b",
                   help="Ollama model name when --llm=ollama.")
    p.add_argument("--scorer", default="numeric",
                   choices=["normalised", "numeric", "overlap",
                            "embedding", "judge"],
                   help="How to compare candidate vs gold.")
    p.add_argument("--judge-llm", default=None,
                   choices=[None, "null", "claude", "ollama"],
                   help="Judge LLM when --scorer=judge (defaults to --llm).")
    p.add_argument("--template", default=None,
                   help="Prompt template path (file). Defaults to a built-in.")
    p.add_argument("--max-questions", type=int, default=None,
                   help="Stop after N questions (for fast smoke runs).")
    p.add_argument("--ingest-full", action="store_true",
                   help="Ingest every USER proposition in the haystack "
                   "(default: only those sharing a content word with Q+A).")
    p.add_argument("--include-history", action="store_true",
                   help="Pass include_history=True to memory.recall().")
    p.add_argument("--output", type=Path, default=None,
                   help="Write per-question + summary JSON here.")
    p.add_argument("--verbose", action="store_true",
                   help="Print progress per question.")
    args = p.parse_args(argv)

    # Load benchmark
    questions: list[dict] = json.loads(args.data.read_text())
    if not isinstance(questions, list):
        sys.exit(f"expected a list at top level of {args.data}")
    if args.max_questions:
        questions = questions[: args.max_questions]
    n = len(questions)

    # Prompt template
    template = (
        args.template and Path(args.template).read_text()
    ) or DEFAULT_TEMPLATE

    # LLM + scorer
    llm = _build_llm(
        args.llm, ollama_model=args.ollama_model, claude_model=args.claude_model,
    )
    judge_llm = None
    if args.scorer == "judge":
        judge_name = args.judge_llm or args.llm
        judge_llm = _build_llm(
            judge_name, ollama_model=args.ollama_model,
            claude_model=args.claude_model,
        )
    scorer = _build_scorer(args.scorer, judge_llm=judge_llm)

    cfg = AnswerEvalConfig(
        llm=llm, prompt_template=template, scorer=scorer,
        include_history=args.include_history,
    )

    # Run with per-question pre-ingest. The factory in answer_eval is
    # global, but we want one-memory-per-question with that question's
    # haystack pre-ingested — handled via the `_ingested_memory` hook.
    print(f"phase 3 runner: {n} questions, llm={args.llm}, "
          f"scorer={args.scorer}", file=sys.stderr)

    started = time.time()
    enriched = []
    for i, q in enumerate(questions, 1):
        if args.verbose:
            print(f"  [{i}/{n}] ingesting haystack for {q['question_id']}",
                  file=sys.stderr)
        mem = _build_memory_for_question(q, ingest_full=args.ingest_full)
        q2 = dict(q)
        q2["_ingested_memory"] = mem
        enriched.append(q2)

    # The factory is unused since each q carries its own pre-ingested memory
    report = run_answer_eval(
        enriched, memory_factory=lambda: None, config=cfg,
    )
    elapsed = time.time() - started

    # Aggregate by question_type if present (LongMemEval-shaped)
    by_qtype: dict[str, list[int]] = {}
    for q, o in zip(questions, report.outcomes):
        qt = q.get("question_type", "?")
        ent = by_qtype.setdefault(qt, [0, 0])
        ent[1] += 1
        if o.correct:
            ent[0] += 1

    # Print headline
    print(f"\nresults: {report.correct}/{report.n} = "
          f"{report.accuracy:.3f}  ({elapsed:.1f}s)")
    print("by strategy:")
    for strat, (c, t) in sorted(report.by_strategy().items()):
        print(f"  {strat:18s}  {c}/{t}  = {c/t:.3f}")
    if by_qtype:
        print("by question_type:")
        for qt, (c, t) in sorted(by_qtype.items()):
            print(f"  {qt:18s}  {c}/{t}  = {c/t:.3f}")

    # Optional JSON dump
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "data": str(args.data),
            "llm": args.llm,
            "scorer": args.scorer,
            "n": report.n,
            "correct": report.correct,
            "accuracy": report.accuracy,
            "elapsed_seconds": elapsed,
            "by_strategy": {
                k: {"correct": c, "total": t}
                for k, (c, t) in report.by_strategy().items()
            },
            "by_question_type": {
                k: {"correct": c, "total": t}
                for k, (c, t) in by_qtype.items()
            },
            "outcomes": [asdict(o) for o in report.outcomes],
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        args.output.write_text(json.dumps(payload, indent=2, default=str))
        print(f"\nwrote: {args.output}")

    return 0 if report.correct == report.n else 1


if __name__ == "__main__":
    sys.exit(main())
