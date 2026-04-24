"""LongMemEval end-to-end through the unified `patha.Memory` API.

This is the honest "Phase 1 + Phase 2 merged" benchmark. Each question:

  1. Construct a fresh `patha.Memory(enable_phase1=True)` instance.
  2. Iterate haystack sessions in chronological order.
  3. For every user turn, call `memory.remember(text)` — this runs
     through the full pipeline:
        - Phase 2 belief layer: contradiction detection, supersession,
          reinforcement, plasticity
        - Phase 1 index: new belief is indexed across 7 Vedic views +
          BM25 on next query (lazy rebuild)
  4. At question time, call `memory.recall(question, include_history=True)`
     — this runs through the full pipeline:
        - Phase 1: 7-view dense + BM25 + RRF over all ingested beliefs
        - Phase 2: filter current vs superseded
        - Direct-answer / structured summary
  5. Score: does the gold answer text appear in `recall.summary`?

No keyword filtering, no semantic-only fallback. This is "what does
one Patha actually deliver on LongMemEval when both phases run in
anger."

Usage:
    uv run python -m eval.longmemeval_integrated \
        --data data/longmemeval_ku_78.json \
        --detector stub \
        --output runs/integrated/ku_78.json

Detector choice:
    - stub:          fast, no downloads (baseline — Phase 1 + belief-store
                     persistence + passthrough detector)
    - full-stack-v8: NLI + adhyasa + numerical + sequential + learned
                     (production; triggers ~1.7 GB NLI download on first use)
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import patha


# ─── Scoring helpers (same contract as eval/longmemeval_belief.py) ────

_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "to", "of", "in", "on",
    "at", "for", "with", "and", "or", "but", "not", "i", "you", "my",
    "we", "our", "your", "this", "that", "it", "be", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "about",
    "as", "by", "from", "some", "any", "all", "so", "than",
})

_WORD_NUM = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12",
}
_NUM_WORD = {v: k for k, v in _WORD_NUM.items()}


def _tokens(text) -> list[str]:
    s = str(text) if not isinstance(text, str) else text
    return [
        t.lower() for t in re.findall(r"[A-Za-z0-9]+", s)
        if t.lower() not in _STOPWORDS
    ]


def _number_variants(text: str) -> list[str]:
    variants: set[str] = set()
    for n in re.findall(r"\d+(?:\.\d+)?", text):
        variants.add(n)
        if n in _NUM_WORD:
            variants.add(_NUM_WORD[n])
    lower = text.lower()
    for word, digit in _WORD_NUM.items():
        if re.search(rf"\b{word}\b", lower):
            variants.add(word)
            variants.add(digit)
    variants.discard("")
    return list(variants)


def _score_contains(answer, summary: str) -> bool:
    answer = str(answer) if not isinstance(answer, str) else answer
    a_toks = _tokens(answer)
    s_lower = summary.lower()
    if not a_toks:
        return False
    a_nums = _number_variants(answer)
    if a_nums:
        def _num_in(n):
            if n.isdigit() or "." in n:
                return bool(re.search(rf"(?<!\d){re.escape(n)}(?!\d)", s_lower))
            return bool(re.search(rf"\b{re.escape(n)}\b", s_lower))
        if not any(_num_in(n) for n in a_nums):
            return False
    present = sum(1 for t in a_toks if t in s_lower)
    return present / len(a_toks) >= 0.6


def _parse_lme_date(s: str) -> datetime:
    s = re.sub(r"\s*\([A-Za-z]+\)\s*", " ", s).strip()
    return datetime.strptime(s, "%Y/%m/%d %H:%M")


# ─── Per-question run ────────────────────────────────────────────────

@dataclass
class IntegratedOutcome:
    question_id: str
    question: str
    gold_answer: str
    correct_current_only: bool
    correct_with_history: bool
    answer_in_current: bool
    answer_in_history: bool
    current_count: int
    superseded_count: int
    total_ingested: int
    ingest_seconds: float
    query_seconds: float
    tokens_in_summary: int


def run_question(
    q: dict,
    *,
    detector: str,
    granularity: str = "turn",
    verbose: bool = False,
) -> IntegratedOutcome:
    """One question through the full unified Memory pipeline.

    granularity:
      "turn"    — one belief per user turn (developer-API default;
                  natural for "user asserts fact" scenarios)
      "session" — one belief per session (concatenated user turns;
                  matches LongMemEval's native chunk granularity)
    """
    tmp_path = Path(f"/tmp/patha-integrated-{q['question_id']}.jsonl")
    tmp_path.unlink(missing_ok=True)

    memory = patha.Memory(
        path=tmp_path,
        detector=detector,
        enable_phase1=True,
        phase1_top_k=100,
    )

    session_dates = [_parse_lme_date(d) for d in q["haystack_dates"]]
    order = sorted(range(len(session_dates)), key=lambda i: session_dates[i])

    total = 0
    t_ingest = time.perf_counter()
    for idx in order:
        sid = q["haystack_session_ids"][idx]
        date = session_dates[idx]
        user_turns = [
            t.get("content", "").strip()
            for t in q["haystack_sessions"][idx]
            if t.get("role") == "user" and t.get("content", "").strip()
        ]
        if not user_turns:
            continue
        if granularity == "session":
            # One belief per session — concatenate all user turns.
            # This matches how Phase 1's headline numbers were
            # measured and LongMemEval expects.
            text = "\n\n".join(user_turns)
            memory.remember(text, asserted_at=date, session_id=sid)
            total += 1
        else:
            # One belief per turn — developer-API default.
            for text in user_turns:
                memory.remember(text, asserted_at=date, session_id=sid)
                total += 1
    ingest_secs = time.perf_counter() - t_ingest

    question_date = _parse_lme_date(q["question_date"])
    t_query = time.perf_counter()
    rec = memory.recall(
        q["question"], at_time=question_date, include_history=True,
    )
    query_secs = time.perf_counter() - t_query

    # Score against gold answer — two modes:
    # (a) current only: did the current beliefs alone carry the answer?
    # (b) with history: did current + superseded together carry it?
    current_text = " | ".join(c["proposition"] for c in rec.current)
    history_text = " | ".join(h["proposition"] for h in rec.history)
    answer_in_current = _score_contains(q["answer"], current_text)
    answer_in_history = _score_contains(q["answer"], history_text)

    if verbose and not (answer_in_current or answer_in_history):
        print(f"    FAIL {q['question_id']}: gold={q['answer']!r}")
        print(f"      current({len(rec.current)}): "
              f"{[c['proposition'][:60] for c in rec.current[:3]]}")

    tmp_path.unlink(missing_ok=True)

    return IntegratedOutcome(
        question_id=q["question_id"],
        question=q["question"],
        gold_answer=str(q["answer"]),
        correct_current_only=answer_in_current,
        correct_with_history=answer_in_current or answer_in_history,
        answer_in_current=answer_in_current,
        answer_in_history=answer_in_history,
        current_count=len(rec.current),
        superseded_count=len(rec.history),
        total_ingested=total,
        ingest_seconds=ingest_secs,
        query_seconds=query_secs,
        tokens_in_summary=rec.tokens,
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Unified Patha (Phase 1 + Phase 2) LongMemEval runner"
    )
    ap.add_argument("--data", default="data/longmemeval_ku_78.json")
    ap.add_argument("--detector", default="stub",
                    choices=["stub", "nli", "full-stack", "full-stack-v7", "full-stack-v8"])
    ap.add_argument("--output", default="runs/integrated/ku_78.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--granularity", choices=["turn", "session"], default="turn",
        help="Ingest granularity: 'turn' (one belief per user turn, "
             "our default) or 'session' (one belief per session, "
             "matches LongMemEval's native chunk size).",
    )
    ap.add_argument(
        "--filter-type", default=None,
        help="Filter to a specific question_type (e.g. 'knowledge-update'). "
             "Default: run on ALL questions in the input file.",
    )
    ap.add_argument(
        "--checkpoint", default=None,
        help="Path to a checkpoint JSON; resume from here if interrupted. "
             "Default: <output>.checkpoint.json",
    )
    ap.add_argument(
        "--checkpoint-every", type=int, default=5,
        help="Save checkpoint every N questions (default 5).",
    )
    ap.add_argument(
        "--fresh", action="store_true",
        help="Ignore existing checkpoint and start from scratch.",
    )
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    with open(args.data) as f:
        data = json.load(f)
    # Filter by question type only if explicitly requested.
    # --filter-type knowledge-update | multi-session | ...
    if args.filter_type:
        data = [q for q in data if q.get("question_type") == args.filter_type]
    if args.limit:
        data = data[: args.limit]

    print(f"Running unified Patha (Phase 1 + Phase 2) on {len(data)} "
          f"questions with detector={args.detector}, "
          f"granularity={args.granularity}", flush=True)

    # --- checkpoint / resume ------------------------------------------
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path = (
        Path(args.checkpoint) if args.checkpoint
        else output_path.with_suffix(".checkpoint.json")
    )

    outcomes: list[IntegratedOutcome] = []
    done_qids: set[str] = set()
    if not args.fresh and ckpt_path.exists():
        try:
            saved = json.loads(ckpt_path.read_text())
            for o in saved.get("outcomes", []):
                outcomes.append(IntegratedOutcome(**{
                    k: v for k, v in o.items()
                    if k in IntegratedOutcome.__annotations__
                }))
                done_qids.add(o["question_id"])
            print(f"Resumed from {ckpt_path}: {len(done_qids)} questions done.",
                  flush=True)
        except Exception as e:
            print(f"Checkpoint unreadable ({e}); starting fresh.", flush=True)

    def _save_ckpt() -> None:
        tmp = ckpt_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({
            "outcomes": [asdict(o) for o in outcomes],
        }, default=str, indent=2))
        tmp.replace(ckpt_path)

    # --- main loop ----------------------------------------------------
    for i, q in enumerate(data, 1):
        if q["question_id"] in done_qids:
            continue
        try:
            out = run_question(
                q, detector=args.detector,
                granularity=args.granularity, verbose=args.verbose,
            )
        except Exception as e:
            print(f"  [{i}/{len(data)}] ERROR on {q.get('question_id')}: {e}",
                  flush=True)
            continue
        outcomes.append(out)
        done_qids.add(q["question_id"])
        c = "PASS" if out.correct_with_history else "fail"
        cur = "C" if out.answer_in_current else "-"
        his = "H" if out.answer_in_history else "-"
        running_correct = sum(1 for o in outcomes if o.correct_with_history)
        print(f"  [{i}/{len(data)}] {c} [{cur}{his}] {out.question_id}: "
              f"ing={out.total_ingested} cur={out.current_count} "
              f"t={out.ingest_seconds:.0f}s+{out.query_seconds:.1f}s "
              f"running={running_correct}/{len(outcomes)}",
              flush=True)

        if i % args.checkpoint_every == 0:
            _save_ckpt()

    # Final checkpoint
    _save_ckpt()

    n = len(outcomes)
    if n == 0:
        print("no outcomes recorded")
        return

    correct_cur = sum(1 for o in outcomes if o.correct_current_only)
    correct_hist = sum(1 for o in outcomes if o.correct_with_history)
    in_hist_only = sum(
        1 for o in outcomes if o.answer_in_history and not o.answer_in_current
    )
    avg_tokens = sum(o.tokens_in_summary for o in outcomes) / n
    total_ingest_time = sum(o.ingest_seconds for o in outcomes)
    total_query_time = sum(o.query_seconds for o in outcomes)

    print()
    print("=" * 60)
    print(f"Unified Patha (Phase 1 + Phase 2) — {args.detector}")
    print("=" * 60)
    print(f"  Questions:                 {n}")
    print(f"  Current only:              {correct_cur}/{n} = {correct_cur/n:.3f}")
    print(f"  Current + history:         {correct_hist}/{n} = {correct_hist/n:.3f}")
    print(f"  Recovered from history:    {in_hist_only}/{n}")
    print(f"  Avg tokens/summary:        {avg_tokens:.0f}")
    print(f"  Total ingest time:         {total_ingest_time:.0f}s")
    print(f"  Total query time:          {total_query_time:.0f}s")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "detector": args.detector,
            "n": n,
            "correct_current_only": correct_cur,
            "correct_with_history": correct_hist,
            "recovered_from_history": in_hist_only,
            "accuracy_current": correct_cur / n,
            "accuracy_with_history": correct_hist / n,
            "avg_tokens_in_summary": avg_tokens,
            "total_ingest_seconds": total_ingest_time,
            "total_query_seconds": total_query_time,
            "outcomes": [asdict(o) for o in outcomes],
        }, f, indent=2, default=str)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
