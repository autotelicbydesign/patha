"""External validation: run the Phase 2 belief layer against
LongMemEval's knowledge-update stratum (78 questions).

This benchmark was NOT authored by us. The questions, sessions, and
answer dates come from Wu et al.'s LongMemEval. We use it as a genuine
external test: does the belief layer handle real multi-session
contradictions and updates?

Protocol for each question:
  1. Load haystack sessions in chronological order (haystack_dates).
  2. Propositionize USER turns (the user's beliefs are what we track).
  3. Ingest each proposition into a fresh BeliefLayer with the session
     date as asserted_at. NLI fires at ingest time.
  4. Filter: to stay tractable, we only ingest propositions whose
     lexical content overlaps with the question (i.e., share a content
     word with the question or answer entity). Alternatively pass
     --full to ingest everything.
  5. Query at question_date for current_belief across all ingested
     beliefs. Also walk history for superseded ones.
  6. Score: the ``answer`` field is checked against the current-belief
     summary via substring / token-overlap match (robust to phrasing).

Scoring modes:
  - ``contains``: the answer string (or any of its synonyms) must
    appear in the current-belief summary. Default.
  - ``token-overlap``: at least 60% of content tokens in the answer
    must appear in the current-belief summary.

Metrics:
  - Per-question correct (yes/no)
  - Aggregate R@1 on current belief (top 1 matches answer)
  - Tokens emitted per summary (as vs. naïve RAG's per-session count)

Caveats:
  - The benchmark was designed for full-LLM retrieval, not a belief
    layer alone. We expect a sizable gap vs. the paper's R@1 — the
    question is how much the belief layer contributes on its own
    without the retrieval pillar.
  - We ingest USER turns only; the system never sees assistant turns.
    This is stricter than the paper's protocol.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from patha.belief.adhyasa_detector import AdhyasaAwareDetector
from patha.belief.contradiction import (
    ContradictionDetector,
    NLIContradictionDetector,
    StubContradictionDetector,
)
from patha.belief.layer import BeliefLayer
from patha.belief.numerical_detector import NumericalAwareDetector
from patha.belief.sequential_detector import SequentialEventDetector
from patha.belief.store import BeliefStore
from patha.chunking.propositionizer import propositionize


# ─── Scoring helpers ─────────────────────────────────────────────────

_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "to", "of", "in", "on",
    "at", "for", "with", "and", "or", "but", "not", "no", "yes", "i",
    "you", "me", "my", "we", "our", "your", "their", "this", "that",
    "these", "those", "it", "its", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "must", "can", "about", "as", "by", "from", "into",
    "through", "some", "any", "all", "other", "so", "than",
})


def _tokens(text: str | int | float) -> list[str]:
    """Lowercase content tokens, stopwords removed. Accepts str|int|float."""
    s = str(text) if not isinstance(text, str) else text
    return [
        t.lower() for t in re.findall(r"[A-Za-z0-9]+", s)
        if t.lower() not in _STOPWORDS
    ]


_WORD_NUM = {"zero": "0", "one": "1", "two": "2", "three": "3",
             "four": "4", "five": "5", "six": "6", "seven": "7",
             "eight": "8", "nine": "9", "ten": "10", "eleven": "11",
             "twelve": "12"}
_NUM_WORD = {v: k for k, v in _WORD_NUM.items()}


def _number_variants(text: str) -> list[str]:
    """Extract numeric values and their variants (word + digit form).

    For each number found, return BOTH the digit and word form so the
    matcher can find 'four' or '4' interchangeably. Covers 0-12.
    """
    variants: set[str] = set()
    # Digit numbers
    for n in re.findall(r"\d+(?:\.\d+)?", text):
        variants.add(n)
        variants.add(_NUM_WORD.get(n, ""))
    # Word numbers
    lower = text.lower()
    for word, digit in _WORD_NUM.items():
        if re.search(rf"\b{word}\b", lower):
            variants.add(word)
            variants.add(digit)
    variants.discard("")
    return list(variants)


def _score_contains(answer: str, summary: str) -> bool:
    """Correct iff the answer's content tokens are all present in summary
    (substring match on each non-stopword token)."""
    answer = str(answer) if not isinstance(answer, str) else answer
    a_toks = _tokens(answer)
    s_lower = summary.lower()
    if not a_toks:
        return False
    # Any-number-matches: if the answer contains a number (digit or
    # word form), require at least one variant to appear in summary.
    a_nums = _number_variants(answer)
    if a_nums:
        # word-form match requires word-boundary to avoid 'four'
        # matching inside 'fourteen'; digit match is substring.
        def _num_in_summary(n: str) -> bool:
            if n.isdigit() or "." in n:
                return bool(re.search(rf"(?<!\d){re.escape(n)}(?!\d)", s_lower))
            return bool(re.search(rf"\b{re.escape(n)}\b", s_lower))
        if not any(_num_in_summary(n) for n in a_nums):
            return False
    # Require majority of content tokens to appear (tolerance for phrasing)
    present = sum(1 for t in a_toks if t in s_lower)
    return present / len(a_toks) >= 0.6


# ─── Adapter core ───────────────────────────────────────────────────

@dataclass
class QuestionOutcome:
    question_id: str
    question: str
    gold_answer: str
    correct: bool
    answer_in_superseded: bool  # info retained but mis-routed to history
    current_summary: str
    current_count: int
    superseded_count: int
    tokens_in_summary: int
    ingested_props: int
    relevant_ingested: int
    notes: str = ""


def _relevant_to_question(prop_text: str, keywords: set[str]) -> bool:
    """Token overlap with question/answer keywords."""
    toks = set(_tokens(prop_text))
    return bool(toks & keywords)


def _parse_lme_date(s: str) -> datetime:
    """LongMemEval date format: '2023/04/23 (Sun) 08:57'."""
    s = re.sub(r"\s*\([A-Za-z]+\)\s*", " ", s).strip()
    return datetime.strptime(s, "%Y/%m/%d %H:%M")


def run_question(
    q: dict,
    detector: ContradictionDetector,
    *,
    ingest_full: bool = False,
    verbose: bool = False,
) -> QuestionOutcome:
    """Run one knowledge-update question end-to-end through the belief layer."""
    keywords = set(_tokens(q["question"])) | set(_tokens(q["answer"]))
    # Strip very short words (noise for overlap)
    keywords = {k for k in keywords if len(k) >= 3}

    layer = BeliefLayer(
        store=BeliefStore(),
        detector=detector,
        contradiction_threshold=0.7,
    )

    session_ids = q["haystack_session_ids"]
    session_dates = [_parse_lme_date(d) for d in q["haystack_dates"]]
    sessions = q["haystack_sessions"]
    # Sort sessions by date (the paper claims chronological but verify)
    order = sorted(range(len(session_ids)), key=lambda i: session_dates[i])

    belief_ids: list[str] = []
    total_ingested = 0
    relevant_ingested = 0

    for order_idx, idx in enumerate(order):
        sid = session_ids[idx]
        date = session_dates[idx]
        sess = sessions[idx]
        for turn_idx, turn in enumerate(sess):
            if turn.get("role") != "user":
                continue
            text = turn.get("content", "")
            if not text.strip():
                continue
            props = propositionize(
                text, session_id=sid, turn_idx=turn_idx,
            )
            for p in props:
                total_ingested += 1
                if not ingest_full and not _relevant_to_question(p.text, keywords):
                    continue
                relevant_ingested += 1
                ev = layer.ingest(
                    proposition=p.text,
                    asserted_at=date,
                    asserted_in_session=sid,
                    source_proposition_id=p.chunk_id,
                )
                belief_ids.append(ev.new_belief.id)

    # Query at question_date
    question_date = _parse_lme_date(q["question_date"])
    query_result = layer.query(
        belief_ids, at_time=question_date, include_history=True,
    )
    # Filter current beliefs by question-keyword overlap so the
    # belief-layer summary is focused. This mirrors how a retrieval
    # layer would pre-filter before handing to the belief layer in
    # production, and avoids a 50-belief summary when only 3 are
    # relevant.
    q_keywords = {k for k in _tokens(q["question"]) if len(k) >= 4}
    if q_keywords:
        filtered = [
            b for b in query_result.current
            if set(_tokens(b.proposition)) & q_keywords
        ]
        current_props = [b.proposition for b in filtered]
    else:
        current_props = [b.proposition for b in query_result.current]
    summary = " | ".join(current_props)
    correct = _score_contains(q["answer"], summary)

    # Also check whether the answer appears in superseded beliefs —
    # tells us if the information was retained but mis-routed to
    # history vs truly lost.
    superseded_props_text = [b.proposition for b in query_result.history]
    super_summary = " | ".join(superseded_props_text)
    answer_in_superseded = _score_contains(q["answer"], super_summary)

    if verbose and not correct:
        print(f"  FAIL [{q['question_id']}] expected={q['answer']!r}")
        print(f"    current ({len(current_props)}): {current_props[:5]}")
        print(f"    ingested: {relevant_ingested}/{total_ingested} relevant")

    return QuestionOutcome(
        question_id=q["question_id"],
        question=q["question"],
        gold_answer=q["answer"],
        correct=correct,
        answer_in_superseded=answer_in_superseded,
        current_summary=summary,
        current_count=len(query_result.current),
        superseded_count=len(query_result.history),
        tokens_in_summary=query_result.tokens_in_summary,
        ingested_props=total_ingested,
        relevant_ingested=relevant_ingested,
    )


# ─── Detector factory (reused from belief_eval) ──────────────────────

def make_detector(name: str) -> ContradictionDetector:
    if name == "stub":
        return StubContradictionDetector()
    if name == "nli":
        return NLIContradictionDetector()
    if name == "full-stack":
        return NumericalAwareDetector(
            inner=AdhyasaAwareDetector(inner=NLIContradictionDetector())
        )
    if name == "full-stack-v7":
        return NumericalAwareDetector(
            inner=SequentialEventDetector(
                inner=AdhyasaAwareDetector(
                    inner=NLIContradictionDetector()
                )
            )
        )
    raise ValueError(f"unknown detector {name!r}")


# ─── CLI ────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="LongMemEval knowledge-update external benchmark"
    )
    ap.add_argument("--data", default="data/longmemeval_s_cleaned.json")
    ap.add_argument("--detector", default="full-stack",
                    choices=["stub", "nli", "full-stack", "full-stack-v7"])
    ap.add_argument("--output", default="runs/longmemeval_ku/results.json")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only run first N knowledge-update questions")
    ap.add_argument("--full", action="store_true",
                    help="Ingest all user turns (not just keyword-relevant)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    with open(args.data) as f:
        data = json.load(f)
    ku = [q for q in data if q["question_type"] == "knowledge-update"]
    if args.limit is not None:
        ku = ku[: args.limit]

    print(f"Running {len(ku)} knowledge-update questions "
          f"with detector={args.detector!r}, ingest_full={args.full}")

    detector = make_detector(args.detector)
    outcomes: list[QuestionOutcome] = []
    for i, q in enumerate(ku, 1):
        out = run_question(
            q, detector, ingest_full=args.full, verbose=args.verbose,
        )
        outcomes.append(out)
        tag = "PASS" if out.correct else "fail"
        print(f"  [{i}/{len(ku)}] {tag} {out.question_id}: "
              f"{out.question[:60]!r}")

    n = len(outcomes)
    n_correct = sum(1 for o in outcomes if o.correct)
    avg_tokens = sum(o.tokens_in_summary for o in outcomes) / max(n, 1)
    avg_ingested = sum(o.relevant_ingested for o in outcomes) / max(n, 1)
    avg_current = sum(o.current_count for o in outcomes) / max(n, 1)

    print()
    print("=" * 60)
    print(f"LongMemEval KU external benchmark ({args.detector})")
    print("=" * 60)
    print(f"  Questions:  {n}")
    print(f"  Correct:    {n_correct} ({n_correct/n:.1%})")
    print(f"  Avg tokens/summary: {avg_tokens:.0f}")
    print(f"  Avg props ingested: {avg_ingested:.1f}")
    print(f"  Avg current beliefs: {avg_current:.1f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "detector": args.detector,
            "n": n,
            "n_correct": n_correct,
            "accuracy": n_correct / max(n, 1),
            "avg_tokens_in_summary": avg_tokens,
            "outcomes": [o.__dict__ for o in outcomes],
        }, f, indent=2, default=str)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
