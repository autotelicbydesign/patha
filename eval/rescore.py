"""Re-score an existing answer-eval run under a different scorer.

The LLM's answers don't depend on the scorer — so once `run_answer_eval`
has produced an outcomes file (with the model's `answer` and the `gold`
for each question), we can deterministically re-score those same answers
under any other scorer WITHOUT re-calling the LLM. This is a legitimate,
common eval operation: report the same run under exact-match, numeric,
token-overlap, and embedding-cosine to show scorer sensitivity.

It also closes an integrity gap: the README/benchmarks cited a
token-overlap number for the qwen2.5:14b run, but only the *numeric*
run had a persisted artifact. This tool regenerates the overlap artifact
from the stored answers so the number is backed and reproducible.

Usage:
    python -m eval.rescore \
        --in runs/answer_eval/ku-qwen14b-numeric.json \
        --scorer overlap \
        --out runs/answer_eval/ku-qwen14b-overlap.json

The output JSON mirrors run_answer_eval's shape (n, correct, accuracy,
by_strategy, by_question_type, outcomes) but with `correct` recomputed
under the new scorer and a `rescored_from` provenance field added.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eval.answer_eval import (
    embedding_cosine_match,
    exact_match,
    normalised_match,
    numeric_match,
    token_overlap_match,
)


def _build_scorer(name: str):
    if name == "exact":
        return exact_match
    if name == "normalised":
        return normalised_match
    if name == "numeric":
        return numeric_match()
    if name == "overlap":
        return token_overlap_match()
    if name == "embedding":
        return embedding_cosine_match()
    raise ValueError(f"unknown --scorer: {name!r}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Re-score an existing answer-eval run under a new scorer."
    )
    p.add_argument("--in", dest="in_path", required=True, type=Path,
                   help="Existing answer-eval output JSON (with outcomes).")
    p.add_argument("--scorer", required=True,
                   choices=["exact", "normalised", "numeric", "overlap",
                            "embedding"],
                   help="Scorer to re-apply to the stored answers.")
    p.add_argument("--out", required=True, type=Path,
                   help="Where to write the re-scored artifact.")
    args = p.parse_args(argv)

    src = json.loads(args.in_path.read_text())
    outcomes = src.get("outcomes")
    if not outcomes:
        sys.exit(f"no outcomes in {args.in_path}; nothing to re-score")

    scorer = _build_scorer(args.scorer)

    correct = 0
    by_strategy: dict[str, list[int]] = {}
    by_qtype: dict[str, list[int]] = {}
    new_outcomes = []
    for o in outcomes:
        ok = bool(scorer(o["answer"], str(o["gold"])))
        if ok:
            correct += 1
        strat = o.get("strategy", "?")
        ent = by_strategy.setdefault(strat, [0, 0])
        ent[1] += 1
        if ok:
            ent[0] += 1
        # question_type isn't stored per-outcome in run_answer_eval; skip
        # unless present
        qt = o.get("question_type")
        if qt is not None:
            qent = by_qtype.setdefault(qt, [0, 0])
            qent[1] += 1
            if ok:
                qent[0] += 1
        new_outcomes.append({**o, "correct": ok})

    n = len(outcomes)
    payload = {
        "data": src.get("data"),
        "llm": src.get("llm"),
        "scorer": args.scorer,
        "rescored_from": str(args.in_path),
        "n": n,
        "correct": correct,
        "accuracy": correct / n if n else 0.0,
        "by_strategy": {
            k: {"correct": c, "total": t} for k, (c, t) in by_strategy.items()
        },
        "by_question_type": {
            k: {"correct": c, "total": t} for k, (c, t) in by_qtype.items()
        },
        "outcomes": new_outcomes,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))

    print(f"re-scored {n} answers from {args.in_path}")
    print(f"  scorer={args.scorer}  {correct}/{n} = {payload['accuracy']:.3f}")
    for k, (c, t) in sorted(by_strategy.items()):
        print(f"  {k:14s}  {c}/{t} = {c/t:.3f}")
    print(f"wrote: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
