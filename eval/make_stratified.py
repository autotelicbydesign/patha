"""Create stratified samples from LongMemEval-S for reproducible eval.

Stratified = keeps the per-stratum proportions of the full 500q set:
  single_session     156/500 = 31.2%
  multi_session      133/500 = 26.6%
  temporal_reasoning 133/500 = 26.6%
  knowledge_update    78/500 = 15.6%

A 300q stratified sample targets:
  single_session      94 (from 156)
  multi_session       80 (from 133)
  temporal_reasoning  80 (from 133)
  knowledge_update    46 (from  78)
  total              300

Uses a fixed seed so the sample is reproducible.

Usage:
    uv run python -m eval.make_stratified --n 300 \\
        --source data/longmemeval_s_cleaned.json \\
        --output data/longmemeval_s_300q.json
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


STRATUM_KEY = "question_type"

STRATUM_MAP = {
    "single-session-user": "single_session",
    "single-session-assistant": "single_session",
    "single-session-preference": "single_session",
    "multi-session": "multi_session",
    "temporal-reasoning": "temporal_reasoning",
    "knowledge-update": "knowledge_update",
}


def _stratum_of(q: dict) -> str:
    raw = q.get(STRATUM_KEY) or q.get("stratum") or "unknown"
    return STRATUM_MAP.get(raw, raw.replace("-", "_"))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Build a stratified LongMemEval sample")
    ap.add_argument("--source", default="data/longmemeval_s_cleaned.json")
    ap.add_argument("--n", type=int, required=True,
                    help="Target sample size")
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    with open(args.source) as f:
        data = json.load(f)

    counts = Counter(_stratum_of(q) for q in data)
    print(f"Full set: {len(data)} questions")
    for s, c in sorted(counts.items()):
        print(f"  {s:22s}: {c} ({c/len(data):.1%})")

    # Proportional allocation (largest-remainder method to hit --n exactly)
    alloc = {}
    ratios = {s: c / len(data) for s, c in counts.items()}
    raw = {s: ratios[s] * args.n for s in counts}
    alloc = {s: int(raw[s]) for s in raw}
    remainders = sorted(raw.items(), key=lambda x: x[1] - alloc[x[0]], reverse=True)
    short = args.n - sum(alloc.values())
    for s, _ in remainders[:short]:
        alloc[s] += 1

    print(f"\nTarget sample ({args.n}):")
    for s, c in sorted(alloc.items()):
        print(f"  {s:22s}: {c}")

    # Sample
    rng = random.Random(args.seed)
    by_stratum: dict[str, list[dict]] = {}
    for q in data:
        by_stratum.setdefault(_stratum_of(q), []).append(q)

    chosen = []
    for s, target in alloc.items():
        pool = by_stratum.get(s, [])
        if target > len(pool):
            print(f"  warn: stratum {s!r} only has {len(pool)} but want {target}; "
                  f"taking all")
            target = len(pool)
        chosen.extend(rng.sample(pool, target))

    # Shuffle for good measure
    rng.shuffle(chosen)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(chosen, f)
    print(f"\nWrote {len(chosen)} questions → {args.output}")


if __name__ == "__main__":
    main()
