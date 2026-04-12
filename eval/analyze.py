"""Error analysis for Patha eval results.

Reads a results.json file and produces a detailed breakdown of misses:
which questions failed, what strata they belong to, and what sessions
were retrieved vs. expected.

Usage:
    python -m eval.analyze runs/diverse_32_ce/results.json
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def analyze(results_path: str | Path) -> None:
    """Analyze eval results and print a detailed report."""
    with open(results_path) as f:
        data = json.load(f)

    summary = data["summary"]
    per_q = data["per_question"]

    print("=" * 70)
    print(f"EVAL ANALYSIS: {results_path}")
    print("=" * 70)

    # Overall metrics
    print(f"\nTotal questions: {summary['total_questions']}")
    print(f"Retrieval questions: {summary['retrieval_questions']}")
    print(f"R@5: {summary.get('recall_any@5', 'N/A')}")
    print(f"R@10: {summary.get('recall_any@10', 'N/A')}")

    # Per-stratum
    print("\nPer-stratum R@5:")
    for stratum, score in sorted(summary.get("per_stratum_recall_any@5", {}).items()):
        print(f"  {stratum}: {score:.3f}")

    # Find misses
    misses_5 = [q for q in per_q if q.get("recall_any@5", 1.0) == 0.0]
    misses_10 = [q for q in per_q if q.get("recall_any@10", 1.0) == 0.0]
    hits_5 = [q for q in per_q if q.get("recall_any@5", 0.0) == 1.0]

    print(f"\nMisses @ K=5: {len(misses_5)} / {len(per_q)}")
    print(f"Misses @ K=10: {len(misses_10)} / {len(per_q)}")

    if misses_5:
        print("\n" + "-" * 50)
        print("DETAILED MISS ANALYSIS @ K=5")
        print("-" * 50)

        miss_by_stratum: Counter[str] = Counter()
        miss_by_type: Counter[str] = Counter()

        for q in misses_5:
            miss_by_stratum[q["stratum"]] += 1
            miss_by_type[q["question_type"]] += 1

            print(f"\n  QID: {q['question_id']}")
            print(f"  Type: {q['question_type']} ({q['stratum']})")
            print(f"  Gold sessions: {q['gold_session_ids']}")
            retrieved = q["retrieved_chunk_ids"][:10]
            retrieved_sessions = list(dict.fromkeys(
                cid.split("#")[0] for cid in retrieved
            ))
            print(f"  Retrieved sessions (top-10): {retrieved_sessions}")
            print(f"  Retrieved chunks (top-5): {retrieved[:5]}")

            # Check if gold appears in top-10
            gold_set = set(q["gold_session_ids"])
            for i, cid in enumerate(retrieved):
                sid = cid.split("#")[0]
                if sid in gold_set:
                    print(f"  ** Gold found at rank {i+1}: {cid}")
                    break
            else:
                print(f"  ** Gold NOT in top-{len(retrieved)} at all")

        print("\n" + "-" * 50)
        print("MISS DISTRIBUTION")
        print("-" * 50)
        total_by_stratum: Counter[str] = Counter()
        for q in per_q:
            total_by_stratum[q["stratum"]] += 1

        for stratum in sorted(total_by_stratum):
            total = total_by_stratum[stratum]
            missed = miss_by_stratum.get(stratum, 0)
            hit_rate = (total - missed) / total if total > 0 else 0
            print(f"  {stratum}: {missed}/{total} missed ({hit_rate:.1%} hit rate)")

    # Success analysis: what makes hits work?
    if hits_5:
        print("\n" + "-" * 50)
        print("HIT ANALYSIS @ K=5 (sample)")
        print("-" * 50)
        for q in hits_5[:3]:
            print(f"  QID: {q['question_id']} ({q['stratum']})")
            retrieved = q["retrieved_chunk_ids"][:5]
            retrieved_sessions = list(dict.fromkeys(
                cid.split("#")[0] for cid in retrieved
            ))
            print(f"  Gold: {q['gold_session_ids']}")
            print(f"  Top-5 sessions: {retrieved_sessions}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m eval.analyze <results.json>")
        sys.exit(1)
    analyze(sys.argv[1])
