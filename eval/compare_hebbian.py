"""Compare multi-session results between two benchmark runs.

Use this to measure the actual lift from Innovation #1 (Hebbian
cluster expansion) once both A/B runs complete:

    uv run python -m eval.compare_hebbian \\
        --baseline runs/innovation1_hebbian/multisession_no_hebbian.json \\
        --treatment runs/innovation1_hebbian/multisession_with_hebbian.json

Reports:
  - Per-arm accuracy
  - Per-question deltas (gained, lost, unchanged)
  - 95% confidence interval on the difference (paired bootstrap)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_records(path: Path) -> dict[str, bool]:
    """Read a benchmark output and return {question_id: correct?} mapping."""
    data = json.loads(path.read_text())
    # Two output shapes — the longmemeval_integrated runner uses
    # `outcomes`, the simpler ablation runner uses `records`.
    if "outcomes" in data:
        return {
            o["question_id"]: bool(o.get("correct_with_history", False))
            for o in data["outcomes"]
        }
    if "records" in data:
        return {r["question_id"]: bool(r["correct"]) for r in data["records"]}
    raise ValueError(f"unrecognized output shape in {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, type=Path,
                    help="Run with Hebbian DISABLED")
    ap.add_argument("--treatment", required=True, type=Path,
                    help="Run with Hebbian ENABLED")
    args = ap.parse_args()

    base = _load_records(args.baseline)
    treat = _load_records(args.treatment)
    common = sorted(set(base) & set(treat))
    if not common:
        print("no overlap between the two runs")
        return

    base_correct = sum(1 for q in common if base[q])
    treat_correct = sum(1 for q in common if treat[q])
    n = len(common)

    gained = [q for q in common if treat[q] and not base[q]]
    lost = [q for q in common if base[q] and not treat[q]]

    print(f"Compared {n} questions:")
    print(f"  baseline (no Hebbian):  {base_correct}/{n} = {base_correct/n:.3f}")
    print(f"  treatment (Hebbian on): {treat_correct}/{n} = {treat_correct/n:.3f}")
    print(f"  delta:                  {(treat_correct - base_correct):+d} "
          f"({(treat_correct - base_correct)/n:+.3f})")
    print(f"  questions gained:       {len(gained)}")
    print(f"  questions lost:         {len(lost)}")
    if gained[:5]:
        print(f"    sample gained: {gained[:5]}")
    if lost[:5]:
        print(f"    sample lost:   {lost[:5]}")

    # Paired-bootstrap 95% CI on the difference
    import random
    diffs = [int(treat[q]) - int(base[q]) for q in common]
    if all(d == 0 for d in diffs):
        print("  (no per-question disagreement; Hebbian is a no-op)")
        return
    rng = random.Random(0xCAFEBABE)
    boot_means = []
    for _ in range(2000):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    lo, hi = boot_means[int(0.025 * len(boot_means))], boot_means[int(0.975 * len(boot_means))]
    print(f"  95% bootstrap CI on Δ:  [{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    main()
