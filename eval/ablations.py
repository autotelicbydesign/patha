"""Ablation matrix runner for Patha.

Runs a suite of ablation experiments against a LongMemEval dataset,
comparing each variant against the full-pipeline baseline. Each ablation
toggles one architectural component off to measure its marginal contribution.

Usage:
    python -m eval.ablations --data /tmp/patha_stratified_100.json [--verbose]

Ablations:
    1. no-reranker     — skip cross-encoder reranking
    2. no-songline     — skip songline graph walks
    3. single-view-v1  — only use v1 (pada) view, no multi-view RRF
    4. views-v1-v4     — only use v1 + v4 (pada + jata), 2 views
    5. no-bm25         — skip BM25 sparse leg of hybrid retrieval

Each ablation saves results to runs/ablation_<name>/results.json and
prints a comparison table at the end.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from eval.runner import (
    load_longmemeval,
    run_evaluation,
    save_results,
)
from eval.metrics import EvalReport
from patha.retrieval.pipeline import PipelineConfig


@dataclass
class AblationConfig:
    """One ablation experiment."""

    name: str
    description: str
    # Overrides for run_evaluation kwargs
    reranker_name: str = "ce-mini"
    use_songline: bool = True
    views: list[str] | None = None  # None = all 7
    use_bm25: bool = True


# ── Standard ablation suite ────────────────────────────────────────

ABLATIONS: list[AblationConfig] = [
    AblationConfig(
        name="baseline",
        description="Full pipeline (7 views + BM25 + songline + CE reranker)",
    ),
    AblationConfig(
        name="no-reranker",
        description="No cross-encoder reranker (RRF scores only)",
        reranker_name="none",
    ),
    AblationConfig(
        name="no-songline",
        description="No songline graph walks (Pillar 2 disabled)",
        use_songline=False,
    ),
    AblationConfig(
        name="single-view-v1",
        description="Single view (v1 pada only, no multi-view redundancy)",
        views=["v1"],
    ),
    AblationConfig(
        name="views-v1-v4",
        description="Two views (v1 pada + v4 jata)",
        views=["v1", "v4"],
    ),
    AblationConfig(
        name="no-reranker-no-songline",
        description="No reranker + no songline (pure hybrid retrieval)",
        reranker_name="none",
        use_songline=False,
    ),
]


def run_ablation(
    ab: AblationConfig,
    data: list[dict],
    *,
    device: str | None = None,
    output_dir: str = "runs",
    verbose: bool = False,
) -> EvalReport:
    """Run a single ablation experiment."""
    from patha.models.embedder_st import SentenceTransformerEmbedder
    from patha.retrieval.pipeline import Reranker

    embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2", device=device)

    reranker_fn: Reranker | None = None
    if ab.reranker_name == "ce-mini":
        from patha.retrieval.reranker import CrossEncoderReranker
        reranker_fn = CrossEncoderReranker(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            device=device,
            rrf_blend=0.2,
        )

    config = PipelineConfig(views=ab.views)

    # Per-question checkpoint so crashes don't lose progress
    checkpoint_path = Path(output_dir) / f"ablation_{ab.name}" / "eval_checkpoint.pkl"

    return run_evaluation(
        data,
        embedder=embedder,
        config=config,
        reranker=reranker_fn,
        use_songline=ab.use_songline,
        eval_checkpoint_path=checkpoint_path,
        verbose=verbose,
    )


def format_table(results: list[tuple[AblationConfig, dict, float]]) -> str:
    """Format ablation results as a markdown table."""
    lines = []
    lines.append("| Ablation | R@5 | R@10 | R_all@5 | Δ R@5 | Time |")
    lines.append("|----------|:---:|:----:|:-------:|:-----:|-----:|")

    baseline_r5 = results[0][1].get("recall_any@5", 0.0) if results else 0.0

    for ab, summary, elapsed in results:
        r5 = summary.get("recall_any@5", 0.0)
        r10 = summary.get("recall_any@10", 0.0)
        rall5 = summary.get("recall_all@5", 0.0)
        delta = r5 - baseline_r5
        delta_str = f"{delta:+.3f}" if ab.name != "baseline" else "—"
        lines.append(
            f"| {ab.name} | {r5:.3f} | {r10:.3f} | {rall5:.3f} | {delta_str} | {elapsed:.0f}s |"
        )

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Patha ablation matrix")
    parser.add_argument("--data", required=True, help="Path to LongMemEval JSON")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit to first N questions (0 = all)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default="runs", help="Base output directory")
    parser.add_argument("--ablations", default=None,
                        help="Comma-separated ablation names to run (default: all)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    print(f"Loading {args.data}...")
    data = load_longmemeval(args.data)
    if args.limit > 0:
        data = data[: args.limit]
    print(f"Running ablations on {len(data)} questions\n")

    # Filter ablations if requested
    ablations = ABLATIONS
    if args.ablations:
        wanted = set(args.ablations.split(","))
        ablations = [a for a in ablations if a.name in wanted]
        # Always include baseline for comparison
        if not any(a.name == "baseline" for a in ablations):
            ablations.insert(0, next(a for a in ABLATIONS if a.name == "baseline"))

    all_results: list[tuple[AblationConfig, dict, float]] = []

    for ab in ablations:
        output_path = Path(args.output_dir) / f"ablation_{ab.name}" / "results.json"

        # Skip ablations that already have saved results
        if output_path.exists():
            existing = json.load(open(output_path))
            existing_n = existing.get("summary", {}).get("total_questions", 0)
            if existing_n >= len(data):
                print(f"Skipping {ab.name}: already have {existing_n}q results at {output_path}")
                all_results.append((ab, existing["summary"], 0.0))
                continue

        print(f"{'=' * 60}")
        print(f"Ablation: {ab.name}")
        print(f"  {ab.description}")
        print(f"{'=' * 60}")

        t0 = time.time()
        report = run_ablation(ab, data, device=args.device, output_dir=args.output_dir, verbose=args.verbose)
        elapsed = time.time() - t0

        summary = report.summary()
        all_results.append((ab, summary, elapsed))

        # Save per-ablation results immediately
        save_results(report, output_path)
        # Clean up checkpoint now that full results are saved
        ckpt = Path(args.output_dir) / f"ablation_{ab.name}" / "eval_checkpoint.pkl"
        if ckpt.exists():
            ckpt.unlink()
        print(f"  Saved to {output_path}")

        r5 = summary.get("recall_any@5", 0.0)
        print(f"\n  R@5: {r5:.3f}  ({elapsed:.0f}s)")
        per_stratum = summary.get("per_stratum_recall_any@5", {})
        for s, v in per_stratum.items():
            print(f"    {s}: {v:.3f}")
        print()

    # Print comparison table
    print("\n" + "=" * 60)
    print("ABLATION COMPARISON")
    print("=" * 60)
    print()
    print(format_table(all_results))
    print()

    # Save comparison
    comparison_path = Path(args.output_dir) / "ablation_comparison.json"
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    with open(comparison_path, "w") as f:
        json.dump(
            [
                {
                    "name": ab.name,
                    "description": ab.description,
                    "summary": summary,
                    "elapsed_seconds": round(elapsed, 1),
                }
                for ab, summary, elapsed in all_results
            ],
            f,
            indent=2,
        )
    print(f"Comparison saved to {comparison_path}")


if __name__ == "__main__":
    main()
