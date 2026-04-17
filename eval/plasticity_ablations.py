"""Plasticity ablations for Phase 2 v0.3.

Measures the marginal contribution of each plasticity mechanism
(LTP via reinforcement, LTD, Hebbian, homeostasis, pruning) on
BeliefEval accuracy and token economy.

Each ablation runs the same scenarios with one mechanism disabled
(or all disabled for the 'no_plasticity' baseline), then compares
against the full-plasticity baseline.

Usage:
    python -m eval.plasticity_ablations \\
        --scenarios eval/belief_eval_data/seed_scenarios.jsonl \\
        --output runs/plasticity_ablations/results.json

Output: JSON with per-ablation BeliefEval accuracy + delta vs baseline.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from eval.belief_eval import (
    load_scenarios,
    run_scenario,
    summarise,
)
from patha.belief.contradiction import StubContradictionDetector
from patha.belief.layer import PlasticityConfig


# ─── Ablation definitions ───────────────────────────────────────────

@dataclass
class Ablation:
    name: str
    description: str
    config: PlasticityConfig


def _ablations() -> list[Ablation]:
    return [
        Ablation(
            name="baseline",
            description="All plasticity mechanisms enabled (v0.3 default)",
            config=PlasticityConfig(enabled=True),
        ),
        Ablation(
            name="no_plasticity",
            description="All plasticity disabled (v0.2-equivalent behaviour)",
            config=PlasticityConfig(enabled=False),
        ),
        Ablation(
            name="no_ltd",
            description="LTD disabled; other mechanisms on",
            config=PlasticityConfig(
                enabled=True, ltd_on_query=False,
            ),
        ),
        Ablation(
            name="no_hebbian",
            description="Hebbian disabled; other mechanisms on",
            config=PlasticityConfig(
                enabled=True, hebbian_on_query=False,
            ),
        ),
        Ablation(
            name="no_homeostasis",
            description="Homeostasis disabled; other mechanisms on",
            config=PlasticityConfig(
                enabled=True, homeostasis_on_ingest=False,
            ),
        ),
        Ablation(
            name="no_pruning",
            description="Synaptic pruning disabled; other mechanisms on",
            config=PlasticityConfig(
                enabled=True, pruning_on_ingest=False,
            ),
        ),
    ]


# ─── Runner ─────────────────────────────────────────────────────────

def _run_one(
    scenarios_path: Path, ablation: Ablation,
) -> dict:
    """Run BeliefEval with the given plasticity config; return summary."""
    # Monkey-patch the layer construction in run_scenario by temporarily
    # wrapping a custom version. Simpler: replicate run_scenario's body
    # with an injected PlasticityConfig.
    from patha.belief.layer import BeliefLayer
    from patha.belief.store import BeliefStore
    from eval.belief_eval import QuestionResult, _score_current_belief, _score_validity_at_time

    detector = StubContradictionDetector()
    scenarios = load_scenarios(scenarios_path)

    all_results = []
    for scenario in scenarios:
        layer = BeliefLayer(
            store=BeliefStore(),
            detector=detector,
            plasticity=ablation.config,
            contradiction_threshold=0.5,
        )
        props_sorted = sorted(scenario.propositions, key=lambda p: p.asserted_at)
        all_ids = []
        for i, p in enumerate(props_sorted):
            ev = layer.ingest(
                proposition=p.text,
                asserted_at=p.asserted_at,
                asserted_in_session=p.session,
                source_proposition_id=f"{scenario.id}-p{i}",
            )
            all_ids.append(ev.new_belief.id)

        for q in scenario.questions:
            at_time = q.at_time if q.at_time is not None else datetime(2030, 1, 1)
            query_result = layer.query(
                all_ids, at_time=at_time, include_history=True
            )
            current_props = [b.proposition for b in query_result.current]
            superseded_props = [b.proposition for b in query_result.history]

            if q.type == "current_belief":
                correct, details = _score_current_belief(
                    q, current_props, superseded_props
                )
            elif q.type == "validity_at_time":
                valid = len(query_result.current) > 0
                correct, details = _score_validity_at_time(q, valid)
            else:
                correct, details = False, {"error": q.type}

            all_results.append(
                QuestionResult(
                    scenario_id=scenario.id,
                    family=scenario.family,
                    question=q.q,
                    type=q.type,
                    correct=correct,
                    tokens_in_summary=query_result.tokens_in_summary,
                    details=details,
                )
            )

    return asdict(summarise(all_results))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenarios",
        default="eval/belief_eval_data/seed_scenarios.jsonl",
    )
    parser.add_argument(
        "--output",
        default="runs/plasticity_ablations/results.json",
    )
    args = parser.parse_args(argv)

    scenarios_path = Path(args.scenarios)
    ablations = _ablations()

    results = []
    for ab in ablations:
        print(f"Running ablation: {ab.name} ({ab.description})")
        summary = _run_one(scenarios_path, ab)
        results.append({
            "name": ab.name,
            "description": ab.description,
            "accuracy": summary["accuracy"],
            "n_correct": summary["n_correct"],
            "n_questions": summary["n_questions"],
            "by_family": summary["by_family"],
            "avg_tokens_in_summary": summary["avg_tokens_in_summary"],
        })

    # Compute deltas vs baseline
    baseline = next((r for r in results if r["name"] == "baseline"), None)
    if baseline is not None:
        for r in results:
            r["delta_vs_baseline"] = r["accuracy"] - baseline["accuracy"]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"scenarios_path": str(scenarios_path), "results": results}, f, indent=2, default=str)

    print()
    print("=" * 60)
    print("Plasticity ablation comparison")
    print("=" * 60)
    print(f"  {'ablation':<20s}  {'accuracy':<10s}  {'Δ vs baseline':<14s}")
    for r in results:
        delta = r.get("delta_vs_baseline", 0.0)
        print(f"  {r['name']:<20s}  {r['accuracy']:<10.3f}  {delta:+.3f}")

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
