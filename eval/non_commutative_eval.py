"""Measure belief-order non-commutativity across the 300 BeliefEval
scenarios.

For each scenario we take its propositions, reingest them under:
  (A) chronological order — the canonical timeline
  (B) reversed order     — everything backwards

If the final current-belief set differs between the two, the scenario
is "non-commutative". The fraction of non-commutative scenarios is the
headline metric.

This is the empirical test for whether Patha's belief evolution is
genuinely order-sensitive (quantum-cognition-inspired) or effectively
commutative in practice.

Note: Since each ingest uses the proposition's original timestamp
from the scenario, and the BeliefStore supersedes based on temporal
ordering, reversing the INGEST order alone might not change much if
timestamps dominate. The test below ALSO replaces timestamps with
"asserted_at equals current time during the run" so ingestion order
maps to temporal order — revealing order dependence.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

from patha.belief.adhyasa_detector import AdhyasaAwareDetector
from patha.belief.contradiction import NLIContradictionDetector
from patha.belief.counterfactual import (
    CounterfactualInput,
    reingest_order_sensitivity,
)
from patha.belief.numerical_detector import NumericalAwareDetector
from patha.belief.sequential_detector import SequentialEventDetector

from eval.belief_eval import load_scenarios


def _make_detector():
    return NumericalAwareDetector(
        inner=SequentialEventDetector(
            inner=AdhyasaAwareDetector(inner=NLIContradictionDetector())
        )
    )


def scenario_to_inputs(scenario, *, rebind_timestamps: bool) -> list[CounterfactualInput]:
    """Convert scenario propositions into CounterfactualInputs.

    rebind_timestamps: if True, each proposition is assigned a
    timestamp based on its position in the SORTED input order, so
    reordering the ingest truly changes temporal precedence.
    """
    sorted_props = sorted(scenario.propositions, key=lambda p: p.asserted_at)
    base = datetime(2024, 1, 1)
    inputs = []
    for i, p in enumerate(sorted_props):
        if rebind_timestamps:
            # Timestamp follows ingest order in the run; we re-bind
            # at reingest time using position index.
            at = p.asserted_at  # will be overridden per-ordering
        else:
            at = p.asserted_at
        inputs.append(CounterfactualInput(
            proposition=p.text,
            asserted_at=at,
            asserted_in_session=p.session,
            source_proposition_id=f"{scenario.id}-p{i}",
        ))
    return inputs


def _rebind_in_order(
    inputs: list[CounterfactualInput], ordering: list[int], base: datetime
) -> list[CounterfactualInput]:
    """Return a new list of inputs in `ordering`, with asserted_at
    monotonically increasing so ingest order = temporal order."""
    out = []
    for k, idx in enumerate(ordering):
        x = inputs[idx]
        new_at = base + timedelta(days=k)
        out.append(CounterfactualInput(
            proposition=x.proposition,
            asserted_at=new_at,
            asserted_in_session=x.asserted_in_session,
            source_proposition_id=x.source_proposition_id,
        ))
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Measure belief-order non-commutativity"
    )
    ap.add_argument("--scenarios",
                    default="eval/belief_eval_data/v05_combined_300.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--output",
                    default="runs/non_commutative/results.json")
    args = ap.parse_args(argv)

    scenarios = load_scenarios(args.scenarios)
    if args.limit is not None:
        scenarios = scenarios[: args.limit]

    det = _make_detector()
    base = datetime(2024, 1, 1)
    records: list[dict] = []
    n_nonc = 0
    total_div = 0.0
    # Per-family counts
    by_family: dict[str, dict] = {}

    for i, sc in enumerate(scenarios, 1):
        if len(sc.propositions) < 2:
            continue
        raw_inputs = scenario_to_inputs(sc, rebind_timestamps=True)
        fwd_order = list(range(len(raw_inputs)))
        rev_order = list(reversed(fwd_order))

        inputs_fwd = _rebind_in_order(raw_inputs, fwd_order, base)
        inputs_rev = _rebind_in_order(raw_inputs, rev_order, base)
        result = reingest_order_sensitivity(
            inputs=(inputs_fwd + inputs_rev),  # unified pool
            orderings=[
                list(range(len(inputs_fwd))),
                list(range(len(inputs_fwd), len(inputs_fwd) + len(inputs_rev))),
            ],
            detector=det,
        )
        non_comm = result["non_commutative"]
        div = result["divergence"]
        if non_comm:
            n_nonc += 1
        total_div += div
        by_family.setdefault(sc.family, {"total": 0, "nonc": 0, "div_sum": 0.0})
        by_family[sc.family]["total"] += 1
        by_family[sc.family]["nonc"] += 1 if non_comm else 0
        by_family[sc.family]["div_sum"] += div

        records.append({
            "scenario_id": sc.id,
            "family": sc.family,
            "non_commutative": non_comm,
            "divergence": div,
            "fwd_current": result["per_ordering"][0]["current_props"],
            "rev_current": result["per_ordering"][1]["current_props"],
        })
        if i % 25 == 0:
            print(f"  [{i}/{len(scenarios)}] nonc={n_nonc} "
                  f"mean_div={total_div/i:.3f}")

    n = len(records)
    print()
    print("=" * 60)
    print("Non-commutativity measurement")
    print("=" * 60)
    print(f"  Scenarios:              {n}")
    print(f"  Non-commutative:        {n_nonc} ({n_nonc/max(n,1):.1%})")
    print(f"  Mean divergence (fwd vs rev): {total_div/max(n,1):.3f}")
    print()
    print("  By family:")
    for fam, st in sorted(by_family.items()):
        total = st["total"]
        nonc = st["nonc"]
        print(f"    {fam}: {nonc}/{total} non-commutative, "
              f"mean div={st['div_sum']/max(total,1):.3f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "n_scenarios": n,
            "n_non_commutative": n_nonc,
            "non_commutative_rate": n_nonc / max(n, 1),
            "mean_divergence": total_div / max(n, 1),
            "by_family": by_family,
            "records": records,
        }, f, indent=2, default=str)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
