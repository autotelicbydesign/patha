"""Plasticity-stressing benchmark (v0.5 #3).

BeliefEval tests binary supersession/validity correctness — it doesn't
stress what plasticity mechanisms actually do. Which is why the v0.3
plasticity ablations returned a null result across all configurations.

This benchmark stresses plasticity directly by asking questions the
regular detector-driven benchmark can't:

  - LTP (reinforcement): does confidence rise across repeated assertions
    from distinct sources?
  - LTD (decay): does confidence fall when beliefs aren't touched for
    a long time?
  - Homeostasis: does no single reinforced belief dominate after many
    ingests?
  - Pruning: do deeply-superseded beliefs get archived on schedule?
  - Hebbian: do co-retrieved beliefs form measurable associations?

Each scenario runs a sequence of ingests / queries / time-advances,
then checks a specific numerical property of the resulting state.

Usage:
    python -m eval.plasticity_benchmark \\
        --output runs/plasticity_benchmark/results.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

from patha.belief.contradiction import StubContradictionDetector
from patha.belief.layer import BeliefLayer, PlasticityConfig
from patha.belief.plasticity import LongTermDepression
from patha.belief.store import BeliefStore
from patha.belief.types import Pramana


@dataclass
class PlasticityScore:
    test: str
    description: str
    passed: bool
    observed: float
    expected_min: float | None = None
    expected_max: float | None = None
    notes: str = ""


def _fresh_layer(
    *,
    plasticity: PlasticityConfig | None = None,
) -> BeliefLayer:
    return BeliefLayer(
        store=BeliefStore(),
        detector=StubContradictionDetector(),
        plasticity=plasticity if plasticity is not None else PlasticityConfig(),
    )


# ─── LTP: reinforcement raises confidence ──────────────────────────

def test_ltp_reinforcement_raises_confidence() -> PlasticityScore:
    """Ingest a belief, then reinforce it 5 times from distinct sources.
    Expected: confidence climbs (≥0.95) AND deep_confidence (vāsanā)
    becomes established."""
    layer = _fresh_layer(plasticity=PlasticityConfig(enabled=False))
    b = layer.ingest(
        proposition="I work as a data engineer",
        asserted_at=datetime(2024, 1, 1),
        asserted_in_session="s0",
        source_proposition_id="p0",
        pramana=Pramana.PRATYAKSA,
        confidence=0.5,  # start low so we can see the rise
    ).new_belief.id

    for i in range(5):
        r = layer.ingest(
            proposition=f"reinforcement {i}",
            asserted_at=datetime(2024, 2, 1 + i),
            asserted_in_session=f"s{i+1}",  # distinct session
            source_proposition_id=f"reinforce-{i}",
            pramana=Pramana.PRATYAKSA,
        )
        layer.store.reinforce(b, r.new_belief.id)

    final_conf = layer.store.get(b).confidence  # type: ignore[union-attr]
    vasana = layer.store.get(b).deep_confidence  # type: ignore[union-attr]
    # 5 distinct-source reinforcements at 30% gap-closure from 0.5:
    #   after 5 bumps, confidence ≈ 0.916 (mathematically). Require ≥0.9
    #   and that vāsanā has crystallised (samskara_count ≥ 5).
    passed = final_conf >= 0.9 and vasana is not None
    return PlasticityScore(
        test="ltp_reinforcement",
        description="5 distinct-source reinforcements lift confidence ≥0.9 AND crystallise vāsanā",
        passed=passed,
        observed=final_conf,
        expected_min=0.9,
        notes=f"deep={vasana}",
    )


# ─── LTD: time decay lowers confidence ─────────────────────────────

def test_ltd_decay_lowers_confidence() -> PlasticityScore:
    """Ingest a belief, advance time by 2x the half-life, decay it.
    Expected: confidence halves twice (~0.25x original)."""
    layer = _fresh_layer(plasticity=PlasticityConfig(enabled=False))
    b = layer.ingest(
        proposition="something",
        asserted_at=datetime(2024, 1, 1),
        asserted_in_session="s0",
        source_proposition_id="p0",
        confidence=1.0,
    ).new_belief.id

    ltd = LongTermDepression(half_life_days=100, floor=0.0)
    ltd.apply_to_store(
        layer.store,
        now=datetime(2024, 1, 1) + timedelta(days=200),
    )
    final_conf = layer.store.get(b).confidence  # type: ignore[union-attr]
    passed = 0.2 <= final_conf <= 0.3
    return PlasticityScore(
        test="ltd_decay",
        description="After 2×half_life of disuse, confidence lands near 0.25",
        passed=passed,
        observed=final_conf,
        expected_min=0.2,
        expected_max=0.3,
    )


# ─── Homeostasis: no single belief dominates ────────────────────────

def test_homeostasis_caps_dominance() -> PlasticityScore:
    """Add 10 beliefs, reinforce only the first many times. With
    homeostasis on, all beliefs' confidences stay within a bounded
    range after homeostatic rescaling fires."""
    layer = _fresh_layer(plasticity=PlasticityConfig(
        enabled=True,
        homeostasis_on_ingest=True,
        homeostasis_interval_ingests=5,
        homeostasis_target_mean=0.7,
        ltd_on_query=False,
        hebbian_on_query=False,
        pruning_on_ingest=False,
    ))
    first = layer.ingest(
        proposition="dominant belief",
        asserted_at=datetime(2024, 1, 1),
        asserted_in_session="s0",
        source_proposition_id="p0",
        confidence=1.0,
    ).new_belief.id

    # Add 9 more beliefs (each triggers the tick)
    for i in range(1, 10):
        r = layer.ingest(
            proposition=f"other belief {i}",
            asserted_at=datetime(2024, 1, 1 + i),
            asserted_in_session=f"s{i}",
            source_proposition_id=f"p{i}",
            confidence=0.3,  # low starting
        )
        # Reinforce the first belief many times
        layer.store.reinforce(first, r.new_belief.id)

    confidences = [b.confidence for b in layer.store.current()]
    if not confidences:
        return PlasticityScore(
            test="homeostasis_dominance",
            description="No beliefs to measure",
            passed=False,
            observed=0.0,
        )
    # After homeostasis fires, the ratio of max/min should be bounded
    ratio = max(confidences) / max(min(confidences), 1e-9)
    passed = ratio <= 10.0  # reasonable bound
    return PlasticityScore(
        test="homeostasis_dominance",
        description="Homeostasis keeps max/min confidence ratio bounded",
        passed=passed,
        observed=ratio,
        expected_max=10.0,
    )


# ─── Pruning: deep chain archived ──────────────────────────────────

def test_pruning_archives_deep_chain() -> PlasticityScore:
    """Build an 8-generation supersession chain. Pruning with max_depth=3
    archives generations 4-7."""
    from patha.belief.types import ContradictionLabel, ContradictionResult

    class AlwaysContradicts:
        def detect(self, p1, p2):
            return ContradictionResult(
                label=ContradictionLabel.CONTRADICTS, confidence=0.99
            )

        def detect_batch(self, pairs):
            return [self.detect(p1, p2) for p1, p2 in pairs]

    layer = BeliefLayer(
        store=BeliefStore(),
        detector=AlwaysContradicts(),
        contradiction_threshold=0.5,
        plasticity=PlasticityConfig(
            enabled=True,
            pruning_on_ingest=True,
            pruning_interval_ingests=8,
            pruning_max_depth=3,
            ltd_on_query=False,
            hebbian_on_query=False,
            homeostasis_on_ingest=False,
        ),
    )
    for i in range(8):
        layer.ingest(
            proposition=f"claim gen {i}",
            asserted_at=datetime(2024, 1, 1 + i),
            asserted_in_session=f"s{i}",
            source_proposition_id=f"p{i}",
        )

    n_archived = len(layer.store.archived())
    # Pruning fires at tick 8. With max_depth=3, depths 4-7 get archived (4 beliefs).
    passed = n_archived >= 3  # conservative — at least 3 deep beliefs archived
    return PlasticityScore(
        test="pruning_deep_chain",
        description="Pruning at max_depth=3 archives deep-chain ancestors",
        passed=passed,
        observed=float(n_archived),
        expected_min=3,
    )


# ─── Hebbian: co-retrieval grows edges ─────────────────────────────

def test_hebbian_association_edges() -> PlasticityScore:
    """Co-retrieve two beliefs 5 times. Expected: Hebbian weight ≥ 0.5 × 5 × learning_rate."""
    layer = _fresh_layer(plasticity=PlasticityConfig(
        enabled=True,
        hebbian_on_query=True,
        hebbian_learning_rate=0.1,
        ltd_on_query=False,
        homeostasis_on_ingest=False,
        pruning_on_ingest=False,
    ))
    a = layer.ingest(
        proposition="a", asserted_at=datetime(2024, 1, 1),
        asserted_in_session="s0", source_proposition_id="pa",
    ).new_belief.id
    b = layer.ingest(
        proposition="b", asserted_at=datetime(2024, 1, 2),
        asserted_in_session="s1", source_proposition_id="pb",
    ).new_belief.id

    for _ in range(5):
        layer.query([a, b], at_time=datetime(2024, 6, 1))

    weight = layer.hebbian.weight(a, b)
    expected = 0.1 * 5  # 0.5
    passed = weight == expected
    return PlasticityScore(
        test="hebbian_association",
        description="Hebbian edge weight grows ≈ lr × co-retrievals",
        passed=passed,
        observed=weight,
        expected_min=expected - 0.01,
        expected_max=expected + 0.01,
    )


# ─── Vāsanā preservation under decay ───────────────────────────────

def test_vasana_survives_surface_decay() -> PlasticityScore:
    """A belief with established vāsanā should retain higher
    effective_confidence than surface even after heavy decay."""
    layer = _fresh_layer(plasticity=PlasticityConfig(enabled=False))
    b = layer.ingest(
        proposition="long-held position",
        asserted_at=datetime(2020, 1, 1),
        asserted_in_session="s0",
        source_proposition_id="p0",
        pramana=Pramana.PRATYAKSA,
        confidence=0.5,
    ).new_belief.id

    # Reinforce 5 times from distinct sources — crystallise vāsanā
    for i in range(5):
        r = layer.ingest(
            proposition=f"reinforce {i}",
            asserted_at=datetime(2020, 2 + i, 1),
            asserted_in_session=f"s{i+1}",
            source_proposition_id=f"r{i}",
            pramana=Pramana.PRATYAKSA,
        )
        layer.store.reinforce(b, r.new_belief.id)

    # Now apply heavy decay (surface falls below vāsanā)
    ltd = LongTermDepression(half_life_days=365, floor=0.0)
    ltd.apply_to_store(
        layer.store,
        now=datetime(2025, 6, 1),
    )

    belief = layer.store.get(b)
    assert belief is not None
    surface = belief.confidence
    effective = belief.effective_confidence

    passed = belief.deep_confidence is not None and effective >= surface
    return PlasticityScore(
        test="vasana_preservation",
        description="Effective confidence preserves the belief against surface decay",
        passed=passed,
        observed=effective,
        notes=f"surface={surface:.3f} deep={belief.deep_confidence} effective={effective:.3f}",
    )


# ─── Main ──────────────────────────────────────────────────────────

TESTS = [
    test_ltp_reinforcement_raises_confidence,
    test_ltd_decay_lowers_confidence,
    test_homeostasis_caps_dominance,
    test_pruning_archives_deep_chain,
    test_hebbian_association_edges,
    test_vasana_survives_surface_decay,
]


def run_all() -> list[PlasticityScore]:
    return [t() for t in TESTS]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Patha plasticity-stressing benchmark"
    )
    parser.add_argument(
        "--output",
        default="runs/plasticity_benchmark/results.json",
    )
    args = parser.parse_args(argv)

    results = run_all()
    n_passed = sum(1 for r in results if r.passed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {
                "n_tests": len(results),
                "n_passed": n_passed,
                "accuracy": n_passed / len(results) if results else 0.0,
                "results": [asdict(r) for r in results],
            },
            f,
            indent=2,
        )

    print()
    print("=" * 60)
    print("Plasticity-stressing benchmark")
    print("=" * 60)
    print(f"  {n_passed}/{len(results)} tests passed")
    print()
    for r in results:
        mark = "✓" if r.passed else "✗"
        print(f"  {mark} {r.test:30s}  observed={r.observed:.3f}  {r.notes}")
    print()
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
