"""BeliefEval — Phase 2 evaluation harness.

Loads scenarios from a JSONL file, ingests each scenario's propositions
into a fresh BeliefLayer (in temporal order), then runs the associated
questions and scores the outputs. Reports correctness metrics and
token-economy metrics side by side.

Scenarios fall into families (currently three):
  - preference_supersession — user's stated preferences change over time
  - factual_supersession    — facts about the user's life change over time
  - temporally_bounded      — beliefs with explicit validity windows

Question types:
  - current_belief    — "What does the user currently believe about X?"
                         Scored via token-overlap with expected_current_contains,
                         and absence of any expected_superseded_contains terms.
  - validity_at_time  — "Is this belief valid at time T?"
                         Scored against expected_valid (bool).

The runner is deliberately lightweight. v0.1 ships with the stub
contradiction detector for CI-friendly smoke-testing; plug in an
NLIContradictionDetector for the real benchmark numbers.

Usage:
    python -m eval.belief_eval \\
        --scenarios eval/belief_eval_data/seed_scenarios.jsonl \\
        --output runs/belief_eval/results.json \\
        [--detector stub|nli] [--verbose]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from patha.belief.contradiction import (
    ContradictionDetector,
    NLIContradictionDetector,
    StubContradictionDetector,
)
from patha.belief.layer import BeliefLayer
from patha.belief.store import BeliefStore


# ─── Scenario data model ─────────────────────────────────────────────

@dataclass
class Proposition:
    text: str
    asserted_at: datetime
    session: str


@dataclass
class Question:
    q: str
    type: str  # "current_belief" | "validity_at_time"
    expected_current_contains: list[str] = field(default_factory=list)
    expected_superseded_contains: list[str] = field(default_factory=list)
    expected_valid: bool | None = None
    at_time: datetime | None = None


@dataclass
class Scenario:
    id: str
    family: str
    propositions: list[Proposition]
    questions: list[Question]


def _parse_scenario(d: dict) -> Scenario:
    props = [
        Proposition(
            text=p["text"],
            asserted_at=datetime.fromisoformat(p["asserted_at"]),
            session=p["session"],
        )
        for p in d["propositions"]
    ]
    qs: list[Question] = []
    for qd in d["questions"]:
        qs.append(
            Question(
                q=qd["q"],
                type=qd["type"],
                expected_current_contains=qd.get("expected_current_contains", []),
                expected_superseded_contains=qd.get("expected_superseded_contains", []),
                expected_valid=qd.get("expected_valid"),
                at_time=(
                    datetime.fromisoformat(qd["at_time"])
                    if qd.get("at_time")
                    else None
                ),
            )
        )
    return Scenario(
        id=d["id"],
        family=d["family"],
        propositions=props,
        questions=qs,
    )


def load_scenarios(path: str | Path) -> list[Scenario]:
    path = Path(path)
    scenarios: list[Scenario] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            scenarios.append(_parse_scenario(json.loads(line)))
    return scenarios


# ─── Per-question scoring ────────────────────────────────────────────

@dataclass
class QuestionResult:
    scenario_id: str
    family: str
    question: str
    type: str
    correct: bool
    tokens_in_summary: int
    details: dict = field(default_factory=dict)


def _score_current_belief(
    q: Question, current_props: list[str], superseded_props: list[str]
) -> tuple[bool, dict]:
    """Score a 'current belief' question.

    Correct iff:
      - every expected_current_contains term appears in at least one
        current proposition (case-insensitive substring match), AND
      - no expected_superseded_contains term appears in a current
        proposition (the superseded belief must not leak through).
    """
    current_joined = " | ".join(current_props).lower()
    superseded_joined = " | ".join(superseded_props).lower()

    missing_current = [
        t for t in q.expected_current_contains
        if t.lower() not in current_joined
    ]
    leaked_superseded = [
        t for t in q.expected_superseded_contains
        if t.lower() in current_joined
    ]
    hit_in_history = [
        t for t in q.expected_superseded_contains
        if t.lower() in superseded_joined
    ]

    correct = not missing_current and not leaked_superseded
    details = {
        "missing_current_terms": missing_current,
        "leaked_superseded_terms": leaked_superseded,
        "superseded_found_in_history": hit_in_history,
        "current_propositions": current_props,
        "superseded_propositions": superseded_props,
    }
    return correct, details


def _score_validity_at_time(
    q: Question, validity_verdict: bool | None
) -> tuple[bool, dict]:
    """Score a 'validity_at_time' question.

    Correct iff the system's verdict matches expected_valid.
    A None verdict (no matching belief found) is treated as False.
    """
    effective = validity_verdict if validity_verdict is not None else False
    correct = effective == q.expected_valid
    return correct, {
        "verdict": effective,
        "expected": q.expected_valid,
    }


# ─── Scenario runner ─────────────────────────────────────────────────

def run_scenario(
    scenario: Scenario,
    detector: ContradictionDetector,
    *,
    verbose: bool = False,
) -> list[QuestionResult]:
    """Ingest a scenario and evaluate its questions.

    Each scenario gets a fresh BeliefLayer — no cross-scenario leakage.
    """
    layer = BeliefLayer(store=BeliefStore(), detector=detector)

    # Ingest in temporal order
    props_sorted = sorted(scenario.propositions, key=lambda p: p.asserted_at)
    all_belief_ids: list[str] = []
    for i, p in enumerate(props_sorted):
        ev = layer.ingest(
            proposition=p.text,
            asserted_at=p.asserted_at,
            asserted_in_session=p.session,
            source_proposition_id=f"{scenario.id}-p{i}",
        )
        all_belief_ids.append(ev.new_belief.id)
        if verbose:
            print(f"  [{scenario.id}] ingest {ev.action}: {p.text[:60]}...")

    # Run questions
    results: list[QuestionResult] = []
    for q in scenario.questions:
        at_time = q.at_time if q.at_time is not None else datetime(2030, 1, 1)
        query_result = layer.query(
            all_belief_ids, at_time=at_time, include_history=True
        )
        current_props = [b.proposition for b in query_result.current]
        superseded_props = [b.proposition for b in query_result.history]

        if q.type == "current_belief":
            correct, details = _score_current_belief(
                q, current_props, superseded_props
            )
        elif q.type == "validity_at_time":
            # For validity_at_time: ingest once, query at the specific
            # time — 'valid' = at least one current belief returned.
            valid = len(query_result.current) > 0
            correct, details = _score_validity_at_time(q, valid)
        else:
            correct = False
            details = {"error": f"unknown question type: {q.type}"}

        results.append(
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

    return results


# ─── Aggregation ─────────────────────────────────────────────────────

@dataclass
class EvalSummary:
    n_scenarios: int
    n_questions: int
    n_correct: int
    accuracy: float
    by_family: dict[str, dict[str, float | int]]
    by_type: dict[str, dict[str, float | int]]
    avg_tokens_in_summary: float


def summarise(results: list[QuestionResult]) -> EvalSummary:
    n = len(results)
    n_correct = sum(1 for r in results if r.correct)
    by_family: dict[str, dict[str, float | int]] = {}
    by_type: dict[str, dict[str, float | int]] = {}

    for r in results:
        by_family.setdefault(
            r.family, {"total": 0, "correct": 0}
        )
        by_family[r.family]["total"] = int(by_family[r.family]["total"]) + 1
        by_family[r.family]["correct"] = (
            int(by_family[r.family]["correct"]) + (1 if r.correct else 0)
        )

        by_type.setdefault(r.type, {"total": 0, "correct": 0})
        by_type[r.type]["total"] = int(by_type[r.type]["total"]) + 1
        by_type[r.type]["correct"] = (
            int(by_type[r.type]["correct"]) + (1 if r.correct else 0)
        )

    for breakdown in (by_family, by_type):
        for stat in breakdown.values():
            total = int(stat["total"])
            correct = int(stat["correct"])
            stat["accuracy"] = correct / total if total else 0.0

    scenarios_seen = {r.scenario_id for r in results}
    avg_tokens = sum(r.tokens_in_summary for r in results) / max(n, 1)

    return EvalSummary(
        n_scenarios=len(scenarios_seen),
        n_questions=n,
        n_correct=n_correct,
        accuracy=n_correct / max(n, 1),
        by_family=by_family,
        by_type=by_type,
        avg_tokens_in_summary=avg_tokens,
    )


# ─── CLI ────────────────────────────────────────────────────────────

def _make_detector(name: str) -> ContradictionDetector:
    if name == "stub":
        return StubContradictionDetector()
    if name == "nli":
        return NLIContradictionDetector()
    raise ValueError(
        f"unknown detector {name!r}; choose 'stub' or 'nli'"
    )


def run(
    scenarios_path: str | Path,
    output_path: str | Path,
    detector_name: str = "stub",
    verbose: bool = False,
) -> EvalSummary:
    """Run BeliefEval and return the summary.

    Also writes a JSON results file with per-question details.
    """
    scenarios = load_scenarios(scenarios_path)
    detector = _make_detector(detector_name)
    all_results: list[QuestionResult] = []
    for sc in scenarios:
        all_results.extend(run_scenario(sc, detector, verbose=verbose))

    summary = summarise(all_results)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "scenarios_path": str(scenarios_path),
        "detector": detector_name,
        "summary": asdict(summary),
        "results": [asdict(r) for r in all_results],
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Patha BeliefEval runner")
    parser.add_argument(
        "--scenarios",
        default="eval/belief_eval_data/seed_scenarios.jsonl",
    )
    parser.add_argument(
        "--output", default="runs/belief_eval/results.json"
    )
    parser.add_argument(
        "--detector",
        choices=["stub", "nli"],
        default="stub",
        help="Contradiction detector (stub = heuristic for CI; nli = DeBERTa)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    print(f"Loading scenarios from {args.scenarios}...")
    summary = run(
        args.scenarios,
        args.output,
        detector_name=args.detector,
        verbose=args.verbose,
    )

    print()
    print("=" * 50)
    print(f"BeliefEval ({args.detector} detector)")
    print("=" * 50)
    print(f"  Scenarios:  {summary.n_scenarios}")
    print(f"  Questions:  {summary.n_questions}")
    print(f"  Accuracy:   {summary.accuracy:.3f} ({summary.n_correct}/{summary.n_questions})")
    print(f"  Avg tokens/summary: {summary.avg_tokens_in_summary:.0f}")
    print()
    print("  By family:")
    for fam, stat in summary.by_family.items():
        print(f"    {fam}: {stat['accuracy']:.3f} ({stat['correct']}/{stat['total']})")
    print()
    print("  By question type:")
    for t, stat in summary.by_type.items():
        print(f"    {t}: {stat['accuracy']:.3f} ({stat['correct']}/{stat['total']})")
    print()
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
