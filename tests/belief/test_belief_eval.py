"""Tests for the BeliefEval runner (fast; stub detector only)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from eval.belief_eval import (
    Proposition,
    Question,
    Scenario,
    load_scenarios,
    run_scenario,
    summarise,
)
from patha.belief.contradiction import StubContradictionDetector


# ─── Parsing ─────────────────────────────────────────────────────────

class TestLoadScenarios:
    def test_loads_seed_file(self) -> None:
        scenarios = load_scenarios(
            "eval/belief_eval_data/seed_scenarios.jsonl"
        )
        assert len(scenarios) == 20
        families = {s.family for s in scenarios}
        assert "preference_supersession" in families
        assert "factual_supersession" in families
        assert "temporally_bounded" in families

    def test_scenario_has_required_fields(self) -> None:
        scenarios = load_scenarios(
            "eval/belief_eval_data/seed_scenarios.jsonl"
        )
        sc = scenarios[0]
        assert sc.id
        assert sc.family
        assert len(sc.propositions) >= 1
        assert len(sc.questions) >= 1


# ─── Scoring ─────────────────────────────────────────────────────────

def _make_scenario_current() -> Scenario:
    """Scenario with a trivially-detectable supersession by the stub."""
    return Scenario(
        id="test-1",
        family="preference_supersession",
        propositions=[
            Proposition(
                text="I love sushi regularly",
                asserted_at=datetime(2023, 1, 1),
                session="s1",
            ),
            Proposition(
                text="I never eat sushi anymore",
                asserted_at=datetime(2024, 1, 1),
                session="s2",
            ),
        ],
        questions=[
            Question(
                q="What does the user currently believe about sushi?",
                type="current_belief",
                expected_current_contains=["never eat sushi"],
                expected_superseded_contains=["love sushi"],
            )
        ],
    )


def _make_scenario_validity() -> Scenario:
    return Scenario(
        id="test-2",
        family="temporally_bounded",
        propositions=[
            Proposition(
                text="I am on holiday for three weeks starting March 1",
                asserted_at=datetime(2024, 3, 1),
                session="s1",
            )
        ],
        questions=[
            Question(
                q="Is the user on holiday on March 15?",
                type="validity_at_time",
                at_time=datetime(2024, 3, 15),
                expected_valid=True,
            ),
            Question(
                q="Is the user on holiday on May 1?",
                type="validity_at_time",
                at_time=datetime(2024, 5, 1),
                expected_valid=False,
            ),
        ],
    )


class TestRunScenario:
    def test_stub_detects_clear_supersession(self) -> None:
        # The stub's asymmetric-negation-with-overlap heuristic can
        # handle this specific pair: "love sushi" vs "never eat sushi"
        # share "sushi" content and have asymmetric negation.
        layer_threshold_low_scenario = _make_scenario_current()

        # Use a lower threshold so stub's 0.6 confidence passes.
        from patha.belief.layer import BeliefLayer
        from patha.belief.store import BeliefStore

        layer = BeliefLayer(
            store=BeliefStore(),
            detector=StubContradictionDetector(),
            contradiction_threshold=0.5,
        )

        # Manually run through the scenario using the lower-threshold
        # layer (the default runner uses 0.75).
        all_ids = []
        for i, p in enumerate(
            sorted(
                layer_threshold_low_scenario.propositions,
                key=lambda x: x.asserted_at,
            )
        ):
            ev = layer.ingest(
                proposition=p.text,
                asserted_at=p.asserted_at,
                asserted_in_session=p.session,
                source_proposition_id=f"p{i}",
            )
            all_ids.append(ev.new_belief.id)

        result = layer.query(
            all_ids,
            at_time=datetime(2030, 1, 1),
            include_history=True,
        )
        # Second belief should be current, first should be in history
        assert len(result.current) == 1
        assert "never eat" in result.current[0].proposition.lower()

    def test_validity_questions_scored_correctly(self) -> None:
        scenario = _make_scenario_validity()
        results = run_scenario(
            scenario, StubContradictionDetector()
        )
        assert len(results) == 2
        assert all(r.correct for r in results), (
            f"results: {[(r.question, r.details) for r in results]}"
        )

    def test_fresh_layer_per_scenario(self) -> None:
        # Two scenarios shouldn't bleed into each other
        s1 = _make_scenario_validity()
        s2 = _make_scenario_validity()
        s2.id = "test-3"
        detector = StubContradictionDetector()
        r1 = run_scenario(s1, detector)
        r2 = run_scenario(s2, detector)
        assert len(r1) == 2
        assert len(r2) == 2


# ─── Aggregation ─────────────────────────────────────────────────────

class TestSummarise:
    def test_empty(self) -> None:
        s = summarise([])
        assert s.n_questions == 0
        assert s.accuracy == 0.0
        assert s.by_family == {}

    def test_mixed_results(self) -> None:
        from eval.belief_eval import QuestionResult

        results = [
            QuestionResult(
                scenario_id="a",
                family="factual_supersession",
                question="q1",
                type="current_belief",
                correct=True,
                tokens_in_summary=30,
            ),
            QuestionResult(
                scenario_id="b",
                family="factual_supersession",
                question="q2",
                type="current_belief",
                correct=False,
                tokens_in_summary=10,
            ),
            QuestionResult(
                scenario_id="c",
                family="temporally_bounded",
                question="q3",
                type="validity_at_time",
                correct=True,
                tokens_in_summary=50,
            ),
        ]
        s = summarise(results)
        assert s.n_questions == 3
        assert s.n_correct == 2
        assert s.accuracy == pytest.approx(2 / 3)
        assert s.by_family["factual_supersession"]["accuracy"] == 0.5
        assert s.by_family["temporally_bounded"]["accuracy"] == 1.0
        assert s.avg_tokens_in_summary == pytest.approx(30)


# ─── End-to-end smoke ────────────────────────────────────────────────

class TestEndToEnd:
    def test_seed_runs_without_errors(self, tmp_path: Path) -> None:
        from eval.belief_eval import run as run_full

        out = tmp_path / "r.json"
        summary = run_full(
            scenarios_path="eval/belief_eval_data/seed_scenarios.jsonl",
            output_path=out,
            detector_name="stub",
            verbose=False,
        )
        assert summary.n_scenarios == 20
        assert summary.n_questions >= 20
        # Output file exists and is parseable
        data = json.loads(out.read_text())
        assert "summary" in data and "results" in data

    def test_stub_baseline_approximates_expected(
        self, tmp_path: Path
    ) -> None:
        """Smoke test that stub baseline stays within a known band.

        This is not a correctness target — it's a regression guard so
        that a silent change in the stub heuristic or scoring logic
        doesn't pass unnoticed.
        """
        from eval.belief_eval import run as run_full

        summary = run_full(
            scenarios_path="eval/belief_eval_data/seed_scenarios.jsonl",
            output_path=tmp_path / "r.json",
            detector_name="stub",
        )
        # Stub gets validity questions right (100%), current-belief
        # mostly wrong. Expect overall accuracy 30-60% as a wide band.
        assert 0.25 <= summary.accuracy <= 0.65
        # Validity should be perfect
        assert (
            summary.by_type["validity_at_time"]["accuracy"]
            == pytest.approx(1.0)
        )
