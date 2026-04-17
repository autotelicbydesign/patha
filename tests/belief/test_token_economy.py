"""Tests for token-economy measurement."""

from __future__ import annotations

from datetime import datetime

import pytest

from eval.belief_eval import Proposition, Question, Scenario
from eval.token_economy import (
    MeasurementRow,
    approx_tokens,
    measure,
)


# ─── Token counter ───────────────────────────────────────────────────

class TestApproxTokens:
    def test_short_text(self) -> None:
        assert approx_tokens("hi") >= 1

    def test_longer_text_more_tokens(self) -> None:
        short = approx_tokens("short")
        long = approx_tokens("this is a much longer string with many more words")
        assert long > short

    def test_empty_never_zero(self) -> None:
        assert approx_tokens("") >= 1


# ─── measure() ───────────────────────────────────────────────────────

def _simple_scenario(sid: str = "test-1") -> Scenario:
    return Scenario(
        id=sid,
        family="preference_supersession",
        propositions=[
            Proposition(
                text="I love sushi weekly",
                asserted_at=datetime(2023, 6, 1),
                session="s1",
            ),
            Proposition(
                text="I never eat sushi anymore",
                asserted_at=datetime(2024, 2, 1),
                session="s2",
            ),
        ],
        questions=[
            Question(
                q="What do I currently believe about sushi?",
                type="current_belief",
                expected_current_contains=["never eat sushi"],
                expected_superseded_contains=["love sushi"],
            )
        ],
    )


class TestMeasure:
    def test_produces_row_per_strategy_size_question(self) -> None:
        scenarios = [_simple_scenario()]
        report = measure(scenarios, memory_sizes=[20, 100])
        # 2 sizes * 1 scenario * 1 question * 3 strategies = 6 rows
        assert len(report.rows) == 6
        strategies_seen = {r.strategy for r in report.rows}
        assert strategies_seen == {"naive_rag", "structured", "direct_answer"}

    def test_direct_answer_uses_no_llm_tokens(self) -> None:
        scenarios = [_simple_scenario()]
        report = measure(scenarios, memory_sizes=[20])
        direct = [r for r in report.rows if r.strategy == "direct_answer"]
        for r in direct:
            assert r.llm_called is False
            assert r.tokens_in == 0

    def test_naive_rag_always_calls_llm(self) -> None:
        scenarios = [_simple_scenario()]
        report = measure(scenarios, memory_sizes=[20, 100])
        naive = [r for r in report.rows if r.strategy == "naive_rag"]
        for r in naive:
            assert r.llm_called is True
            assert r.tokens_in > 0

    def test_naive_rag_grows_with_memory(self) -> None:
        scenarios = [_simple_scenario()]
        report = measure(scenarios, memory_sizes=[20, 1000])
        by_size: dict[int, list[MeasurementRow]] = {}
        for r in report.rows:
            if r.strategy == "naive_rag":
                by_size.setdefault(r.memory_size, []).append(r)
        small_mean = sum(r.tokens_in for r in by_size[20]) / len(by_size[20])
        large_mean = sum(r.tokens_in for r in by_size[1000]) / len(by_size[1000])
        # Larger memory dumps more candidate context into the prompt
        assert large_mean >= small_mean

    def test_summary_by_strategy_shape(self) -> None:
        scenarios = [_simple_scenario()]
        report = measure(scenarios, memory_sizes=[20])
        summary = report.summary_by_strategy()
        assert set(summary.keys()) == {"naive_rag", "structured", "direct_answer"}
        for stats in summary.values():
            assert "mean_tokens_in" in stats
            assert "mean_compression_vs_naive" in stats
            assert "llm_call_rate" in stats

    def test_compression_ratios_are_sane(self) -> None:
        """Direct-answer compression vs naive should be high; structured
        should be >= 1x for non-trivial context."""
        scenarios = [_simple_scenario()]
        report = measure(scenarios, memory_sizes=[100])
        summary = report.summary_by_strategy()

        # direct_answer should be the most compressed (or tied)
        direct_comp = summary["direct_answer"]["mean_compression_vs_naive"]
        naive_comp = summary["naive_rag"]["mean_compression_vs_naive"]
        assert direct_comp >= naive_comp  # 1.00 naive, higher for direct
        # Direct should save real tokens on a lookup query
        assert direct_comp > 2.0
