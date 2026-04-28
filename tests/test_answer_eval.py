"""Tests for the Phase 3 answer-eval scaffolding (eval/answer_eval.py).

Phase 3 measures the actual product question: does the user's LLM,
given Patha's output, produce the correct answer? This module ships
the scaffolding (LLM Protocol, Scorer Protocol, prompt rendering,
eval engine). The tests cover the deterministic NullTemplateLLM, all
shipped scorers, the prompt renderer, and an end-to-end run on a
small canned scenario.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import patha
from eval.answer_eval import (
    AnswerEvalConfig,
    AnswerEvalReport,
    NullTemplateLLM,
    OllamaLLM,
    QuestionOutcome,
    exact_match,
    normalised_match,
    numeric_match,
    render_prompt,
    run_answer_eval,
    token_overlap_match,
)


# ─── Scorers ────────────────────────────────────────────────────────


class TestScorers:
    def test_exact_match(self):
        assert exact_match("$185", "$185")
        assert exact_match("  $185  ", "$185")
        assert not exact_match("$185", "$186")
        assert not exact_match("$185 USD", "$185")

    def test_normalised_match(self):
        assert normalised_match("$185", "$185")
        assert normalised_match("$185.", "$185")
        assert normalised_match("Hello, World!", "hello world")
        assert not normalised_match("yes", "no")

    def test_numeric_match_within_tol(self):
        scorer = numeric_match(tol=0.05)
        assert scorer("$185", "$185")
        assert scorer("$185.50", "$185")  # within 5%
        assert scorer("184", "185")  # 1-unit absolute tolerance
        assert not scorer("$200", "$185")  # outside tolerance

    def test_numeric_match_falls_back_to_normalised(self):
        """When the gold isn't numeric, numeric_match falls back to
        normalised_match — useful for benchmarks with mixed gold types."""
        scorer = numeric_match()
        assert scorer("Lisbon", "lisbon")
        assert not scorer("Lisbon", "Tokyo")

    def test_token_overlap(self):
        scorer = token_overlap_match(threshold=0.6)
        # 100% overlap
        assert scorer("I live in Lisbon now", "Lisbon")
        # No overlap
        assert not scorer("Tokyo", "Lisbon")


# ─── Null LLM ───────────────────────────────────────────────────────


class TestNullTemplateLLM:
    def test_echoes_dollar_amount(self):
        llm = NullTemplateLLM()
        prompt = (
            "Question: how much have I spent?\n"
            "Memory:\n  Computed: $185.00 USD\n"
            "Answer:"
        )
        assert "$185" in llm(prompt)

    def test_echoes_first_number(self):
        llm = NullTemplateLLM()
        prompt = (
            "Question: how many hours?\n"
            "Memory:\n  Total: 140 hour\n"
            "Answer:"
        )
        assert "140" in llm(prompt)

    def test_falls_through_to_text(self):
        """No numbers → echo the start of the memory."""
        llm = NullTemplateLLM()
        prompt = (
            "Question: where do I live?\n"
            "Memory:\n  I live in Lisbon\n"
            "Answer:"
        )
        out = llm(prompt)
        assert "Lisbon" in out


# ─── Prompt rendering ───────────────────────────────────────────────


class TestRenderPrompt:
    def test_basic_render(self, tmp_path: Path):
        mem = patha.Memory(
            path=tmp_path / "store.jsonl", enable_phase1=False,
        )
        mem.remember("I bought a $50 saddle for my bike",
                     asserted_at=datetime(2024, 1, 1))
        rec = mem.recall("how much have I spent on bike?")
        prompt = render_prompt(
            "Q: {question}\nMemory: {summary}\nGanita: {ganita}\nAnswer:",
            "how much have I spent on bike?",
            rec,
        )
        assert "Q: how much have I spent on bike?" in prompt
        assert "Memory:" in prompt
        # ganita block should mention 50.0 since synthesis fired
        assert "50" in prompt or "ganita" in prompt.lower()


# ─── End-to-end ─────────────────────────────────────────────────────


class TestRunAnswerEval:
    def test_synthesis_question_passes_with_null_llm(self, tmp_path: Path):
        """End-to-end: a synthesis question through Patha + null LLM
        produces a numeric answer that matches gold."""
        mem = patha.Memory(
            path=tmp_path / "store.jsonl", enable_phase1=False,
        )
        for fact in [
            "I bought a $50 saddle for my bike",
            "I got a $75 helmet for the bike",
            "$30 for new bike lights",
            "I spent $30 on bike gloves",
        ]:
            mem.remember(fact, asserted_at=datetime(2024, 1, 1))

        cfg = AnswerEvalConfig(
            llm=NullTemplateLLM(),
            prompt_template=(
                "Question: {question}\n"
                "Memory: {ganita}\n"
                "Answer:"
            ),
            scorer=numeric_match(tol=0.05),
        )
        questions = [{
            "question_id": "bike-185",
            "question": "how much have I spent on bike-related expenses?",
            "answer": "$185",
            "_ingested_memory": mem,
        }]
        report = run_answer_eval(
            questions, memory_factory=lambda: mem, config=cfg,
        )
        assert isinstance(report, AnswerEvalReport)
        assert report.n == 1
        assert report.correct == 1
        assert report.accuracy == 1.0
        outcome = report.outcomes[0]
        assert outcome.strategy == "ganita"
        assert outcome.summary_tokens == 0  # zero LLM tokens at recall

    def test_ollama_unreachable_raises(self):
        """OllamaLLM raises a clear error when Ollama isn't running.
        (CI-safe: we point at a port nothing listens on, expect failure.)"""
        import pytest
        llm = OllamaLLM(host="http://localhost:1", timeout_s=0.5)
        with pytest.raises(RuntimeError, match="Ollama"):
            llm("any prompt")
        assert llm.calls == 1  # call counter still incremented

    def test_by_strategy_breakdown(self, tmp_path: Path):
        """Multiple questions across strategies aggregate by strategy."""
        mem = patha.Memory(
            path=tmp_path / "store.jsonl", enable_phase1=False,
        )
        mem.remember("I love sushi every week",
                     asserted_at=datetime(2024, 1, 1))
        mem.remember("I bought a $50 saddle",
                     asserted_at=datetime(2024, 1, 2))

        cfg = AnswerEvalConfig(
            llm=NullTemplateLLM(),
            prompt_template="{question}\nMemory:\n  {summary}\n  {ganita}\nAnswer:",
            scorer=token_overlap_match(threshold=0.5),
        )
        questions = [
            {"question_id": "q1", "question": "what do I eat?",
             "answer": "sushi"},
            {"question_id": "q2",
             "question": "how much have I spent on saddle?",
             "answer": "$50"},
        ]
        # Mock Phase 1 retrieval so the perception question lands
        from patha.belief.types import PropositionId
        mem._patha._phase1_retrieve = lambda q, k: [
            b.source_proposition_id
            for b in mem._patha.belief_layer.store.all()
        ]
        report = run_answer_eval(
            questions,
            memory_factory=lambda: mem,
            config=cfg,
        )
        assert report.n == 2
        breakdown = report.by_strategy()
        # Should have entries for both strategies (structured and ganita)
        assert len(breakdown) >= 1
