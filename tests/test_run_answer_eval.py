"""Tests for the Phase 3 runner CLI (eval/run_answer_eval.py).

The runner wires answer_eval.py to LongMemEval-shaped haystack JSON.
These tests exercise:

  - the per-question memory builder (haystack ingest → recall-ready Memory)
  - the LLM/scorer factories (string name → callable)
  - main() end-to-end on a tiny synthetic fixture (1 question, null LLM,
    numeric scorer) with --output and --max-questions

The full LongMemEval 78q run is intentionally NOT a unit test — it's
~2 minutes and best run by hand or a slow-CI job.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.answer_eval import (
    ClaudeLLM,
    NullTemplateLLM,
    OllamaLLM,
)
from eval.run_answer_eval import (
    _build_llm,
    _build_memory_for_question,
    _build_scorer,
    main,
)


# ─── Synthetic fixture ──────────────────────────────────────────────


def _tiny_question() -> dict:
    """One LongMemEval-shaped question that NullTemplateLLM can answer
    (since the gold is a number that appears verbatim in the haystack)."""
    return {
        "question_id": "tiny-001",
        "question_type": "knowledge-update",
        "question": "How much did I spend on the saddle?",
        "question_date": "2024/01/05 (Fri) 12:00",
        "answer": "$50",
        "answer_session_ids": ["sess-1"],
        "haystack_dates": ["2024/01/01 (Mon) 09:00"],
        "haystack_session_ids": ["sess-1"],
        "haystack_sessions": [[
            {"role": "user", "content": "I bought a $50 saddle for my bike."},
            {"role": "assistant", "content": "Nice, what kind?"},
        ]],
    }


# ─── Helper-level tests ─────────────────────────────────────────────


class TestFactories:
    def test_build_llm_null(self):
        llm = _build_llm("null", ollama_model="x", claude_model="y")
        assert isinstance(llm, NullTemplateLLM)

    def test_build_llm_ollama_constructible(self):
        """OllamaLLM is constructible without network — only call() reaches out."""
        llm = _build_llm("ollama", ollama_model="llama3.2:3b", claude_model="x")
        assert isinstance(llm, OllamaLLM)
        assert llm.model == "llama3.2:3b"

    def test_build_llm_claude_constructible(self):
        """ClaudeLLM is constructible without API key — only call() needs it."""
        llm = _build_llm("claude", ollama_model="x", claude_model="claude-test")
        assert isinstance(llm, ClaudeLLM)
        assert llm.model == "claude-test"

    def test_build_llm_unknown_raises(self):
        with pytest.raises(ValueError, match="unknown --llm"):
            _build_llm("nope", ollama_model="x", claude_model="y")

    def test_build_scorer_known(self):
        for name in ["normalised", "numeric", "overlap", "embedding"]:
            scorer = _build_scorer(name)
            assert callable(scorer)

    def test_build_scorer_judge_requires_judge_llm(self):
        with pytest.raises(ValueError, match="--judge-llm"):
            _build_scorer("judge", judge_llm=None)

    def test_build_scorer_judge_with_llm(self):
        class _JudgeStub:
            def __call__(self, prompt: str) -> str:
                return "MATCH"
        scorer = _build_scorer("judge", judge_llm=_JudgeStub())
        assert callable(scorer)
        assert scorer("anything", "gold") is True

    def test_build_scorer_unknown_raises(self):
        with pytest.raises(ValueError, match="unknown --scorer"):
            _build_scorer("nope")


# ─── Memory-builder tests ───────────────────────────────────────────


class TestMemoryBuilder:
    def test_ingests_user_turns_only(self, tmp_path: Path):
        """The builder ingests USER content; assistant turns are skipped."""
        q = _tiny_question()
        mem = _build_memory_for_question(
            q, ingest_full=True,
            store_path=tmp_path / "store.jsonl",
        )
        # The belief store should hold the user proposition (one or more
        # propositions from "I bought a $50 saddle for my bike."), and
        # nothing from the assistant turn.
        beliefs = list(mem._patha.belief_layer.store.all())
        assert len(beliefs) >= 1
        joined = " ".join(b.proposition for b in beliefs).lower()
        assert "saddle" in joined
        assert "what kind" not in joined  # assistant turn, must be filtered

    def test_recall_finds_synthesis_value(self, tmp_path: Path):
        """End-to-end: builder + recall returns ganita value for $-question."""
        q = _tiny_question()
        mem = _build_memory_for_question(
            q, ingest_full=True,
            store_path=tmp_path / "store.jsonl",
        )
        rec = mem.recall("how much did I spend on the saddle?")
        # Saddle question is a synthesis intent; ganita should fire.
        assert rec.ganita is not None
        assert rec.ganita.value == 50.0


# ─── End-to-end CLI test ────────────────────────────────────────────


class TestMainEndToEnd:
    def test_main_writes_output_and_returns_zero(self, tmp_path: Path):
        """main() runs on a one-question fixture, exits 0 (correct), JSON
        output contains the expected aggregate + per-question outcome."""
        fixture = tmp_path / "tiny.json"
        fixture.write_text(json.dumps([_tiny_question()]))
        out = tmp_path / "results.json"

        rc = main([
            "--data", str(fixture),
            "--llm", "null",
            "--scorer", "numeric",
            "--ingest-full",
            "--output", str(out),
        ])
        assert rc == 0  # 1/1 correct → main returns 0

        payload = json.loads(out.read_text())
        assert payload["n"] == 1
        assert payload["correct"] == 1
        assert payload["accuracy"] == 1.0
        assert payload["llm"] == "null"
        assert payload["scorer"] == "numeric"
        # Per-question outcome present, structured correctly.
        assert len(payload["outcomes"]) == 1
        oc = payload["outcomes"][0]
        assert oc["question_id"] == "tiny-001"
        assert oc["correct"] is True
        assert oc["strategy"] == "ganita"

    def test_main_max_questions_truncates(self, tmp_path: Path):
        """--max-questions caps the run length even if the file has more."""
        q1, q2 = _tiny_question(), _tiny_question()
        q2["question_id"] = "tiny-002"
        fixture = tmp_path / "two.json"
        fixture.write_text(json.dumps([q1, q2]))
        out = tmp_path / "results.json"

        rc = main([
            "--data", str(fixture),
            "--llm", "null",
            "--scorer", "numeric",
            "--ingest-full",
            "--max-questions", "1",
            "--output", str(out),
        ])
        assert rc == 0
        payload = json.loads(out.read_text())
        assert payload["n"] == 1
