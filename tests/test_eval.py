"""Tests for the eval harness: metrics, runner, and end-to-end on the synthetic fixture."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from eval.metrics import (
    EvalReport,
    QuestionResult,
    RETRIEVAL_STRATA,
    _classify_stratum,
)
from eval.runner import load_longmemeval, run_evaluation, save_results

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "synthetic_longmemeval.json"


# ─── Stratum classification ──────────────────────────────────────────

class TestStratumClassification:
    def test_single_session_types(self):
        assert _classify_stratum("single-session-user") == "single_session"
        assert _classify_stratum("single-session-assistant") == "single_session"
        assert _classify_stratum("single-session-preference") == "single_session"

    def test_multi_session(self):
        assert _classify_stratum("multi-session") == "multi_session"

    def test_temporal_reasoning(self):
        assert _classify_stratum("temporal-reasoning") == "temporal_reasoning"

    def test_knowledge_update(self):
        assert _classify_stratum("knowledge-update") == "knowledge_update"

    def test_abstention(self):
        assert _classify_stratum("abstention") == "abstention"

    def test_abstention_variant(self):
        assert _classify_stratum("q005_abs") == "abstention"


# ─── QuestionResult metrics ──────────────────────────────────────────

class TestQuestionResult:
    def test_perfect_recall(self):
        qr = QuestionResult(
            question_id="q1", question_type="single-session-user",
            stratum="single_session",
            gold_session_ids=["s1"],
            retrieved_chunk_ids=["s1#t0#p0", "s2#t0#p0"],
        )
        assert qr.recall_any_at_k(5) == 1.0

    def test_miss_recall(self):
        qr = QuestionResult(
            question_id="q1", question_type="single-session-user",
            stratum="single_session",
            gold_session_ids=["s1"],
            retrieved_chunk_ids=["s2#t0#p0", "s3#t0#p0"],
        )
        assert qr.recall_any_at_k(5) == 0.0

    def test_recall_all_needs_all_sessions(self):
        qr = QuestionResult(
            question_id="q1", question_type="multi-session",
            stratum="multi_session",
            gold_session_ids=["s1", "s2"],
            retrieved_chunk_ids=["s1#t0#p0", "s3#t0#p0"],
        )
        assert qr.recall_any_at_k(5) == 1.0  # s1 is there
        assert qr.recall_all_at_k(5) == 0.0  # s2 is missing

    def test_recall_all_succeeds_when_all_present(self):
        qr = QuestionResult(
            question_id="q1", question_type="multi-session",
            stratum="multi_session",
            gold_session_ids=["s1", "s2"],
            retrieved_chunk_ids=["s1#t0#p0", "s2#t0#p0", "s3#t0#p0"],
        )
        assert qr.recall_all_at_k(5) == 1.0

    def test_empty_gold_returns_zero(self):
        qr = QuestionResult(
            question_id="q1", question_type="abstention",
            stratum="abstention",
            gold_session_ids=[],
            retrieved_chunk_ids=["s1#t0#p0"],
        )
        assert qr.recall_any_at_k(5) == 0.0

    def test_ndcg_perfect(self):
        qr = QuestionResult(
            question_id="q1", question_type="single-session-user",
            stratum="single_session",
            gold_session_ids=["s1"],
            retrieved_chunk_ids=["s1#t0#p0", "s2#t0#p0"],
        )
        assert qr.ndcg_at_k(5) == 1.0  # gold at rank 1


# ─── EvalReport ───────────────────────────────────────────────────────

class TestEvalReport:
    def test_abstention_excluded_from_retrieval_metrics(self):
        report = EvalReport(results=[
            QuestionResult("q1", "single-session-user", "single_session",
                           ["s1"], ["s1#t0#p0"]),
            QuestionResult("q2", "abstention", "abstention",
                           [], ["s2#t0#p0"]),
        ])
        # Only 1 retrieval question
        assert report.summary()["retrieval_questions"] == 1
        assert report.recall_any_at_k(5) == 1.0  # only q1 counts

    def test_per_stratum_breakdown(self):
        report = EvalReport(results=[
            QuestionResult("q1", "single-session-user", "single_session",
                           ["s1"], ["s1#t0#p0"]),
            QuestionResult("q2", "multi-session", "multi_session",
                           ["s1", "s2"], ["s3#t0#p0"]),
        ])
        strata = report.per_stratum_recall_any_at_k(5)
        assert strata["single_session"] == 1.0
        assert strata["multi_session"] == 0.0

    def test_summary_has_expected_keys(self):
        report = EvalReport(results=[
            QuestionResult("q1", "single-session-user", "single_session",
                           ["s1"], ["s1#t0#p0"]),
        ])
        s = report.summary(ks=[5, 10])
        assert "recall_any@5" in s
        assert "recall_any@10" in s
        assert "per_stratum_recall_any@5" in s
        assert "total_questions" in s


# ─── Fixture loading ─────────────────────────────────────────────────

class TestLoadFixture:
    def test_load_synthetic(self):
        data = load_longmemeval(FIXTURE_PATH)
        assert len(data) == 5
        assert data[0]["question_id"] == "q001"
        assert "haystack_sessions" in data[0]
        assert "answer_session_ids" in data[0]

    def test_question_types_present(self):
        data = load_longmemeval(FIXTURE_PATH)
        types = {e["question_type"] for e in data}
        assert "single-session-user" in types
        assert "multi-session" in types
        assert "knowledge-update" in types
        assert "temporal-reasoning" in types
        assert "abstention" in types


# ─── End-to-end runner ────────────────────────────────────────────────

class TestRunner:
    def test_run_on_synthetic_fixture(self):
        data = load_longmemeval(FIXTURE_PATH)
        report = run_evaluation(data, use_songline=True, verbose=False)

        assert len(report.results) == 5
        summary = report.summary()
        assert summary["total_questions"] == 5
        # 4 retrieval questions (1 abstention excluded)
        assert summary["retrieval_questions"] == 4

        # With stub embedder, exact-text match won't work semantically,
        # but the harness should run without errors
        assert 0.0 <= summary["recall_any@5"] <= 1.0

    def test_run_without_songline(self):
        data = load_longmemeval(FIXTURE_PATH)
        report = run_evaluation(data, use_songline=False, verbose=False)
        assert len(report.results) == 5

    def test_save_and_reload_results(self):
        data = load_longmemeval(FIXTURE_PATH)
        report = run_evaluation(data, use_songline=False, verbose=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "results.json"
            save_results(report, out_path)
            assert out_path.exists()

            with open(out_path) as f:
                loaded = json.load(f)
            assert "summary" in loaded
            assert "per_question" in loaded
            assert len(loaded["per_question"]) == 5
