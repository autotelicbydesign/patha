"""Integration tests for IntegratedPatha — Phase 1 + Phase 2 end-to-end."""

from __future__ import annotations

from datetime import datetime

import pytest

from patha.belief.contradiction import StubContradictionDetector
from patha.belief.layer import BeliefLayer
from patha.belief.store import BeliefStore
from patha.belief.types import ContradictionLabel, ContradictionResult
from patha.integrated import IntegratedPatha, IntegratedResponse


class _ScriptedDetector:
    def __init__(self, scripts):
        self.scripts = scripts

    def detect(self, p1, p2):
        return self.scripts.get(
            (p1, p2),
            ContradictionResult(label=ContradictionLabel.NEUTRAL, confidence=0.5),
        )

    def detect_batch(self, pairs):
        return [self.detect(p1, p2) for p1, p2 in pairs]


# ─── Setup helper ────────────────────────────────────────────────────

def _build_integrated_with_supersession() -> IntegratedPatha:
    """Standard scenario: 'I love sushi' -> 'I am avoiding raw fish'."""
    scripts = {
        (
            "I love sushi and eat it every week",
            "I am avoiding raw fish on my doctor's advice",
        ): ContradictionResult(
            label=ContradictionLabel.CONTRADICTS, confidence=0.95
        ),
    }
    layer = BeliefLayer(store=BeliefStore(), detector=_ScriptedDetector(scripts))
    patha = IntegratedPatha(belief_layer=layer)

    patha.ingest(
        proposition="I love sushi and eat it every week",
        asserted_at=datetime(2023, 6, 1),
        asserted_in_session="s1",
        source_proposition_id="prop-1",
    )
    patha.ingest(
        proposition="I am avoiding raw fish on my doctor's advice",
        asserted_at=datetime(2024, 2, 1),
        asserted_in_session="s2",
        source_proposition_id="prop-2",
    )
    return patha


# ─── Query routing ───────────────────────────────────────────────────

class TestQueryRouting:
    def test_lookup_query_gets_direct_answer(self) -> None:
        patha = _build_integrated_with_supersession()
        # Mock Phase 1: return all propositions in order
        patha._phase1_retrieve = lambda q, k: ["prop-1", "prop-2"]

        response = patha.query(
            "What do I currently believe about sushi?",
            at_time=datetime(2024, 6, 1),
        )
        assert response.strategy == "direct_answer"
        assert "avoiding raw fish" in response.answer
        assert response.tokens_in == 0
        assert response.prompt == ""

    def test_generation_query_gets_structured(self) -> None:
        patha = _build_integrated_with_supersession()
        patha._phase1_retrieve = lambda q, k: ["prop-1", "prop-2"]

        response = patha.query(
            "Summarise the user's relationship with sushi over the past year",
            at_time=datetime(2024, 6, 1),
        )
        assert response.strategy == "structured"
        assert response.prompt  # non-empty prompt to send to LLM
        assert "Current beliefs" in response.prompt
        assert response.tokens_in > 0

    def test_no_current_beliefs_falls_back_to_raw(self) -> None:
        layer = BeliefLayer(store=BeliefStore(), detector=StubContradictionDetector())
        patha = IntegratedPatha(belief_layer=layer)
        # No beliefs ingested; Phase 1 returns some proposition ids
        patha._phase1_retrieve = lambda q, k: ["phantom-1"]

        response = patha.query(
            "Summarise the user's life",
            at_time=datetime(2024, 6, 1),
        )
        # phantom-1 isn't in the belief store → no current beliefs → raw path
        assert response.strategy == "raw"
        # raw prompt is built even when no props resolved
        assert response.prompt


# ─── Source-proposition traceability ─────────────────────────────────

class TestProvenance:
    def test_direct_answer_carries_source_proposition_ids(self) -> None:
        patha = _build_integrated_with_supersession()
        patha._phase1_retrieve = lambda q, k: ["prop-1", "prop-2"]

        response = patha.query(
            "What do I currently believe about sushi?",
            at_time=datetime(2024, 6, 1),
        )
        assert response.strategy == "direct_answer"
        # Only the current belief's source is surfaced
        assert response.source_proposition_ids == ["prop-2"]

    def test_structured_carries_all_current_sources(self) -> None:
        patha = _build_integrated_with_supersession()
        patha._phase1_retrieve = lambda q, k: ["prop-1", "prop-2"]

        response = patha.query(
            "Summarise the user's diet decisions",
            at_time=datetime(2024, 6, 1),
        )
        assert response.strategy == "structured"
        assert "prop-2" in response.source_proposition_ids


# ─── Phase 1 integration via callable ────────────────────────────────

class TestPhase1Integration:
    def test_phase1_is_called_with_query_and_top_k(self) -> None:
        patha = _build_integrated_with_supersession()
        calls: list[tuple[str, int]] = []

        def fake_retrieve(query: str, k: int) -> list[str]:
            calls.append((query, k))
            return ["prop-1", "prop-2"]

        patha._phase1_retrieve = fake_retrieve
        patha.query("test query", phase1_top_k=7)
        assert calls == [("test query", 7)]

    def test_no_phase1_falls_back_to_all_current(self) -> None:
        patha = _build_integrated_with_supersession()
        # Force the no-phase1 path
        patha._phase1_retrieve = None
        response = patha.query(
            "What do I currently believe about sushi?",
            at_time=datetime(2024, 6, 1),
        )
        assert response.strategy == "direct_answer"
        assert "avoiding raw fish" in response.answer


# ─── Include history ────────────────────────────────────────────────

class TestHistory:
    def test_include_history_surfaces_lineage_in_structured(self) -> None:
        patha = _build_integrated_with_supersession()
        patha._phase1_retrieve = lambda q, k: ["prop-1", "prop-2"]

        response = patha.query(
            "Summarise the user's sushi journey",
            at_time=datetime(2024, 6, 1),
            include_history=True,
        )
        assert response.strategy == "structured"
        assert "love sushi" in response.prompt  # history surfaced
        assert "avoiding raw fish" in response.prompt  # current too
