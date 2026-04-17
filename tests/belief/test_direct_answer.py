"""Tests for direct-answer compression (D7 Option C)."""

from __future__ import annotations

from datetime import datetime

import pytest

from patha.belief.direct_answer import (
    DirectAnswer,
    DirectAnswerer,
    is_belief_lookup,
)
from patha.belief.store import BeliefStore


# ─── Lookup classification ───────────────────────────────────────────

class TestIsBeliefLookup:
    @pytest.mark.parametrize("q", [
        "What do I currently believe about sushi?",
        "what do you currently think about that?",
        "Where does the user live now?",
        "What is the user's current address?",
        "Is the user still avoiding raw fish?",
        "Does the user currently eat fish?",
        "What car does the user drive now?",
        "When did the user move?",
    ])
    def test_lookup_queries_classified(self, q: str) -> None:
        assert is_belief_lookup(q)

    @pytest.mark.parametrize("q", [
        "Summarise the user's journey with sushi over the past year.",
        "Tell me a story about the user's preferences.",
        "Why might the user have stopped eating sushi?",
        "Explain the relationship between diet and health.",
        "Give me advice on what to eat tonight.",
    ])
    def test_generation_queries_fall_through(self, q: str) -> None:
        # These should NOT be classified as lookups
        assert not is_belief_lookup(q)


# ─── DirectAnswerer ──────────────────────────────────────────────────

@pytest.fixture
def populated_store() -> tuple[BeliefStore, list[str]]:
    store = BeliefStore()
    b1 = store.add(
        proposition="I love sushi and eat it every week",
        asserted_at=datetime(2023, 6, 1),
        asserted_in_session="s1",
        source_proposition_id="p1",
        belief_id="b1",
    )
    b2 = store.add(
        proposition="I am avoiding raw fish on my doctor's advice",
        asserted_at=datetime(2024, 2, 1),
        asserted_in_session="s2",
        source_proposition_id="p2",
        belief_id="b2",
    )
    store.supersede("b1", "b2")
    return store, ["b1", "b2"]


class TestDirectAnswerer:
    def test_returns_none_for_non_lookup(
        self, populated_store: tuple[BeliefStore, list[str]]
    ) -> None:
        store, ids = populated_store
        answerer = DirectAnswerer(store)
        result = answerer.try_answer(
            "Tell me a story about sushi", ids, at_time=datetime(2024, 6, 1)
        )
        assert result is None

    def test_returns_none_when_no_current_belief(self) -> None:
        store = BeliefStore()
        answerer = DirectAnswerer(store)
        result = answerer.try_answer(
            "What do I currently believe about sushi?",
            [],
            at_time=datetime(2024, 6, 1),
        )
        assert result is None

    def test_answers_current_belief_lookup(
        self, populated_store: tuple[BeliefStore, list[str]]
    ) -> None:
        store, ids = populated_store
        answerer = DirectAnswerer(store)
        result = answerer.try_answer(
            "What do I currently believe about sushi?",
            ids,
            at_time=datetime(2024, 6, 1),
        )
        assert result is not None
        assert isinstance(result, DirectAnswer)
        # Only the superseding belief surfaces
        assert "avoiding raw fish" in result.text
        assert "love sushi" not in result.text  # explicitly not returned
        assert result.belief_ids == ["b2"]
        assert result.tokens_used > 0

    def test_change_queries_include_history(
        self, populated_store: tuple[BeliefStore, list[str]]
    ) -> None:
        store, ids = populated_store
        answerer = DirectAnswerer(store)
        result = answerer.try_answer(
            "When did the user change their view on sushi?",
            ids,
            at_time=datetime(2024, 6, 1),
        )
        assert result is not None
        assert result.include_history
        assert "avoiding raw fish" in result.text
        assert "love sushi" in result.text  # history surfaces here

    def test_explicit_history_flag_default(
        self, populated_store: tuple[BeliefStore, list[str]]
    ) -> None:
        store, ids = populated_store
        answerer = DirectAnswerer(store, include_history_by_default=True)
        result = answerer.try_answer(
            "What do I currently believe about sushi?",
            ids,
            at_time=datetime(2024, 6, 1),
        )
        assert result is not None
        assert result.include_history
        assert "love sushi" in result.text

    def test_compression_vs_raw(
        self, populated_store: tuple[BeliefStore, list[str]]
    ) -> None:
        """The direct answer uses far fewer tokens than raw RAG would."""
        store, ids = populated_store
        answerer = DirectAnswerer(store)
        result = answerer.try_answer(
            "What do I currently believe about sushi?",
            ids,
            at_time=datetime(2024, 6, 1),
        )
        assert result is not None
        # Rough baseline: a naive RAG would send all 2 raw propositions
        # AND a system prompt AND the question AND wait for LLM output.
        # Just the two propositions alone are ~80 chars; system prompt
        # adds ~500; question adds another ~50. So raw RAG input >= ~630
        # chars = ~158 tokens. Our direct answer is just the current
        # belief (~50 chars = ~13 tokens). That's a ~12x compression on
        # this minimal example, and compounds as memory grows.
        raw_rag_approx = sum(len(store.get(i).proposition) for i in ids)  # type: ignore[union-attr]
        raw_rag_approx += 500  # system prompt
        raw_rag_tokens = raw_rag_approx // 4
        assert result.tokens_used < raw_rag_tokens / 3  # at least 3x saving

    def test_disputed_belief_surfaces_caveat(self) -> None:
        store = BeliefStore()
        store.add(
            proposition="Ravi is the lead",
            asserted_at=datetime(2024, 1, 1),
            asserted_in_session="s1",
            source_proposition_id="p1",
            belief_id="a",
        )
        store.add(
            proposition="Emma is the lead",
            asserted_at=datetime(2024, 2, 1),
            asserted_in_session="s2",
            source_proposition_id="p2",
            belief_id="b",
        )
        store.dispute("a", "b")

        answerer = DirectAnswerer(store)
        result = answerer.try_answer(
            "Who is currently the lead?",
            ["a", "b"],
            at_time=datetime(2024, 6, 1),
        )
        assert result is not None
        assert "disputed" in result.text.lower()

    def test_expired_validity_filtered_out(self) -> None:
        from patha.belief.types import Validity

        store = BeliefStore()
        store.add(
            proposition="I am on paternity leave",
            asserted_at=datetime(2024, 3, 1),
            asserted_in_session="s1",
            source_proposition_id="p1",
            belief_id="a",
            validity=Validity(
                mode="dated_range",
                start=datetime(2024, 3, 1),
                end=datetime(2024, 6, 1),
                source="explicit",
            ),
        )
        answerer = DirectAnswerer(store)
        # Inside validity window
        r_inside = answerer.try_answer(
            "Is the user currently on paternity leave?",
            ["a"],
            at_time=datetime(2024, 4, 1),
        )
        assert r_inside is not None
        # Outside validity window
        r_outside = answerer.try_answer(
            "Is the user currently on paternity leave?",
            ["a"],
            at_time=datetime(2024, 8, 1),
        )
        assert r_outside is None  # No current belief matches
