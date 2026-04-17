"""Tests for contextuality: session/context-scoped beliefs (v0.4 #3)."""

from __future__ import annotations

from datetime import datetime

from patha.belief.contradiction import StubContradictionDetector
from patha.belief.layer import BeliefLayer, PlasticityConfig
from patha.belief.store import BeliefStore
from patha.belief.types import (
    ContradictionLabel,
    ContradictionResult,
)


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


# ─── Store-level context ───────────────────────────────────────────

class TestStoreContext:
    def test_context_stored(self) -> None:
        store = BeliefStore()
        b = store.add(
            proposition="I'm available",
            asserted_at=datetime(2024, 1, 1),
            asserted_in_session="s1",
            source_proposition_id="p1",
            context="work",
        )
        assert b.context == "work"

    def test_default_context_is_none(self) -> None:
        store = BeliefStore()
        b = store.add(
            proposition="x",
            asserted_at=datetime(2024, 1, 1),
            asserted_in_session="s1",
            source_proposition_id="p1",
        )
        assert b.context is None

    def test_by_context_filter(self) -> None:
        store = BeliefStore()
        store.add(
            proposition="I'm available",
            asserted_at=datetime(2024, 1, 1),
            asserted_in_session="s1",
            source_proposition_id="p1",
            context="work",
            belief_id="work-1",
        )
        store.add(
            proposition="I'm on holiday",
            asserted_at=datetime(2024, 1, 2),
            asserted_in_session="s2",
            source_proposition_id="p2",
            context="personal",
            belief_id="personal-1",
        )
        store.add(
            proposition="I love sushi",
            asserted_at=datetime(2024, 1, 3),
            asserted_in_session="s3",
            source_proposition_id="p3",
            belief_id="no-context",
        )
        assert {b.id for b in store.by_context("work")} == {"work-1"}
        assert {b.id for b in store.by_context("personal")} == {"personal-1"}
        assert {b.id for b in store.by_context(None)} == {"no-context"}


# ─── Ingest-time context filtering ─────────────────────────────────

class TestIngestContext:
    def test_different_contexts_dont_contradict(self) -> None:
        """'I'm available' in work context should not contradict
        'I'm on holiday' in personal context."""
        scripts = {
            ("I am available", "I am on holiday"): ContradictionResult(
                label=ContradictionLabel.CONTRADICTS, confidence=0.95
            ),
        }
        layer = BeliefLayer(
            store=BeliefStore(),
            detector=_ScriptedDetector(scripts),
            plasticity=PlasticityConfig(enabled=False),
            contradiction_threshold=0.5,
        )
        e1 = layer.ingest(
            proposition="I am available",
            asserted_at=datetime(2024, 1, 1),
            asserted_in_session="s1",
            source_proposition_id="p1",
            context="work",
        )
        e2 = layer.ingest(
            proposition="I am on holiday",
            asserted_at=datetime(2024, 1, 2),
            asserted_in_session="s2",
            source_proposition_id="p2",
            context="personal",
        )
        # Different contexts → no supersession
        assert e2.action == "added"
        assert layer.store.get(e1.new_belief.id).is_current  # type: ignore[union-attr]
        assert layer.store.get(e2.new_belief.id).is_current  # type: ignore[union-attr]

    def test_same_context_does_contradict(self) -> None:
        """Same context, contradicting propositions → supersession fires."""
        scripts = {
            ("I am available", "I am on holiday"): ContradictionResult(
                label=ContradictionLabel.CONTRADICTS, confidence=0.95
            ),
        }
        layer = BeliefLayer(
            store=BeliefStore(),
            detector=_ScriptedDetector(scripts),
            plasticity=PlasticityConfig(enabled=False),
            contradiction_threshold=0.5,
        )
        layer.ingest(
            proposition="I am available",
            asserted_at=datetime(2024, 1, 1),
            asserted_in_session="s1",
            source_proposition_id="p1",
            context="work",
        )
        e2 = layer.ingest(
            proposition="I am on holiday",
            asserted_at=datetime(2024, 1, 2),
            asserted_in_session="s2",
            source_proposition_id="p2",
            context="work",  # same context
        )
        # Same context → contradiction fires
        assert e2.action == "superseded"

    def test_context_independent_belief_still_contradicts(self) -> None:
        """A context-independent belief (context=None) participates in
        contradiction checks across every context."""
        scripts = {
            ("I am vegan", "I had steak today"): ContradictionResult(
                label=ContradictionLabel.CONTRADICTS, confidence=0.95
            ),
        }
        layer = BeliefLayer(
            store=BeliefStore(),
            detector=_ScriptedDetector(scripts),
            plasticity=PlasticityConfig(enabled=False),
            contradiction_threshold=0.5,
        )
        layer.ingest(
            proposition="I am vegan",
            asserted_at=datetime(2024, 1, 1),
            asserted_in_session="s1",
            source_proposition_id="p1",
            # no context = universal
        )
        e2 = layer.ingest(
            proposition="I had steak today",
            asserted_at=datetime(2024, 1, 2),
            asserted_in_session="s2",
            source_proposition_id="p2",
            context="food",
        )
        # Vegan is context-independent → still collides
        assert e2.action == "superseded"


# ─── Query-time context filtering ──────────────────────────────────

class TestQueryContext:
    def test_query_filters_by_context(self) -> None:
        layer = BeliefLayer(
            store=BeliefStore(),
            detector=StubContradictionDetector(),
            plasticity=PlasticityConfig(enabled=False),
        )
        e1 = layer.ingest(
            proposition="I'm available",
            asserted_at=datetime(2024, 1, 1),
            asserted_in_session="s1",
            source_proposition_id="p1",
            context="work",
        )
        e2 = layer.ingest(
            proposition="I'm on holiday",
            asserted_at=datetime(2024, 1, 2),
            asserted_in_session="s2",
            source_proposition_id="p2",
            context="personal",
        )
        ids = [e1.new_belief.id, e2.new_belief.id]

        work_result = layer.query(ids, context="work")
        assert len(work_result.current) == 1
        assert work_result.current[0].context == "work"

        personal_result = layer.query(ids, context="personal")
        assert len(personal_result.current) == 1
        assert personal_result.current[0].context == "personal"

    def test_query_without_context_returns_all(self) -> None:
        layer = BeliefLayer(
            store=BeliefStore(),
            detector=StubContradictionDetector(),
            plasticity=PlasticityConfig(enabled=False),
        )
        e1 = layer.ingest(
            proposition="a", asserted_at=datetime(2024, 1, 1),
            asserted_in_session="s1", source_proposition_id="p1",
            context="work",
        )
        e2 = layer.ingest(
            proposition="b", asserted_at=datetime(2024, 1, 2),
            asserted_in_session="s2", source_proposition_id="p2",
            context="personal",
        )
        result = layer.query([e1.new_belief.id, e2.new_belief.id])
        # No context filter → both surface
        assert len(result.current) == 2

    def test_universal_belief_surfaces_in_every_context(self) -> None:
        layer = BeliefLayer(
            store=BeliefStore(),
            detector=StubContradictionDetector(),
            plasticity=PlasticityConfig(enabled=False),
        )
        e_univ = layer.ingest(
            proposition="I was born in Sofia",
            asserted_at=datetime(2024, 1, 1),
            asserted_in_session="s1",
            source_proposition_id="p1",
            # universal, no context
        )
        e_work = layer.ingest(
            proposition="I'm available for meetings",
            asserted_at=datetime(2024, 1, 2),
            asserted_in_session="s2",
            source_proposition_id="p2",
            context="work",
        )
        # Query in work context: universal belief + work-tagged belief
        result = layer.query(
            [e_univ.new_belief.id, e_work.new_belief.id],
            context="work",
        )
        assert len(result.current) == 2
