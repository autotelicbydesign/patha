"""Tests for pramāṇa-hierarchy-aware contradiction resolution and
source-reliability tracking (the 'what if doctor is wrong' layer).
"""

from __future__ import annotations

from datetime import datetime

import pytest

from patha.belief.contradiction import StubContradictionDetector
from patha.belief.layer import BeliefLayer, PlasticityConfig
from patha.belief.store import BeliefStore, SourceReliability
from patha.belief.types import (
    PRAMANA_STRENGTH,
    ContradictionLabel,
    ContradictionResult,
    Pramana,
    ResolutionStatus,
)


# ─── Default confidence by pramāṇa ──────────────────────────────────

class TestDefaultConfidenceByPramana:
    def test_pratyaksa_defaults_to_one(self) -> None:
        store = BeliefStore()
        b = store.add(
            proposition="I saw it",
            asserted_at=datetime(2024, 1, 1),
            asserted_in_session="s1",
            source_proposition_id="p1",
            pramana=Pramana.PRATYAKSA,
        )
        assert b.confidence == 1.0

    def test_shabda_defaults_to_six(self) -> None:
        store = BeliefStore()
        b = store.add(
            proposition="my doctor told me",
            asserted_at=datetime(2024, 1, 1),
            asserted_in_session="s1",
            source_proposition_id="p1",
            pramana=Pramana.SHABDA,
        )
        # Base strength 0.6, no source history → reliability=1.0 → 0.6
        assert b.confidence == pytest.approx(0.6)

    def test_anumana_defaults_to_eight(self) -> None:
        store = BeliefStore()
        b = store.add(
            proposition="i think it works",
            asserted_at=datetime(2024, 1, 1),
            asserted_in_session="s1",
            source_proposition_id="p1",
            pramana=Pramana.ANUMANA,
        )
        assert b.confidence == pytest.approx(0.8)

    def test_unknown_defaults_to_one(self) -> None:
        """Bare self-assertions without epistemic markers → full confidence.
        Preserves v0.1/v0.2 behaviour for untagged beliefs.
        """
        store = BeliefStore()
        b = store.add(
            proposition="the coffee is hot",
            asserted_at=datetime(2024, 1, 1),
            asserted_in_session="s1",
            source_proposition_id="p1",
        )
        assert b.confidence == 1.0

    def test_explicit_confidence_overrides(self) -> None:
        store = BeliefStore()
        b = store.add(
            proposition="x",
            asserted_at=datetime(2024, 1, 1),
            asserted_in_session="s1",
            source_proposition_id="p1",
            pramana=Pramana.SHABDA,
            confidence=0.95,  # explicit override
        )
        assert b.confidence == 0.95


# ─── Source reliability ─────────────────────────────────────────────

class TestSourceReliability:
    def test_default_reliability_is_one(self) -> None:
        store = BeliefStore()
        rel = store.source_reliability("unknown-source")
        assert rel.score == 1.0

    def test_reliability_drops_on_sublation(self) -> None:
        store = BeliefStore()
        # An unreliable source makes a testimony claim
        store.add(
            proposition="the result is X",
            asserted_at=datetime(2024, 1, 1),
            asserted_in_session="doctor-wrong",
            source_proposition_id="p1",
            belief_id="weak",
            pramana=Pramana.SHABDA,
        )
        # User later observes not-X directly
        store.add(
            proposition="I observed not-X",
            asserted_at=datetime(2024, 2, 1),
            asserted_in_session="user-self",
            source_proposition_id="p2",
            belief_id="strong",
            pramana=Pramana.PRATYAKSA,
        )
        store.sublate("weak", "strong")

        rel = store.source_reliability("doctor-wrong")
        assert rel.sublated == 1
        assert rel.score < 1.0

    def test_reliability_floor_is_point_three(self) -> None:
        store = BeliefStore()
        # Simulate a source repeatedly sublated
        for i in range(20):
            store.add(
                proposition=f"claim {i}",
                asserted_at=datetime(2024, 1, i + 1),
                asserted_in_session="flaky-source",
                source_proposition_id=f"p{i}",
                belief_id=f"weak-{i}",
                pramana=Pramana.SHABDA,
            )
            store.add(
                proposition=f"observed not-claim {i}",
                asserted_at=datetime(2024, 2, i + 1),
                asserted_in_session="user",
                source_proposition_id=f"obs-{i}",
                belief_id=f"strong-{i}",
                pramana=Pramana.PRATYAKSA,
            )
            store.sublate(f"weak-{i}", f"strong-{i}")

        rel = store.source_reliability("flaky-source")
        assert rel.score == 0.3  # floor

    def test_shabda_confidence_discounted_by_source_reliability(self) -> None:
        """A new SHABDA claim from a previously-sublated source starts lower."""
        store = BeliefStore()
        # Build up bad reputation for 'dodgy-source'
        for i in range(3):
            store.add(
                proposition=f"claim {i}",
                asserted_at=datetime(2024, 1, i + 1),
                asserted_in_session="dodgy-source",
                source_proposition_id=f"p{i}",
                belief_id=f"w{i}",
                pramana=Pramana.SHABDA,
            )
            store.add(
                proposition=f"observed not-claim {i}",
                asserted_at=datetime(2024, 2, i + 1),
                asserted_in_session="user",
                source_proposition_id=f"obs{i}",
                belief_id=f"s{i}",
                pramana=Pramana.PRATYAKSA,
            )
            store.sublate(f"w{i}", f"s{i}")

        # New SHABDA claim from the dodgy source
        new_b = store.add(
            proposition="I heard X",
            asserted_at=datetime(2024, 3, 1),
            asserted_in_session="dodgy-source",
            source_proposition_id="p-new",
            pramana=Pramana.SHABDA,
        )
        # Base strength 0.6 × reliability (< 1.0) → confidence < 0.6
        assert new_b.confidence < 0.6


# ─── Sublation (BADHITA status) ─────────────────────────────────────

class TestSublation:
    def test_sublate_sets_badhita(self) -> None:
        store = BeliefStore()
        store.add(
            proposition="doctor said X",
            asserted_at=datetime(2024, 1, 1),
            asserted_in_session="doc",
            source_proposition_id="p1",
            belief_id="weak",
            pramana=Pramana.SHABDA,
        )
        store.add(
            proposition="I saw not-X",
            asserted_at=datetime(2024, 2, 1),
            asserted_in_session="me",
            source_proposition_id="p2",
            belief_id="strong",
            pramana=Pramana.PRATYAKSA,
        )
        store.sublate("weak", "strong")

        assert store.get("weak").status == ResolutionStatus.BADHITA  # type: ignore[union-attr]
        assert store.get("strong").is_current  # type: ignore[union-attr]

    def test_badhita_beliefs_not_current(self) -> None:
        store = BeliefStore()
        store.add(
            proposition="w",
            asserted_at=datetime(2024, 1, 1),
            asserted_in_session="s1",
            source_proposition_id="p1",
            belief_id="w",
            pramana=Pramana.SHABDA,
        )
        store.add(
            proposition="s",
            asserted_at=datetime(2024, 2, 1),
            asserted_in_session="s2",
            source_proposition_id="p2",
            belief_id="s",
            pramana=Pramana.PRATYAKSA,
        )
        store.sublate("w", "s")
        assert store.get("w").is_superseded  # type: ignore[union-attr]


# ─── resolve_contradiction: pramāṇa hierarchy wins ──────────────────

class TestResolveContradiction:
    def test_stronger_pramana_sublates_weaker(self) -> None:
        """PRATYAKṢA contradicts earlier SHABDA → SHABDA becomes BADHITA."""
        store = BeliefStore()
        store.add(
            proposition="doc said x",
            asserted_at=datetime(2024, 1, 1),
            asserted_in_session="doc",
            source_proposition_id="p1",
            belief_id="shabda",
            pramana=Pramana.SHABDA,
        )
        store.add(
            proposition="I saw not-x",
            asserted_at=datetime(2024, 2, 1),
            asserted_in_session="me",
            source_proposition_id="p2",
            belief_id="prat",
            pramana=Pramana.PRATYAKSA,
        )
        status = store.resolve_contradiction("shabda", "prat")
        assert status == ResolutionStatus.BADHITA
        assert store.get("shabda").status == ResolutionStatus.BADHITA  # type: ignore[union-attr]
        assert store.get("prat").is_current  # type: ignore[union-attr]

    def test_weaker_pramana_cannot_overthrow_stronger(self) -> None:
        """Later SHABDA cannot overthrow earlier PRATYAKṢA — instead,
        the SHABDA claim is itself sublated."""
        store = BeliefStore()
        store.add(
            proposition="I saw x",
            asserted_at=datetime(2024, 1, 1),
            asserted_in_session="me",
            source_proposition_id="p1",
            belief_id="prat",
            pramana=Pramana.PRATYAKSA,
        )
        store.add(
            proposition="someone said not-x",
            asserted_at=datetime(2024, 2, 1),
            asserted_in_session="gossip",
            source_proposition_id="p2",
            belief_id="shabda",
            pramana=Pramana.SHABDA,
        )
        status = store.resolve_contradiction("prat", "shabda")
        assert status == ResolutionStatus.BADHITA
        # The new SHABDA is the one sublated (cannot overthrow perception)
        assert store.get("shabda").status == ResolutionStatus.BADHITA  # type: ignore[union-attr]
        assert store.get("prat").is_current  # type: ignore[union-attr]

    def test_equal_pramana_temporal_supersession(self) -> None:
        """Two PRATYAKṢA claims contradicting → temporal supersession."""
        store = BeliefStore()
        store.add(
            proposition="I see x",
            asserted_at=datetime(2024, 1, 1),
            asserted_in_session="s1",
            source_proposition_id="p1",
            belief_id="old",
            pramana=Pramana.PRATYAKSA,
        )
        store.add(
            proposition="I see not-x",
            asserted_at=datetime(2024, 2, 1),
            asserted_in_session="s2",
            source_proposition_id="p2",
            belief_id="new",
            pramana=Pramana.PRATYAKSA,
        )
        status = store.resolve_contradiction("old", "new")
        assert status == ResolutionStatus.SUPERSEDED
        assert store.get("old").status == ResolutionStatus.SUPERSEDED  # type: ignore[union-attr]


# ─── Layer-level integration ────────────────────────────────────────

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


class TestLayerIntegration:
    def test_what_if_doctor_is_wrong(self) -> None:
        """The flagship scenario: user's SHABDA-belief from a doctor is
        sublated by their own PRATYAKṢA observation."""
        scripts = {
            (
                "my doctor told me I have diabetes",
                "I checked my own blood sugar and it is normal",
            ): ContradictionResult(
                label=ContradictionLabel.CONTRADICTS, confidence=0.95
            ),
        }
        layer = BeliefLayer(
            store=BeliefStore(),
            detector=_ScriptedDetector(scripts),
            plasticity=PlasticityConfig(enabled=False),
        )

        # Explicitly pass pramanas for deterministic testing
        doctor_claim = layer.ingest(
            proposition="my doctor told me I have diabetes",
            asserted_at=datetime(2024, 1, 1),
            asserted_in_session="doctor-visit",
            source_proposition_id="p1",
            pramana=Pramana.SHABDA,
        )
        user_obs = layer.ingest(
            proposition="I checked my own blood sugar and it is normal",
            asserted_at=datetime(2024, 2, 1),
            asserted_in_session="home",
            source_proposition_id="p2",
            pramana=Pramana.PRATYAKSA,
        )

        # Doctor's claim should be BADHITA, not merely SUPERSEDED
        doctor_belief = layer.store.get(doctor_claim.new_belief.id)
        user_belief = layer.store.get(user_obs.new_belief.id)
        assert doctor_belief is not None and user_belief is not None
        assert doctor_belief.status == ResolutionStatus.BADHITA
        assert user_belief.is_current

        # Source reliability for the doctor should have dropped
        rel = layer.store.source_reliability("doctor-visit")
        assert rel.sublated == 1
        assert rel.score < 1.0

    def test_auto_detect_drives_sublation(self) -> None:
        """Full-stack: auto-detected pramāṇa drives the hierarchy.

        'The doctor told me X' auto-detects SHABDA;
        'I saw not-X' auto-detects PRATYAKṢA;
        contradiction → BADHITA on the SHABDA.
        """
        scripts = {
            (
                "my doctor told me I have diabetes",
                "I saw my blood sugar reading was normal",
            ): ContradictionResult(
                label=ContradictionLabel.CONTRADICTS, confidence=0.95
            ),
        }
        layer = BeliefLayer(
            store=BeliefStore(),
            detector=_ScriptedDetector(scripts),
            plasticity=PlasticityConfig(enabled=False),
        )
        e1 = layer.ingest(
            proposition="my doctor told me I have diabetes",
            asserted_at=datetime(2024, 1, 1),
            asserted_in_session="visit",
            source_proposition_id="p1",
        )
        e2 = layer.ingest(
            proposition="I saw my blood sugar reading was normal",
            asserted_at=datetime(2024, 2, 1),
            asserted_in_session="home",
            source_proposition_id="p2",
        )
        # Pramana should be auto-detected
        assert e1.new_belief.pramana == Pramana.SHABDA
        assert e2.new_belief.pramana == Pramana.PRATYAKSA
        # Sublation happened — doctor's claim is BADHITA
        assert layer.store.get(e1.new_belief.id).status == ResolutionStatus.BADHITA  # type: ignore[union-attr]
