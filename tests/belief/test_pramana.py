"""Tests for pramāṇa detection and pramāṇa-diversity reinforcement."""

from __future__ import annotations

from datetime import datetime

import pytest

from patha.belief.contradiction import StubContradictionDetector
from patha.belief.layer import BeliefLayer, PlasticityConfig
from patha.belief.pramana import detect_pramana, list_patterns
from patha.belief.store import BeliefStore
from patha.belief.types import Pramana


# ─── detect_pramana ────────────────────────────────────────────────

class TestDetectPramana:
    @pytest.mark.parametrize("text", [
        "My doctor told me I have high cholesterol",
        "According to the BBC, it will rain tomorrow",
        "I was told the meeting moved to 3pm",
        "I heard from a colleague that the launch is delayed",
    ])
    def test_shabda_patterns(self, text: str) -> None:
        assert detect_pramana(text).pramana == Pramana.SHABDA

    @pytest.mark.parametrize("text", [
        "I saw her leave the office around 6",
        "We watched the sunset from the balcony",
        "I noticed a new sign on the shop window",
        "I felt a draft in the living room",
    ])
    def test_pratyaksa_patterns(self, text: str) -> None:
        assert detect_pramana(text).pramana == Pramana.PRATYAKSA

    @pytest.mark.parametrize("text", [
        "I don't see any reason to rush",
        "I didn't hear the doorbell",
    ])
    def test_anupalabdhi_patterns(self, text: str) -> None:
        assert detect_pramana(text).pramana == Pramana.ANUPALABDHI

    @pytest.mark.parametrize("text", [
        "She must have left before we arrived",
        "He must be in Sydney by now",
    ])
    def test_arthapatti_patterns(self, text: str) -> None:
        assert detect_pramana(text).pramana == Pramana.ARTHAPATTI

    @pytest.mark.parametrize("text", [
        "I think the code works",
        "I believe the deployment succeeded",
        "It seems like the server is down",
        "Probably a good idea to retry",
    ])
    def test_anumana_patterns(self, text: str) -> None:
        assert detect_pramana(text).pramana == Pramana.ANUMANA

    @pytest.mark.parametrize("text", [
        "This situation is like last year's outage",
        "The new UI is similar to Notion's",
    ])
    def test_upamana_patterns(self, text: str) -> None:
        assert detect_pramana(text).pramana == Pramana.UPAMANA

    @pytest.mark.parametrize("text", [
        "The coffee is hot",
        "Blueberries are my favourite",
        "The meeting is at 3pm",
    ])
    def test_unknown_when_no_cue(self, text: str) -> None:
        # Bare assertions with no epistemic markers
        assert detect_pramana(text).pramana == Pramana.UNKNOWN

    def test_cue_populated_on_hit(self) -> None:
        r = detect_pramana("I saw her leave")
        assert r.pramana == Pramana.PRATYAKSA
        assert "saw" in r.cue.lower()

    def test_cue_empty_on_unknown(self) -> None:
        r = detect_pramana("The sky is blue")
        assert r.cue == ""

    def test_list_patterns_shape(self) -> None:
        pats = list_patterns()
        assert "shabda" in pats
        assert "pratyaksa" in pats
        assert pats["shabda"] >= 1


# ─── Pramāṇa in BeliefStore ─────────────────────────────────────────

class TestStorePramana:
    def test_default_is_unknown(self) -> None:
        store = BeliefStore()
        b = store.add(
            proposition="x",
            asserted_at=datetime(2024, 1, 1),
            asserted_in_session="s1",
            source_proposition_id="p1",
        )
        assert b.pramana == Pramana.UNKNOWN

    def test_explicit_pramana_stored(self) -> None:
        store = BeliefStore()
        b = store.add(
            proposition="I saw her leave",
            asserted_at=datetime(2024, 1, 1),
            asserted_in_session="s1",
            source_proposition_id="p1",
            pramana=Pramana.PRATYAKSA,
        )
        assert b.pramana == Pramana.PRATYAKSA

    def test_pramana_roundtrips_via_persistence(self, tmp_path) -> None:
        path = tmp_path / "beliefs.jsonl"

        s1 = BeliefStore(persistence_path=path)
        s1.add(
            proposition="The doctor told me",
            asserted_at=datetime(2024, 1, 1),
            asserted_in_session="s1",
            source_proposition_id="p1",
            belief_id="b1",
            pramana=Pramana.SHABDA,
        )

        s2 = BeliefStore(persistence_path=path)
        b = s2.get("b1")
        assert b is not None
        assert b.pramana == Pramana.SHABDA


# ─── Pramāṇa-diversity reinforcement ────────────────────────────────

class TestPramanaDiversityReinforce:
    def test_cross_pramana_bigger_bump(self) -> None:
        """Different pramāṇa + different source → 40% of gap closed."""
        store = BeliefStore()
        a = store.add(
            proposition="I have high cholesterol",
            asserted_at=datetime(2024, 1, 1),
            asserted_in_session="s1",
            source_proposition_id="p1",
            belief_id="a",
            pramana=Pramana.SHABDA,  # told by doctor
        )
        store.add(
            proposition="My blood test shows high cholesterol",
            asserted_at=datetime(2024, 2, 1),
            asserted_in_session="s2",  # different session
            source_proposition_id="p2",
            belief_id="b",
            pramana=Pramana.PRATYAKSA,  # I saw the result
        )
        a.confidence = 0.5
        store.reinforce("a", "b")
        # Distinct source AND distinct pramana → 40% bump of gap
        # gap = 0.5, new = 0.5 + 0.4 * 0.5 = 0.7
        assert a.confidence == pytest.approx(0.7)

    def test_same_pramana_distinct_source(self) -> None:
        """Distinct source, same pramāṇa → 30%."""
        store = BeliefStore()
        a = store.add(
            proposition="I saw her leave",
            asserted_at=datetime(2024, 1, 1),
            asserted_in_session="s1",
            source_proposition_id="p1",
            belief_id="a",
            pramana=Pramana.PRATYAKSA,
        )
        store.add(
            proposition="I saw her leave earlier",
            asserted_at=datetime(2024, 1, 2),
            asserted_in_session="s2",
            source_proposition_id="p2",
            belief_id="b",
            pramana=Pramana.PRATYAKSA,
        )
        a.confidence = 0.5
        store.reinforce("a", "b")
        # 30% of 0.5 = 0.15; new = 0.65
        assert a.confidence == pytest.approx(0.65)

    def test_distinct_pramana_same_source(self) -> None:
        """Same source, distinct pramāṇa → 20%."""
        store = BeliefStore()
        a = store.add(
            proposition="I think the code works",
            asserted_at=datetime(2024, 1, 1),
            asserted_in_session="s1",
            source_proposition_id="p1",
            belief_id="a",
            pramana=Pramana.ANUMANA,
        )
        store.add(
            proposition="I saw the tests pass",
            asserted_at=datetime(2024, 1, 1, 12, 5),
            asserted_in_session="s1",  # same session
            source_proposition_id="p2",
            belief_id="b",
            pramana=Pramana.PRATYAKSA,  # different pramana
        )
        a.confidence = 0.5
        store.reinforce("a", "b")
        # 20% of 0.5 = 0.1; new = 0.6
        assert a.confidence == pytest.approx(0.6)

    def test_both_same_smallest_bump(self) -> None:
        """Same source + same pramāṇa → 10% (pure echo)."""
        store = BeliefStore()
        a = store.add(
            proposition="I saw her",
            asserted_at=datetime(2024, 1, 1),
            asserted_in_session="s1",
            source_proposition_id="p1",
            belief_id="a",
            pramana=Pramana.PRATYAKSA,
        )
        store.add(
            proposition="I saw her again",
            asserted_at=datetime(2024, 1, 1, 12, 5),
            asserted_in_session="s1",
            source_proposition_id="p2",
            belief_id="b",
            pramana=Pramana.PRATYAKSA,
        )
        a.confidence = 0.5
        store.reinforce("a", "b")
        # 10% of 0.5 = 0.05; new = 0.55
        assert a.confidence == pytest.approx(0.55)

    def test_unknown_pramana_treated_as_non_distinct(self) -> None:
        """A reinforcer with UNKNOWN pramana doesn't claim diversity."""
        store = BeliefStore()
        a = store.add(
            proposition="x",
            asserted_at=datetime(2024, 1, 1),
            asserted_in_session="s1",
            source_proposition_id="p1",
            belief_id="a",
            pramana=Pramana.PRATYAKSA,
        )
        store.add(
            proposition="y",
            asserted_at=datetime(2024, 2, 1),
            asserted_in_session="s2",
            source_proposition_id="p2",
            belief_id="b",
            pramana=Pramana.UNKNOWN,  # unknown — no diversity claim
        )
        a.confidence = 0.5
        store.reinforce("a", "b")
        # Distinct source, UNKNOWN pramana → 30% not 40%
        assert a.confidence == pytest.approx(0.65)


# ─── BeliefLayer auto-detection integration ─────────────────────────

class TestLayerAutoDetect:
    def test_auto_detect_pratyaksa(self) -> None:
        layer = BeliefLayer(
            store=BeliefStore(),
            detector=StubContradictionDetector(),
            plasticity=PlasticityConfig(enabled=False),  # isolate
        )
        ev = layer.ingest(
            proposition="I saw the deploy succeed",
            asserted_at=datetime(2024, 1, 1),
            asserted_in_session="s1",
            source_proposition_id="p1",
        )
        assert ev.new_belief.pramana == Pramana.PRATYAKSA

    def test_auto_detect_shabda(self) -> None:
        layer = BeliefLayer(
            store=BeliefStore(),
            detector=StubContradictionDetector(),
            plasticity=PlasticityConfig(enabled=False),
        )
        ev = layer.ingest(
            proposition="The doctor told me I have diabetes",
            asserted_at=datetime(2024, 1, 1),
            asserted_in_session="s1",
            source_proposition_id="p1",
        )
        assert ev.new_belief.pramana == Pramana.SHABDA

    def test_explicit_pramana_overrides_autodetect(self) -> None:
        layer = BeliefLayer(
            store=BeliefStore(),
            detector=StubContradictionDetector(),
            plasticity=PlasticityConfig(enabled=False),
        )
        ev = layer.ingest(
            proposition="I saw the deploy succeed",  # would auto = PRATYAKSA
            asserted_at=datetime(2024, 1, 1),
            asserted_in_session="s1",
            source_proposition_id="p1",
            pramana=Pramana.SHABDA,  # explicit override
        )
        assert ev.new_belief.pramana == Pramana.SHABDA

    def test_unknown_when_no_cue(self) -> None:
        layer = BeliefLayer(
            store=BeliefStore(),
            detector=StubContradictionDetector(),
            plasticity=PlasticityConfig(enabled=False),
        )
        ev = layer.ingest(
            proposition="The coffee is hot",
            asserted_at=datetime(2024, 1, 1),
            asserted_in_session="s1",
            source_proposition_id="p1",
        )
        assert ev.new_belief.pramana == Pramana.UNKNOWN
