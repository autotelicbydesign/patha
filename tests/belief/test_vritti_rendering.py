"""Tests for vṛtti-aware rendering (v0.5)."""

from __future__ import annotations

from datetime import datetime

import pytest

from patha.belief.contradiction import StubContradictionDetector
from patha.belief.layer import BeliefLayer, PlasticityConfig
from patha.belief.store import BeliefStore
from patha.belief.types import (
    ContradictionLabel,
    ContradictionResult,
    ResolutionStatus,
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


class TestVrittiRendering:
    def test_default_no_vritti_tag(self) -> None:
        layer = BeliefLayer(
            store=BeliefStore(),
            detector=StubContradictionDetector(),
            plasticity=PlasticityConfig(enabled=False),
        )
        ev = layer.ingest(
            proposition="I live in Sofia",
            asserted_at=datetime(2024, 1, 1),
            asserted_in_session="s1",
            source_proposition_id="p1",
        )
        result = layer.query([ev.new_belief.id])
        rendered = layer.render_summary(result)
        assert "[pramana]" not in rendered
        assert "I live in Sofia" in rendered

    def test_vritti_tag_surfaces_pramana_for_current(self) -> None:
        layer = BeliefLayer(
            store=BeliefStore(),
            detector=StubContradictionDetector(),
            plasticity=PlasticityConfig(enabled=False),
        )
        ev = layer.ingest(
            proposition="I live in Sofia",
            asserted_at=datetime(2024, 1, 1),
            asserted_in_session="s1",
            source_proposition_id="p1",
        )
        result = layer.query([ev.new_belief.id])
        rendered = layer.render_summary(result, include_vritti=True)
        assert "[pramana]" in rendered

    def test_vritti_tag_for_superseded_history(self) -> None:
        """Superseded belief surfaced as history should tag as smṛti."""
        scripts = {
            ("I live in Sydney", "I moved to Sofia"): ContradictionResult(
                label=ContradictionLabel.CONTRADICTS, confidence=0.95
            ),
        }
        layer = BeliefLayer(
            store=BeliefStore(),
            detector=_ScriptedDetector(scripts),
            plasticity=PlasticityConfig(enabled=False),
        )
        e1 = layer.ingest(
            proposition="I live in Sydney",
            asserted_at=datetime(2023, 1, 1),
            asserted_in_session="s1",
            source_proposition_id="p1",
        )
        e2 = layer.ingest(
            proposition="I moved to Sofia",
            asserted_at=datetime(2024, 1, 1),
            asserted_in_session="s2",
            source_proposition_id="p2",
        )
        result = layer.query(
            [e1.new_belief.id, e2.new_belief.id],
            include_history=True,
        )
        rendered = layer.render_summary(
            result, include_history=True, include_vritti=True
        )
        # Current belief tagged pramana
        assert "[pramana]" in rendered
        # History belief tagged smrti
        assert "[smrti]" in rendered

    def test_low_confidence_tagged_vikalpa(self) -> None:
        """A low-confidence current belief renders as vikalpa, not pramana."""
        layer = BeliefLayer(
            store=BeliefStore(),
            detector=StubContradictionDetector(),
            plasticity=PlasticityConfig(enabled=False),
        )
        ev = layer.ingest(
            proposition="I think the meeting is on Thursday",
            asserted_at=datetime(2024, 1, 1),
            asserted_in_session="s1",
            source_proposition_id="p1",
            confidence=0.3,  # low
        )
        result = layer.query([ev.new_belief.id])
        rendered = layer.render_summary(result, include_vritti=True)
        assert "[vikalpa]" in rendered
