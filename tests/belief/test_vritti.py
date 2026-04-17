"""Tests for vṛtti (cognitive-mode) classification."""

from __future__ import annotations

from datetime import datetime

import pytest

from patha.belief.store import BeliefStore
from patha.belief.types import Pramana, ResolutionStatus
from patha.belief.vritti import VrittiClass, vritti_label, vritti_of


def _mk(status: ResolutionStatus, confidence: float = 1.0):
    store = BeliefStore()
    b = store.add(
        proposition="x",
        asserted_at=datetime(2024, 1, 1),
        asserted_in_session="s1",
        source_proposition_id="p1",
        confidence=confidence,
    )
    b.status = status
    return b


class TestVrittiOf:
    def test_current_high_conf_is_pramana(self) -> None:
        b = _mk(ResolutionStatus.CURRENT, confidence=0.9)
        assert vritti_of(b) == VrittiClass.PRAMANA

    def test_current_low_conf_is_vikalpa(self) -> None:
        """Low-confidence current beliefs are asserted-but-unverified."""
        b = _mk(ResolutionStatus.CURRENT, confidence=0.4)
        assert vritti_of(b) == VrittiClass.VIKALPA

    def test_coexists_is_pramana(self) -> None:
        b = _mk(ResolutionStatus.COEXISTS, confidence=0.9)
        assert vritti_of(b) == VrittiClass.PRAMANA

    def test_superseded_as_history_is_smrti(self) -> None:
        """Surfaced as part of supersession lineage → recollection."""
        b = _mk(ResolutionStatus.SUPERSEDED)
        assert vritti_of(b, surfaced_as_history=True) == VrittiClass.SMRTI

    def test_superseded_not_as_history_is_viparyaya(self) -> None:
        """Surfaced outside history → erroneous cognition."""
        b = _mk(ResolutionStatus.SUPERSEDED)
        assert vritti_of(b, surfaced_as_history=False) == VrittiClass.VIPARYAYA

    def test_badhita_as_history_is_smrti(self) -> None:
        b = _mk(ResolutionStatus.BADHITA)
        assert vritti_of(b, surfaced_as_history=True) == VrittiClass.SMRTI

    def test_badhita_not_as_history_is_viparyaya(self) -> None:
        b = _mk(ResolutionStatus.BADHITA)
        assert vritti_of(b, surfaced_as_history=False) == VrittiClass.VIPARYAYA

    def test_disputed_is_viparyaya(self) -> None:
        b = _mk(ResolutionStatus.DISPUTED)
        assert vritti_of(b) == VrittiClass.VIPARYAYA

    def test_ambiguous_is_vikalpa(self) -> None:
        b = _mk(ResolutionStatus.AMBIGUOUS)
        assert vritti_of(b) == VrittiClass.VIKALPA

    def test_archived_is_nidra(self) -> None:
        b = _mk(ResolutionStatus.ARCHIVED)
        assert vritti_of(b) == VrittiClass.NIDRA


class TestVrittiLabel:
    def test_all_classes_have_labels(self) -> None:
        for c in VrittiClass:
            label = vritti_label(c)
            assert isinstance(label, str)
            assert len(label) > 0

    def test_pramana_label(self) -> None:
        assert "valid" in vritti_label(VrittiClass.PRAMANA).lower()

    def test_viparyaya_label(self) -> None:
        assert "erroneous" in vritti_label(VrittiClass.VIPARYAYA).lower()
