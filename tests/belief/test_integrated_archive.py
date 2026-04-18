"""Tests for raw-archive integration with IntegratedPatha (v0.5)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from patha.belief.contradiction import StubContradictionDetector
from patha.belief.layer import BeliefLayer, PlasticityConfig
from patha.belief.raw_archive import RawArchive
from patha.belief.store import BeliefStore
from patha.integrated import IntegratedPatha


def _make_integrated(with_archive: bool = True) -> IntegratedPatha:
    layer = BeliefLayer(
        store=BeliefStore(),
        detector=StubContradictionDetector(),
        plasticity=PlasticityConfig(enabled=False),
    )
    archive = RawArchive() if with_archive else None
    return IntegratedPatha(
        belief_layer=layer,
        raw_archive=archive,
    )


class TestArchiveIntegration:
    def test_ingest_records_raw_turn(self) -> None:
        p = _make_integrated()
        p.ingest(
            proposition="I live in Sofia",
            asserted_at=datetime(2024, 1, 1),
            asserted_in_session="s1",
            source_proposition_id="prop-1",
        )
        assert len(p.raw_archive) == 1
        turn = p.raw_archive.turn_for_proposition("prop-1")
        assert turn is not None
        assert turn.content == "I live in Sofia"
        assert turn.session_id == "s1"

    def test_turn_index_increments_per_session(self) -> None:
        p = _make_integrated()
        p.ingest(
            proposition="first thing",
            asserted_at=datetime(2024, 1, 1),
            asserted_in_session="s1",
            source_proposition_id="p1",
        )
        p.ingest(
            proposition="second thing",
            asserted_at=datetime(2024, 1, 2),
            asserted_in_session="s1",
            source_proposition_id="p2",
        )
        turns = p.raw_archive.turns_by_session("s1")
        assert len(turns) == 2
        assert turns[0].turn_index == 0
        assert turns[1].turn_index == 1

    def test_separate_sessions_have_independent_counters(self) -> None:
        p = _make_integrated()
        p.ingest(
            proposition="a", asserted_at=datetime(2024, 1, 1),
            asserted_in_session="s1", source_proposition_id="p1",
        )
        p.ingest(
            proposition="b", asserted_at=datetime(2024, 1, 1),
            asserted_in_session="s2", source_proposition_id="p2",
        )
        s1_turns = p.raw_archive.turns_by_session("s1")
        s2_turns = p.raw_archive.turns_by_session("s2")
        assert s1_turns[0].turn_index == 0
        assert s2_turns[0].turn_index == 0

    def test_raw_content_different_from_proposition(self) -> None:
        """When raw_content is passed, it's stored distinct from the
        distilled proposition (long utterance → atomic claim)."""
        p = _make_integrated()
        p.ingest(
            proposition="I like sushi",  # atomic claim
            asserted_at=datetime(2024, 1, 1),
            asserted_in_session="s1",
            source_proposition_id="p1",
            raw_content="Honestly, after trying it last week, I think I like sushi a lot.",
        )
        turn = p.raw_archive.turn_for_proposition("p1")
        assert turn is not None
        assert "after trying it last week" in turn.content

    def test_source_name_recorded(self) -> None:
        p = _make_integrated()
        p.ingest(
            proposition="I had a meeting with Alex",
            asserted_at=datetime(2024, 1, 1),
            asserted_in_session="s1",
            source_proposition_id="p1",
            source_name="slack-dm",
        )
        turn = p.raw_archive.turn_for_proposition("p1")
        assert turn is not None
        assert turn.source_name == "slack-dm"

    def test_no_archive_still_works(self) -> None:
        """Backwards compat: omit raw_archive, system still ingests."""
        p = _make_integrated(with_archive=False)
        ev = p.ingest(
            proposition="x",
            asserted_at=datetime(2024, 1, 1),
            asserted_in_session="s1",
            source_proposition_id="p1",
        )
        assert ev.new_belief.proposition == "x"

    def test_proposition_traceable_from_belief_to_raw(self) -> None:
        """End-to-end provenance: a belief can be traced back to its
        original turn via source_proposition_id → raw archive."""
        p = _make_integrated()
        ev = p.ingest(
            proposition="I moved to Sofia",
            asserted_at=datetime(2024, 1, 1),
            asserted_in_session="s1",
            source_proposition_id="prop-1",
            raw_content="Just to let you know, I moved to Sofia last month.",
            source_name="email",
        )
        # Belief has source_proposition_id
        assert ev.new_belief.source_proposition_id == "prop-1"
        # Archive resolves that id to the raw turn
        turn = p.raw_archive.turn_for_proposition(
            ev.new_belief.source_proposition_id
        )
        assert turn is not None
        assert "Just to let you know" in turn.content
        assert turn.source_name == "email"
