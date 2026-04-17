"""Tests for the belief store."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from patha.belief.store import BeliefStore
from patha.belief.types import ResolutionStatus, Validity


# ─── Fixtures and helpers ────────────────────────────────────────────

@pytest.fixture
def store() -> BeliefStore:
    return BeliefStore()


def _add(store: BeliefStore, bid: str, prop: str, session: str = "s1"):
    return store.add(
        proposition=prop,
        asserted_at=datetime(2024, 1, 1, 12, 0),
        asserted_in_session=session,
        source_proposition_id=f"prop-{bid}",
        belief_id=bid,
    )


# ─── add() ───────────────────────────────────────────────────────────

class TestAdd:
    def test_basic_add(self, store: BeliefStore) -> None:
        b = _add(store, "b1", "I love sushi")
        assert b.id == "b1"
        assert b.proposition == "I love sushi"
        assert b.is_current

    def test_auto_id_when_none(self, store: BeliefStore) -> None:
        b = store.add(
            proposition="I love sushi",
            asserted_at=datetime(2024, 1, 1),
            asserted_in_session="s1",
            source_proposition_id="p1",
        )
        assert b.id  # non-empty UUID

    def test_duplicate_id_raises(self, store: BeliefStore) -> None:
        _add(store, "b1", "I love sushi")
        with pytest.raises(ValueError, match="already exists"):
            _add(store, "b1", "something else")

    def test_length_and_membership(self, store: BeliefStore) -> None:
        _add(store, "b1", "x")
        _add(store, "b2", "y")
        assert len(store) == 2
        assert "b1" in store
        assert "nonexistent" not in store


# ─── supersession ────────────────────────────────────────────────────

class TestSupersession:
    def test_bidirectional(self, store: BeliefStore) -> None:
        old = _add(store, "old", "I love sushi")
        new = _add(store, "new", "I'm avoiding raw fish")
        store.supersede("old", "new")
        assert "new" in old.superseded_by
        assert "old" in new.supersedes
        assert old.is_superseded
        assert new.is_current

    def test_self_supersession_rejected(self, store: BeliefStore) -> None:
        _add(store, "b1", "x")
        with pytest.raises(ValueError, match="cannot supersede itself"):
            store.supersede("b1", "b1")

    def test_missing_id_raises(self, store: BeliefStore) -> None:
        _add(store, "b1", "x")
        with pytest.raises(KeyError):
            store.supersede("b1", "nonexistent")
        with pytest.raises(KeyError):
            store.supersede("nonexistent", "b1")

    def test_idempotent(self, store: BeliefStore) -> None:
        old = _add(store, "old", "x")
        _add(store, "new", "y")
        store.supersede("old", "new")
        store.supersede("old", "new")  # duplicate — should dedupe
        assert old.superseded_by == ["new"]

    def test_one_new_supersedes_many_olds(self, store: BeliefStore) -> None:
        _add(store, "old1", "I love sushi")
        _add(store, "old2", "I love steak")
        new = _add(store, "new", "I'm vegetarian now")
        store.supersede("old1", "new")
        store.supersede("old2", "new")
        assert "old1" in new.supersedes
        assert "old2" in new.supersedes
        assert not store.get("old1").is_current  # type: ignore[union-attr]
        assert not store.get("old2").is_current  # type: ignore[union-attr]

    def test_non_destructive_old_still_queryable(
        self, store: BeliefStore
    ) -> None:
        _add(store, "old", "I love sushi")
        _add(store, "new", "I'm avoiding raw fish")
        store.supersede("old", "new")
        assert store.get("old") is not None
        assert store.get("old") in store.all()


# ─── reinforcement ───────────────────────────────────────────────────

class TestReinforcement:
    def test_same_source_small_bump(self, store: BeliefStore) -> None:
        """Repeated reinforcement from the SAME session gets a small bump."""
        b1 = _add(store, "b1", "I love sushi", session="s1")
        _add(store, "b2", "I love sushi so much", session="s1")  # same session
        b1.confidence = 0.6  # simulate decay
        store.reinforce("b1", "b2")
        assert "b2" in b1.reinforced_by
        # Same-source bump = 10% of gap. gap=0.4 → 0.6 + 0.04 = 0.64
        assert b1.confidence == pytest.approx(0.64)

    def test_distinct_source_full_bump(self, store: BeliefStore) -> None:
        """Reinforcement from a DISTINCT session gets the full 30% bump."""
        b1 = _add(store, "b1", "I love sushi", session="s1")
        _add(store, "b2", "Sushi is the best", session="s2")  # distinct session
        b1.confidence = 0.6
        store.reinforce("b1", "b2")
        # Distinct-source bump = 30% of gap. gap=0.4 → 0.6 + 0.12 = 0.72
        assert b1.confidence == pytest.approx(0.72)
        assert "s2" in b1.reinforcement_sources

    def test_confidence_caps_at_one(self, store: BeliefStore) -> None:
        b1 = _add(store, "b1", "x")
        b1.confidence = 0.99
        _add(store, "b2", "y", session="s2")
        store.reinforce("b1", "b2")
        assert b1.confidence <= 1.0

    def test_self_reinforce_rejected(self, store: BeliefStore) -> None:
        _add(store, "b1", "x")
        with pytest.raises(ValueError, match="cannot reinforce itself"):
            store.reinforce("b1", "b1")

    def test_source_independence_prevents_runaway(
        self, store: BeliefStore
    ) -> None:
        """Ten reinforcements from one source don't reach full confidence."""
        b1 = _add(store, "b1", "x", session="s1")
        b1.confidence = 0.5
        for i in range(10):
            _add(store, f"reinforcer-{i}", f"x-restated-{i}", session="s1")
            store.reinforce("b1", f"reinforcer-{i}")
        # 10 × same-source 10% gap-closes doesn't saturate
        # starting gap = 0.5, each closes 10%, so after 10 iterations:
        # conf ≈ 1 - 0.5 * 0.9^10 ≈ 1 - 0.5 * 0.3487 ≈ 0.826
        assert b1.confidence < 0.95


# ─── multi-outcome resolution ────────────────────────────────────────

class TestCoexist:
    def test_bidirectional_and_status_promotion(self, store: BeliefStore) -> None:
        a = _add(store, "a", "I like sushi")
        b = _add(store, "b", "I also like steak")
        store.coexist("a", "b")
        assert "b" in a.coexists_with
        assert "a" in b.coexists_with
        assert a.status == ResolutionStatus.COEXISTS
        assert b.status == ResolutionStatus.COEXISTS

    def test_self_coexist_rejected(self, store: BeliefStore) -> None:
        _add(store, "a", "x")
        with pytest.raises(ValueError, match="cannot coexist with itself"):
            store.coexist("a", "a")

    def test_coexisting_query(self, store: BeliefStore) -> None:
        _add(store, "a", "x")
        _add(store, "b", "y")
        _add(store, "c", "z")
        store.coexist("a", "b")
        coexisting = store.coexisting()
        assert {x.id for x in coexisting} == {"a", "b"}


class TestDispute:
    def test_dispute_symmetric_and_status(self, store: BeliefStore) -> None:
        a = _add(store, "a", "Ravi is lead")
        b = _add(store, "b", "Emma is lead")
        store.dispute("a", "b")
        assert "b" in a.disputed_with
        assert "a" in b.disputed_with
        assert a.status == ResolutionStatus.DISPUTED
        assert b.status == ResolutionStatus.DISPUTED
        # Still queryable, both remain current-ish (not superseded)
        assert not a.is_superseded
        assert not b.is_superseded
        assert a.is_disputed
        assert b.is_disputed

    def test_ambiguous_flag_sets_ambiguous_status(
        self, store: BeliefStore
    ) -> None:
        _add(store, "a", "x")
        _add(store, "b", "y")
        store.dispute("a", "b", ambiguous=True)
        assert store.get("a").status == ResolutionStatus.AMBIGUOUS  # type: ignore[union-attr]

    def test_disputed_query(self, store: BeliefStore) -> None:
        _add(store, "a", "x")
        _add(store, "b", "y")
        _add(store, "c", "z")
        store.dispute("a", "b")
        disputed = store.disputed()
        assert {x.id for x in disputed} == {"a", "b"}

    def test_self_dispute_rejected(self, store: BeliefStore) -> None:
        _add(store, "a", "x")
        with pytest.raises(ValueError, match="cannot dispute itself"):
            store.dispute("a", "a")


class TestResolveDispute:
    def test_winner_supersedes_loser(self, store: BeliefStore) -> None:
        winner = _add(store, "winner", "Emma is lead now")
        loser = _add(store, "loser", "Ravi is lead")
        store.dispute("winner", "loser")

        store.resolve_dispute("winner", "loser")

        assert loser.status == ResolutionStatus.SUPERSEDED
        assert winner.status == ResolutionStatus.CURRENT
        # Supersession edges were added
        assert "loser" in winner.supersedes
        assert "winner" in loser.superseded_by
        # Dispute edges cleaned up
        assert loser.disputed_with == []
        assert winner.disputed_with == []

    def test_resolve_non_disputed_pair_raises(
        self, store: BeliefStore
    ) -> None:
        _add(store, "a", "x")
        _add(store, "b", "y")
        with pytest.raises(ValueError, match="not disputed with"):
            store.resolve_dispute("a", "b")


class TestArchive:
    def test_archive_sets_status(self, store: BeliefStore) -> None:
        a = _add(store, "a", "x")
        store.archive("a")
        assert a.status == ResolutionStatus.ARCHIVED
        # Not current any more (is_current checks archive too)
        assert not a.is_current

    def test_archived_query(self, store: BeliefStore) -> None:
        _add(store, "a", "x")
        _add(store, "b", "y")
        store.archive("a")
        assert {b.id for b in store.archived()} == {"a"}


class TestByStatus:
    def test_by_status_filters(self, store: BeliefStore) -> None:
        _add(store, "a", "x")
        b = _add(store, "b", "y")
        c = _add(store, "c", "z")
        store.supersede("a", "b")
        store.archive("c")
        # a is SUPERSEDED, b is CURRENT, c is ARCHIVED
        assert {x.id for x in store.by_status(ResolutionStatus.SUPERSEDED)} == {"a"}
        assert {x.id for x in store.by_status(ResolutionStatus.CURRENT)} == {"b"}
        assert {x.id for x in store.by_status(ResolutionStatus.ARCHIVED)} == {"c"}


# ─── queries ─────────────────────────────────────────────────────────

class TestQueries:
    def test_current_vs_superseded(self, store: BeliefStore) -> None:
        _add(store, "b1", "I love sushi")
        _add(store, "b2", "I avoid sushi")
        store.supersede("b1", "b2")
        current = store.current()
        superseded = store.superseded()
        assert len(current) == 1 and current[0].id == "b2"
        assert len(superseded) == 1 and superseded[0].id == "b1"

    def test_lineage_walks_history(self, store: BeliefStore) -> None:
        # Three-generation chain: b1 ← b2 ← b3
        _add(store, "b1", "I live in Sydney")
        _add(store, "b2", "I moved to Sofia")
        _add(store, "b3", "I moved to Berlin")
        store.supersede("b1", "b2")
        store.supersede("b2", "b3")

        lineage = store.lineage("b3")
        ids = [b.id for b in lineage]
        assert ids == ["b3", "b2", "b1"]

    def test_lineage_of_leaf_is_just_itself(self, store: BeliefStore) -> None:
        _add(store, "b1", "x")
        lineage = store.lineage("b1")
        assert [b.id for b in lineage] == ["b1"]

    def test_lineage_of_missing(self, store: BeliefStore) -> None:
        assert store.lineage("nonexistent") == []

    def test_by_session(self, store: BeliefStore) -> None:
        _add(store, "b1", "x", session="s1")
        _add(store, "b2", "y", session="s1")
        _add(store, "b3", "z", session="s2")
        s1_beliefs = store.by_session("s1")
        assert {b.id for b in s1_beliefs} == {"b1", "b2"}

    def test_by_proposition(self, store: BeliefStore) -> None:
        _add(store, "b1", "x")
        found = store.by_proposition("prop-b1")
        assert found is not None and found.id == "b1"
        assert store.by_proposition("nonexistent") is None


# ─── persistence ─────────────────────────────────────────────────────

class TestPersistence:
    def test_roundtrip_via_replay(self, tmp_path: Path) -> None:
        path = tmp_path / "beliefs.jsonl"

        # First session: add, supersede, reinforce
        s1 = BeliefStore(persistence_path=path)
        s1.add(
            proposition="I love sushi",
            asserted_at=datetime(2024, 1, 1, 12, 0),
            asserted_in_session="s1",
            source_proposition_id="p1",
            belief_id="old",
        )
        s1.add(
            proposition="I'm avoiding raw fish",
            asserted_at=datetime(2024, 3, 1, 12, 0),
            asserted_in_session="s2",
            source_proposition_id="p2",
            belief_id="new",
            validity=Validity(
                mode="dated_range",
                start=datetime(2024, 3, 1),
                end=datetime(2024, 9, 1),
                source="explicit",
            ),
        )
        s1.supersede("old", "new")

        # Second session: load from disk, state should match
        s2 = BeliefStore(persistence_path=path)
        assert len(s2) == 2
        assert s2.get("old") is not None
        assert s2.get("new") is not None
        assert s2.get("old").is_superseded  # type: ignore[union-attr]
        assert s2.get("new").is_current  # type: ignore[union-attr]
        # Validity preserved
        new = s2.get("new")
        assert new is not None
        assert new.validity.mode == "dated_range"
        assert new.validity.source == "explicit"
        assert new.validity.start == datetime(2024, 3, 1)
        assert new.validity.end == datetime(2024, 9, 1)

    def test_events_append_only(self, tmp_path: Path) -> None:
        path = tmp_path / "beliefs.jsonl"
        s = BeliefStore(persistence_path=path)
        s.add(
            proposition="x",
            asserted_at=datetime(2024, 1, 1),
            asserted_in_session="s1",
            source_proposition_id="p1",
            belief_id="b1",
        )
        # One event line expected
        assert len(path.read_text().strip().split("\n")) == 1

        s.add(
            proposition="y",
            asserted_at=datetime(2024, 1, 2),
            asserted_in_session="s1",
            source_proposition_id="p2",
            belief_id="b2",
        )
        s.supersede("b1", "b2")
        assert len(path.read_text().strip().split("\n")) == 3

    def test_no_persistence_path_leaves_no_file(
        self, tmp_path: Path
    ) -> None:
        # Default: no persistence
        s = BeliefStore()
        s.add(
            proposition="x",
            asserted_at=datetime(2024, 1, 1),
            asserted_in_session="s1",
            source_proposition_id="p1",
        )
        # No file should have been created in tmp_path
        assert list(tmp_path.iterdir()) == []
