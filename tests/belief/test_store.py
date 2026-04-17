"""Tests for the belief store."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from patha.belief.store import BeliefStore
from patha.belief.types import Validity


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
    def test_records_reinforced_by(self, store: BeliefStore) -> None:
        b1 = _add(store, "b1", "I love sushi")
        _add(store, "b2", "I love sushi so much")
        assert b1.confidence == 1.0
        b1.confidence = 0.6  # simulate decay
        store.reinforce("b1", "b2")
        assert "b2" in b1.reinforced_by
        # Confidence should bump toward 1.0 by 30% of gap.
        # gap was 0.4 → expect 0.6 + 0.12 = 0.72
        assert b1.confidence == pytest.approx(0.72)

    def test_confidence_caps_at_one(self, store: BeliefStore) -> None:
        b1 = _add(store, "b1", "x")
        b1.confidence = 0.99
        _add(store, "b2", "y")
        store.reinforce("b1", "b2")
        assert b1.confidence <= 1.0

    def test_self_reinforce_rejected(self, store: BeliefStore) -> None:
        _add(store, "b1", "x")
        with pytest.raises(ValueError, match="cannot reinforce itself"):
            store.reinforce("b1", "b1")


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
