"""Tests for the Raw Archive Layer (provenance substrate)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from patha.belief.raw_archive import RawArchive, RawTurn, _content_hash


# ─── add_turn ───────────────────────────────────────────────────────

class TestAddTurn:
    def test_basic_add(self) -> None:
        arch = RawArchive()
        t = arch.add_turn(
            session_id="s1",
            turn_index=0,
            speaker="user",
            content="hello",
            timestamp=datetime(2024, 1, 1, 12, 0),
        )
        assert isinstance(t, RawTurn)
        assert len(arch) == 1
        assert t.id in arch

    def test_content_addressable_id_default(self) -> None:
        arch = RawArchive()
        t1 = arch.add_turn(
            session_id="s1", turn_index=0, speaker="user",
            content="hello", timestamp=datetime(2024, 1, 1),
        )
        # Same content → same id
        expected = _content_hash("s1", 0, "user", "hello")
        assert t1.id == expected

    def test_idempotent_on_duplicate(self) -> None:
        arch = RawArchive()
        t1 = arch.add_turn(
            session_id="s1", turn_index=0, speaker="user",
            content="hi", timestamp=datetime(2024, 1, 1),
        )
        t2 = arch.add_turn(
            session_id="s1", turn_index=0, speaker="user",
            content="hi", timestamp=datetime(2024, 1, 1),
        )
        assert t1.id == t2.id
        assert len(arch) == 1

    def test_explicit_id_accepted(self) -> None:
        arch = RawArchive()
        t = arch.add_turn(
            session_id="s1", turn_index=0, speaker="user",
            content="hi", timestamp=datetime(2024, 1, 1),
            raw_turn_id="custom-id",
        )
        assert t.id == "custom-id"

    def test_metadata_preserved(self) -> None:
        arch = RawArchive()
        t = arch.add_turn(
            session_id="s1", turn_index=0, speaker="user",
            content="hi", timestamp=datetime(2024, 1, 1),
            source_name="slack-dm",
            metadata={"channel": "general", "msg_id": "42"},
        )
        assert t.source_name == "slack-dm"
        assert t.metadata["channel"] == "general"


# ─── link_proposition ───────────────────────────────────────────────

class TestLinkProposition:
    def test_basic_link(self) -> None:
        arch = RawArchive()
        t = arch.add_turn(
            session_id="s1", turn_index=0, speaker="user",
            content="i love sushi and i work at Canva",
            timestamp=datetime(2024, 1, 1),
        )
        arch.link_proposition(
            raw_turn_id=t.id, proposition_id="prop-1"
        )
        arch.link_proposition(
            raw_turn_id=t.id, proposition_id="prop-2"
        )
        assert arch.turn_for_proposition("prop-1") is t
        assert arch.turn_for_proposition("prop-2") is t
        assert "prop-1" in t.derived_proposition_ids

    def test_unknown_turn_raises(self) -> None:
        arch = RawArchive()
        with pytest.raises(KeyError):
            arch.link_proposition(
                raw_turn_id="nonexistent", proposition_id="p1"
            )

    def test_duplicate_link_is_idempotent(self) -> None:
        arch = RawArchive()
        t = arch.add_turn(
            session_id="s1", turn_index=0, speaker="user",
            content="x", timestamp=datetime(2024, 1, 1),
        )
        arch.link_proposition(raw_turn_id=t.id, proposition_id="p1")
        arch.link_proposition(raw_turn_id=t.id, proposition_id="p1")
        # Second call is a no-op; proposition still linked once
        assert t.derived_proposition_ids.count("p1") == 1

    def test_rebind_rejected(self) -> None:
        arch = RawArchive()
        t1 = arch.add_turn(
            session_id="s1", turn_index=0, speaker="user",
            content="x", timestamp=datetime(2024, 1, 1),
        )
        t2 = arch.add_turn(
            session_id="s1", turn_index=1, speaker="user",
            content="y", timestamp=datetime(2024, 1, 2),
        )
        arch.link_proposition(raw_turn_id=t1.id, proposition_id="p1")
        with pytest.raises(ValueError, match="already linked"):
            arch.link_proposition(raw_turn_id=t2.id, proposition_id="p1")


# ─── queries ────────────────────────────────────────────────────────

class TestQueries:
    def test_turns_by_session(self) -> None:
        arch = RawArchive()
        arch.add_turn(
            session_id="s1", turn_index=0, speaker="user",
            content="hi", timestamp=datetime(2024, 1, 1),
        )
        arch.add_turn(
            session_id="s1", turn_index=1, speaker="assistant",
            content="hello", timestamp=datetime(2024, 1, 1, 12, 1),
        )
        arch.add_turn(
            session_id="s2", turn_index=0, speaker="user",
            content="bye", timestamp=datetime(2024, 1, 2),
        )
        turns = arch.turns_by_session("s1")
        assert len(turns) == 2
        assert {t.turn_index for t in turns} == {0, 1}

    def test_turn_for_unknown_proposition(self) -> None:
        arch = RawArchive()
        assert arch.turn_for_proposition("nonexistent") is None


# ─── persistence ────────────────────────────────────────────────────

class TestPersistence:
    def test_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "archive.jsonl"

        a1 = RawArchive(persistence_path=path)
        t = a1.add_turn(
            session_id="s1", turn_index=0, speaker="user",
            content="doctor told me I have X",
            timestamp=datetime(2024, 1, 1),
            source_name="voice-memo",
            metadata={"file": "/tmp/a.wav"},
        )
        a1.link_proposition(raw_turn_id=t.id, proposition_id="prop-1")

        a2 = RawArchive(persistence_path=path)
        assert len(a2) == 1
        restored = a2.get_turn(t.id)
        assert restored is not None
        assert restored.content == "doctor told me I have X"
        assert restored.source_name == "voice-memo"
        assert restored.metadata["file"] == "/tmp/a.wav"
        assert a2.turn_for_proposition("prop-1") is restored

    def test_append_only_log(self, tmp_path: Path) -> None:
        path = tmp_path / "archive.jsonl"
        arch = RawArchive(persistence_path=path)
        arch.add_turn(
            session_id="s1", turn_index=0, speaker="user",
            content="x", timestamp=datetime(2024, 1, 1),
        )
        t1 = len(path.read_text().strip().split("\n"))
        arch.add_turn(
            session_id="s1", turn_index=1, speaker="user",
            content="y", timestamp=datetime(2024, 1, 2),
        )
        t2 = len(path.read_text().strip().split("\n"))
        assert t2 > t1

    def test_no_persistence_writes_nothing(self, tmp_path: Path) -> None:
        arch = RawArchive()
        arch.add_turn(
            session_id="s1", turn_index=0, speaker="user",
            content="x", timestamp=datetime(2024, 1, 1),
        )
        assert list(tmp_path.iterdir()) == []
