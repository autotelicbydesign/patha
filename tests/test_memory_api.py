"""Tests for the top-level `patha.Memory` developer API.

Thin wrapper over IntegratedPatha, but the public surface needs
its own regression coverage — the whole point of this API is that
`import patha; patha.Memory()` should be stable across versions.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import patha


class TestMemoryConstruction:
    def test_default_path(self, monkeypatch, tmp_path):
        # Temporarily move ~ to tmp_path so we don't touch the real home dir
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        m = patha.Memory()
        assert str(m.path).endswith(".patha/beliefs.jsonl")
        assert m.path.parent.exists()

    def test_explicit_path(self, tmp_path):
        p = tmp_path / "my_memory.jsonl"
        m = patha.Memory(path=p)
        assert m.path == p

    def test_repr_doesnt_crash(self, tmp_path):
        m = patha.Memory(path=tmp_path / "x.jsonl")
        assert "Memory" in repr(m)

    def test_invalid_detector_name_raises(self, tmp_path):
        with pytest.raises(ValueError):
            patha.Memory(path=tmp_path / "x.jsonl", detector="not-a-real-detector")


class TestRememberRecall:
    def test_basic_ingest_then_recall(self, tmp_path):
        m = patha.Memory(path=tmp_path / "m.jsonl")
        r = m.remember("I live in Lisbon")
        assert r["action"] == "added"
        assert r["belief_id"]
        assert r["proposition"] == "I live in Lisbon"

        rec = m.recall("what do I know about the user?")
        assert isinstance(rec, patha.Recall)
        assert rec.summary
        assert rec.strategy in ("direct_answer", "structured", "raw")
        assert rec.tokens > 0

    def test_recall_returns_compact_summary(self, tmp_path):
        """The whole pitch of the library is token compression."""
        m = patha.Memory(path=tmp_path / "m.jsonl")
        for claim in [
            "I live in Lisbon",
            "I work at Anthropic",
            "I am vegetarian",
            "I prefer dark chocolate",
            "I ride a bike to work",
        ]:
            m.remember(claim)
        rec = m.recall("what do I know about the user?")
        # Naive conversation-history would dump all 5 propositions (>40 tokens).
        # Our structured summary should be compact regardless.
        assert rec.tokens < 200  # generous upper bound

    def test_history_finds_past_mentions(self, tmp_path):
        m = patha.Memory(path=tmp_path / "m.jsonl")
        m.remember("I live in Sofia")
        m.remember("The weather is nice")
        matches = m.history("Sofia")
        assert len(matches) == 1
        assert matches[0]["proposition"] == "I live in Sofia"
        assert matches[0]["status"] == "current"

    def test_history_is_case_insensitive(self, tmp_path):
        m = patha.Memory(path=tmp_path / "m.jsonl")
        m.remember("I work at Anthropic")
        assert len(m.history("ANTHROPIC")) == 1
        assert len(m.history("anthropic")) == 1


class TestPersistence:
    def test_survives_reopening(self, tmp_path):
        """The headline portability claim: open the same file, same data."""
        p = tmp_path / "m.jsonl"
        m1 = patha.Memory(path=p)
        m1.remember("I live in Lisbon")
        assert p.exists()

        # Simulate process restart: new Memory instance over same file
        m2 = patha.Memory(path=p)
        stats = m2.stats()
        assert stats["total"] == 1
        assert stats["current"] == 1
        assert m2.history("Lisbon")[0]["proposition"] == "I live in Lisbon"


class TestEscapeHatches:
    def test_belief_layer_accessible(self, tmp_path):
        m = patha.Memory(path=tmp_path / "m.jsonl")
        from patha.belief import BeliefLayer
        assert isinstance(m.belief_layer, BeliefLayer)

    def test_store_accessible(self, tmp_path):
        m = patha.Memory(path=tmp_path / "m.jsonl")
        from patha.belief import BeliefStore
        assert isinstance(m.store, BeliefStore)

    def test_path_exposed(self, tmp_path):
        p = tmp_path / "m.jsonl"
        m = patha.Memory(path=p)
        assert m.path == p


class TestExports:
    def test_top_level_exports(self):
        """New developers should find these at `patha.X`."""
        expected = {
            "Memory", "Recall",
            "BeliefLayer", "BeliefStore", "IntegratedPatha",
            "IntegratedResponse", "DirectAnswerer",
            "make_detector", "AVAILABLE_DETECTORS",
            "__version__",
        }
        for name in expected:
            assert hasattr(patha, name), f"patha.{name} missing"

    def test_version_format(self):
        assert isinstance(patha.__version__, str)
        # Rough semver shape check
        parts = patha.__version__.split(".")
        assert len(parts) >= 2
