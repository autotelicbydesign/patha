"""Tests for upamāna (analogical recall — the sixth question class).

Model-free where possible (injected embedder); AnalogyEval owns quality
measurement, this suite pins detection, the ranking formula's anti-trap
property, frame naming, and recall() integration."""

from __future__ import annotations

from datetime import datetime

from patha.belief.upamana import (
    detect_analogy_question,
    frames_of,
    rank_analogues,
)


class TestDetection:
    def test_analogy_phrasings_fire(self):
        for q in (
            "what does this remind me of?",
            "have I been in a situation like this before?",
            "what past project was most similar to this one?",
            "is this like the time I switched teams?",
            "what's the closest thing in my past to this negotiation?",
            "does this job offer resemble any decision I've faced before?",
        ):
            assert detect_analogy_question(q), q

    def test_non_analogy_declines(self):
        for q in (
            "what did I say about the saddle?",
            "how much did I spend on the bike in total?",
            "how has my thinking about budgeting evolved?",
            "have I ever been to Japan?",
        ):
            assert not detect_analogy_question(q), q


class TestRanking:
    def test_lexical_overlap_penalty_beats_the_trap(self):
        # gold shares structure, trap shares words: with equal semantic
        # similarity the overlap penalty must rank gold first
        episodes = {
            "s-gold": ["the offer came with a two-day expiry and I "
                       "flip-flopped all night before committing"],
            "s-trap": ["bought new running shoes for marathon training "
                       "and a knee sleeve for running volume"],
        }
        q = ("my marathon training hit an injury and I have to pull my "
             "running volume back — when have I handled this?")
        result = rank_analogues(
            q, episodes, embed_fn=lambda texts: [[1.0, 0.0]] * len(texts),
        )
        # identical embeddings → only the penalty differentiates
        assert result.sessions[0] == "s-gold"

    def test_deterministic_tiebreak(self):
        episodes = {"s-b": ["text one"], "s-a": ["text two"]}
        r1 = rank_analogues("anything like this before?", episodes,
                            embed_fn=lambda t: [[1.0]] * len(t))
        r2 = rank_analogues("anything like this before?", episodes,
                            embed_fn=lambda t: [[1.0]] * len(t))
        assert r1.sessions == r2.sessions  # name-ordered tiebreak

    def test_frames_shared_structure(self):
        q = "the landlord gave us a deadline and I keep flip-flopping"
        text = ("the offer came with a 48-hour expiry and I went back "
                "and forth all night")
        shared = set(frames_of(q)) & set(frames_of(text))
        assert "hard time limit forcing a choice" in shared


class TestRouting:
    def test_recall_routes_analogy_and_controls_hold(self, tmp_path):
        import patha
        mem = patha.Memory(path=tmp_path / "b.jsonl", detector="stub",
                           enable_phase1=True)
        for text, sess, when in (
            ("the job offer came with a 48-hour expiry and I "
             "flip-flopped all night", "s-offer", datetime(2024, 3, 1)),
            ("accepted an hour before the deadline and felt instantly "
             "calm", "s-offer", datetime(2024, 3, 2)),
            ("picked the quartz counters after months of browsing",
             "s-kitchen", datetime(2024, 6, 1)),
            ("the contractor starts in May", "s-kitchen",
             datetime(2024, 6, 10)),
        ):
            mem.remember(text, session_id=sess, asserted_at=when)
        r = mem.recall("the landlord gave us until Friday to sign and "
                       "I keep going back and forth — have I been in a "
                       "situation like this before?")
        assert r.strategy == "analogy"
        assert r.analogy is not None
        assert r.analogy.sessions[0] == "s-offer"
        assert r.tokens == 0
        # control: retrieval phrasing untouched
        r2 = mem.recall("what did I say about the contractor?")
        assert r2.strategy != "analogy" and r2.analogy is None
