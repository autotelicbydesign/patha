"""Verify the three innovations compose without interfering.

A user that turns on all three together (Hebbian expansion +
karaṇa ingest-time extraction + filesystem import) should see them
work harmoniously:

  1. The importer creates beliefs with frontmatter dates.
  2. The karaṇa extractor turns those beliefs into gaṇita tuples
     at ingest time.
  3. Hebbian co-retrieval edges accumulate naturally as queries fire.
  4. A subsequent aggregation question uses the gaṇita arithmetic
     (Innovation #2) restricted to a Phase-2 candidate set that was
     expanded via Hebbian (Innovation #1).

This test deliberately avoids Ollama — we use a scripted karaṇa
extractor so it's deterministic and CI-friendly.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from textwrap import dedent

import pytest

import patha
from patha.belief.ganita import GanitaTuple
from patha.importers import import_obsidian_vault


class _ScriptedKarana:
    """A karaṇa extractor that emits a fixed tuple per ingest, in
    sequence. Lets us pretend the LLM correctly identified each
    expense."""

    def __init__(self, scripted: list[GanitaTuple]) -> None:
        self._scripted = list(scripted)
        self._cursor = 0

    def extract(self, text, *, belief_id, time=None):
        if self._cursor >= len(self._scripted):
            return []
        tup = self._scripted[self._cursor]
        self._cursor += 1
        # Re-issue under the actual belief_id this ingest got.
        return [GanitaTuple(
            entity=tup.entity, attribute=tup.attribute,
            value=tup.value, unit=tup.unit, time=time,
            belief_id=belief_id, raw_text=tup.raw_text,
            entity_aliases=tup.entity_aliases,
        )]


def test_obsidian_then_hebbian_then_ganita(tmp_path: Path) -> None:
    """End-to-end: Obsidian import → karaṇa extraction → Hebbian
    expansion → gaṇita aggregation."""

    # Step 1 — make a small Obsidian vault.
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "2024-01-15.md").write_text(dedent("""\
        ---
        date: 2024-01-15
        ---
        I bought a $50 saddle for my [[bike]] today.
    """))
    (vault / "2024-01-20.md").write_text(dedent("""\
        ---
        date: 2024-01-20
        ---
        Got the [[helmet]] for the bike — $75.
    """))
    (vault / "2024-02-01.md").write_text(dedent("""\
        ---
        date: 2024-02-01
        ---
        Bought $30 in lights and $30 in gloves for the bike.
    """))

    # Step 2 — use a scripted karaṇa extractor that pretends the LLM
    # correctly tagged each ingest. (Real LLM is verified separately
    # in test_karana_ollama_live.py.)
    scripted = [
        GanitaTuple(entity="saddle", attribute="expense", value=50,
                    unit="USD", time=None, belief_id="seed",
                    raw_text="$50 saddle bike",
                    entity_aliases=("saddle", "bike")),
        GanitaTuple(entity="helmet", attribute="expense", value=75,
                    unit="USD", time=None, belief_id="seed",
                    raw_text="helmet bike $75",
                    entity_aliases=("helmet", "bike")),
        GanitaTuple(entity="light", attribute="expense", value=60,
                    unit="USD", time=None, belief_id="seed",
                    raw_text="$30 lights $30 gloves bike",
                    entity_aliases=("light", "glove", "bike")),
    ]

    # Step 3 — Patha with all three innovations on.
    mem = patha.Memory(
        path=tmp_path / "store.jsonl",
        enable_phase1=False,  # not needed — we feed candidates directly
        hebbian_expansion=True,
        hebbian_session_seed_weight=0.05,
        karana_extractor=_ScriptedKarana(scripted),
    )

    # Step 4 — import the vault.
    stats = import_obsidian_vault(vault, mem)
    assert stats.beliefs_added == 3
    # Each ingested belief produced exactly one gaṇita tuple.
    assert len(mem._ganita_index) == 3

    # Step 5 — ask the aggregation question.
    rec = mem.recall(
        "how much total did I spend on bike-related expenses?",
        at_time=datetime(2024, 6, 1),
    )
    # gaṇita should fire because every tuple has 'bike' in its aliases.
    assert rec.ganita is not None
    assert rec.ganita.value == 50 + 75 + 60  # = 185
    assert rec.ganita.unit == "USD"
    # All 3 source beliefs contributed.
    assert len(rec.ganita.contributing_belief_ids) == 3


def test_hebbian_does_not_break_ganita(tmp_path: Path) -> None:
    """When Hebbian expansion adds beliefs to the Phase-2 candidate
    set, those beliefs' gaṇita tuples should still flow into the
    aggregation answer."""

    scripted = [
        GanitaTuple(entity="bike", attribute="expense", value=100,
                    unit="USD", time=None, belief_id="seed",
                    raw_text="bike $100", entity_aliases=("bike",)),
    ]

    mem = patha.Memory(
        path=tmp_path / "store.jsonl",
        enable_phase1=False,
        hebbian_expansion=True,
        hebbian_session_seed_weight=0.05,
        karana_extractor=_ScriptedKarana(scripted),
    )
    mem.remember("Bought a $100 bike accessory",
                 asserted_at=datetime(2024, 1, 1),
                 session_id="bike-session")

    rec = mem.recall("how much did I spend on bike?",
                     at_time=datetime(2024, 6, 1))
    assert rec.ganita is not None
    assert rec.ganita.value == 100


def test_synthesis_intent_bypasses_phase1(tmp_path: Path) -> None:
    """The architectural distinction:
       retrieval queries go through Phase 1; synthesis queries don't.

    Patha's gaṇita layer detects synthesis intent (sum/count/avg/min/max)
    and queries the belief store DIRECTLY — Phase 1 never gets called.
    Top-K retrieval is the wrong primitive for synthesis: top-K of N
    misses (N-K) of the inputs you need to sum.

    This test installs a Phase-1 retriever that would mis-rank the
    bike-shopping sessions out of the candidate set entirely. A
    synthesis question still recovers \$185 because gaṇita bypasses
    Phase 1.
    """

    class _ScriptedKarana:
        """Emits a per-fact tuple with bike alias, like a real LLM."""

        _facts = [
            ("saddle", 50.0),
            ("helmet", 75.0),
            ("light", 30.0),
            ("glove", 30.0),
        ]
        _i = 0

        def extract(self, text, *, belief_id, time=None):
            if self._i >= len(self._facts):
                return []
            entity, value = self._facts[self._i]
            self._i += 1
            return [GanitaTuple(
                entity=entity, attribute="expense", value=value,
                unit="USD", time=time, belief_id=belief_id,
                raw_text=f"${value:.0f} {entity}",
                entity_aliases=(entity, "bike"),
            )]

    mem = patha.Memory(
        path=tmp_path / "store.jsonl",
        enable_phase1=False,  # would be irrelevant anyway — synthesis bypasses
        karana_extractor=_ScriptedKarana(),
    )
    # Ingest 4 bike-shopping facts across different sessions
    for i, fact in enumerate([
        "I bought a $50 saddle for my bike",
        "I got a $75 helmet for the bike",
        "$30 for new bike lights",
        "I spent $30 on bike gloves",
    ]):
        mem.remember(fact, asserted_at=datetime(2024, 1, i + 1),
                     session_id=f"bike-{i}")

    # SABOTAGE Phase 2: replace the phase1_retrieve with one that
    # returns NOTHING. This proves the synthesis-intent path doesn't
    # touch retrieval — if it did, we'd get 0 here.
    mem._patha._phase1_retrieve = lambda q, k: []

    # Synthesis question — must recover $185 even with retrieval gone.
    rec = mem.recall("how much have I spent on bike-related expenses?",
                     at_time=datetime(2024, 6, 1))
    assert rec.ganita is not None
    assert rec.ganita.value == 185.0
    assert rec.strategy == "ganita"
    # `rec.current` is populated with the SOURCE beliefs that gaṇita
    # summed — the user sees what was used to compute the answer.
    # All 4 bike-shopping facts contributed.
    assert len(rec.current) == 4
    assert all("bike" in c["proposition"].lower() for c in rec.current)
    assert rec.tokens == 0  # zero LLM tokens at recall — that's the point


def test_retrieval_intent_uses_phase1(tmp_path: Path) -> None:
    """Conversely: a perception question goes through Phase 1.
    No aggregation operator → no synthesis path → standard retrieval."""

    class _NoOpKarana:
        def extract(self, text, *, belief_id, time=None):
            return []

    mem = patha.Memory(
        path=tmp_path / "store.jsonl",
        enable_phase1=False,
        karana_extractor=_NoOpKarana(),
    )
    mem.remember("I love sushi every week",
                 asserted_at=datetime(2024, 1, 1),
                 session_id="food")

    # Mock Phase-1 to return the proposition by id
    bid = mem._patha.belief_layer.store.all()[0].source_proposition_id
    mem._patha._phase1_retrieve = lambda q, k: [bid]

    rec = mem.recall("what do I like to eat?",
                     at_time=datetime(2024, 6, 1))
    # Retrieval path: ganita didn't fire; got the structured/raw result
    assert rec.ganita is None
    assert rec.strategy in ("structured", "direct_answer", "raw")
    assert any("sushi" in c["proposition"].lower() for c in rec.current)


def test_dense_haystack_phase1_misses_some_bike_sessions(tmp_path: Path) -> None:
    r"""Reproduces the synthesis-bounded LongMemEval failure mode.

    Setup: 50 sessions, of which:
      - 4 are bike-shopping sessions ($50, $75, $30, $30 in expenses)
      - 46 are random other topics (rent, groceries, dental, etc)

    Phase 1 returns the wrong cluster — say, 10 random sessions plus
    only 1 of the bike sessions. Without our fix, ``restrict_to_belief_ids``
    would scope arithmetic to those 10 retrieved sessions and miss the
    other 3 bike-expense tuples, returning a partial sum or worse —
    the noise from one bike-mentioning random session.

    With the fix, the global entity+attribute match recovers all 4
    bike-expense tuples and returns \$185 deterministically.
    """
    from patha.belief.ganita import (
        GanitaIndex, answer_aggregation_question,
    )

    idx = GanitaIndex()
    # 4 bike-expense tuples (the gold answer)
    for value, item, bid in [
        (50, "saddle", "bike-1"),
        (75, "helmet", "bike-2"),
        (30, "light", "bike-3"),
        (30, "glove", "bike-4"),
    ]:
        idx.add(GanitaTuple(
            entity=item, attribute="expense", value=value, unit="USD",
            time=None, belief_id=bid,
            raw_text=f"${value} {item} bike",
            entity_aliases=(item, "bike"),
        ))

    # 5 random other-topic expense tuples whose source text mentions "bike"
    # in passing (e.g., "I rented a place near the bike path").
    # These shouldn't be summed for "bike-related expenses".
    for value, item, bid in [
        (1500, "rent", "rent-1"),       # "rent for a flat near the bike path"
        (200, "dentist", "dental-1"),
        (120, "groceries", "grocery-1"),
        (300, "concert", "music-1"),
        (180, "gym", "gym-1"),
    ]:
        idx.add(GanitaTuple(
            entity=item, attribute="expense", value=value, unit="USD",
            time=None, belief_id=bid,
            raw_text=f"${value} for {item}",
            entity_aliases=(item,),  # NO "bike" alias — the LLM
            # correctly didn't tag these as bike-related
        ))

    # Phase 1 retrieved ONLY some of the random beliefs and ONE bike belief
    retrieved_belief_ids = {"rent-1", "dental-1", "music-1", "bike-3"}

    # Question: how much spent on bike-related expenses?
    rec = answer_aggregation_question(
        "how much total did I spend on bike-related expenses?",
        idx,
        restrict_to_belief_ids=retrieved_belief_ids,
    )

    # With the fix:
    # - Pull all bike-aliased expense tuples → 4 (saddle, helmet, light, glove)
    # - Post-attribute filter still has 4
    # - 4 ≤ ambiguity_threshold (30) → don't restrict by retrieved
    # - Sum = $185
    assert rec is not None
    assert rec.value == 185.0, (
        f"expected $185 (the global bike-expense sum), got "
        f"${rec.value}. Restriction kicked in despite a precise "
        f"4-tuple match."
    )
    assert len(rec.contributing_belief_ids) == 4


def test_dedup_distinguishes_different_entities_same_value(
    tmp_path: Path,
) -> None:
    """Two genuinely-different purchases at the same price ($40 bike
    lights AND $40 bike pump) must NOT be deduped. Dedup only collapses
    tuples that match on (entity, attribute, value, unit) — different
    entities are different facts even at the same value."""

    class _AlternatingKarana:
        """Emits a different bike-related $40 expense per ingest."""
        _items = [("lights", "bike"), ("pump", "bike")]
        _i = 0

        def extract(self, text, *, belief_id, time=None):
            entity, alias = self._items[self._i % len(self._items)]
            self._i += 1
            return [GanitaTuple(
                entity=entity, attribute="expense",
                value=40.0, unit="USD", time=time,
                belief_id=belief_id,
                raw_text=f"$40 {entity}",
                entity_aliases=(entity, alias),
            )]

    mem = patha.Memory(
        path=tmp_path / "store.jsonl",
        enable_phase1=False,
        karana_extractor=_AlternatingKarana(),
    )
    mem.remember("Bought $40 bike lights")
    mem.remember("Bought $40 bike pump")

    # Both should remain — different entities, even though same value.
    assert len(mem._ganita_index) == 2

    rec = mem.recall("how much have I spent on bike-related expenses?")
    assert rec.ganita is not None
    assert rec.ganita.value == 80.0  # NOT $40 (deduped)
    assert len(rec.ganita.contributing_belief_ids) == 2


def test_same_fact_across_multiple_sessions_dedups(tmp_path: Path) -> None:
    """The same purchase ($40 bike lights) mentioned across N sessions
    must count once, not N times, in the aggregation arithmetic.

    LongMemEval haystacks routinely repeat facts in different
    sessions (the user reminisces). Without dedup, gaṇita would over-
    count by a factor of N. The fix has two layers:
      1. The belief layer reinforces (doesn't add) on duplicate
         assertions when a real detector is in use.
      2. The karaṇa pipeline drops tuples whose (entity, attribute,
         value, unit) already exists, even if the belief layer added
         a new belief (stub detector misses semantic duplicates).
    """

    class _BikeLightsKarana:
        """Always emits the bike-lights tuple regardless of input —
        simulates the LLM seeing the same fact restated."""

        def extract(self, text, *, belief_id, time=None):
            return [GanitaTuple(
                entity="lights", attribute="expense",
                value=40.0, unit="USD", time=time,
                belief_id=belief_id,
                raw_text="$40 bike lights",
                entity_aliases=("lights", "bike"),
            )]

    mem = patha.Memory(
        path=tmp_path / "store.jsonl",
        enable_phase1=False,
        karana_extractor=_BikeLightsKarana(),
    )
    # Same fact asserted in 3 different sessions
    mem.remember("I got new bike lights for $40",
                 asserted_at=datetime(2024, 1, 1), session_id="s1")
    mem.remember("Speaking of my bike, the lights I got were $40",
                 asserted_at=datetime(2024, 1, 15), session_id="s2")
    mem.remember("Just to confirm: $40 was a great deal for the bike lights",
                 asserted_at=datetime(2024, 2, 1), session_id="s3")

    # Belief store still has 3 beliefs (stub detector doesn't NLI-merge).
    # But the gaṇita index should dedup: only one $40 bike-lights tuple.
    assert len(mem._ganita_index) == 1

    rec = mem.recall("how much have I spent on bike-related expenses?")
    assert rec.ganita is not None
    assert rec.ganita.value == 40.0  # NOT $120 (over-counted 3x)


def test_proximity_fallback_catches_missed_alias(tmp_path: Path) -> None:
    """When an LLM misses tagging a fact with the expected category
    alias, but the fact's source text mentions the topic word within
    proximity_chars of the value, the proximity-fallback recovers it."""
    from patha.belief.ganita import (
        GanitaIndex, answer_aggregation_question,
    )

    idx = GanitaIndex()
    # Fact 1: LLM correctly tagged $40 with bike alias.
    idx.add(GanitaTuple(
        entity="lights", attribute="expense", value=40.0, unit="USD",
        time=None, belief_id="b-lights",
        raw_text="I got new bike lights installed for $40",
        entity_aliases=("lights", "bike", "cycling"),
    ))
    # Fact 2: LLM emitted entity='helmet' but FORGOT to add 'bike' as
    # alias. However the raw_text mentions 'bike shop' near $120.
    idx.add(GanitaTuple(
        entity="helmet", attribute="expense", value=120.0, unit="USD",
        time=None, belief_id="b-helmet",
        raw_text="bought a Bell Zephyr helmet at the bike shop for $120",
        entity_aliases=("helmet", "safety"),  # NO 'bike'!
    ))
    # Fact 3: rent — the source text mentions 'bike path' but >60 chars
    # from $999. Should NOT be pulled by proximity fallback.
    idx.add(GanitaTuple(
        entity="rent", attribute="expense", value=999.0, unit="USD",
        time=None, belief_id="b-rent",
        raw_text=(
            "Paid $999 rent on a flat. The location is great because "
            "the building is right next to the bike path."
        ),
        entity_aliases=("rent", "housing"),
    ))

    rec = answer_aggregation_question(
        "how much have I spent on bike-related expenses?", idx,
    )
    assert rec is not None
    # $40 (alias match) + $120 (proximity match) = $160
    # rent ($999) excluded because 'bike' is too far from '$999' in its text
    assert rec.value == 160.0
    contributing = set(rec.contributing_belief_ids)
    assert contributing == {"b-lights", "b-helmet"}


def test_proximity_fallback_excludes_distant_mentions(tmp_path: Path) -> None:
    """The proximity bound rejects incidental mentions that are too
    far from the value in the source text."""
    from patha.belief.ganita import (
        GanitaIndex, answer_aggregation_question,
    )

    idx = GanitaIndex()
    # Rent fact, casual mention of bike >100 chars away from value
    idx.add(GanitaTuple(
        entity="rent", attribute="expense", value=1500.0, unit="USD",
        time=None, belief_id="b1",
        raw_text=(
            "$1500 rent paid this month. " + "..." * 30 + " bike path"
        ),
        entity_aliases=("rent", "housing"),
    ))
    rec = answer_aggregation_question(
        "how much have I spent on bike?", idx,
    )
    # No bike-aliased tuple AND no proximate bike-mention → no match
    assert rec is None or rec.value != 1500.0


def test_ambiguous_query_falls_back_to_retrieval_scope(tmp_path: Path) -> None:
    """When the query is ambiguous (matches many tuples globally),
    ``restrict_to_belief_ids`` is the right tiebreaker."""
    from patha.belief.ganita import (
        GanitaIndex, answer_aggregation_question,
    )

    idx = GanitaIndex()
    # Many "expense" tuples that all alias to "shopping" (a
    # deliberately broad query target). Use a non-stopword so the
    # hint extraction actually produces it.
    for i in range(50):
        idx.add(GanitaTuple(
            entity="shopping", attribute="expense", value=10.0 * (i + 1),
            unit="USD", time=None, belief_id=f"b-{i}",
            raw_text=f"${10*(i+1)} item-{i}",
            entity_aliases=("shopping", f"item{i}"),
        ))

    # Retrieved set scopes arithmetic to a topical cluster
    retrieved = {"b-1", "b-2", "b-3"}  # values 20, 30, 40 → sum=90
    rec = answer_aggregation_question(
        "how much shopping spending did I do?", idx,
        restrict_to_belief_ids=retrieved,
        ambiguity_threshold=30,
    )
    assert rec is not None
    # 50 candidates > ambiguity_threshold (30) → restriction kicks in
    # Sum = 20 + 30 + 40 = 90
    assert rec.value == 90.0
    assert len(rec.contributing_belief_ids) == 3
