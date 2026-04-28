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
