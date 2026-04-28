"""End-to-end demo of Patha's architectural claim:

  Patha separates retrieval queries from synthesis queries.
  No mainstream AI memory system does this.

Run with:

    uv run python examples/three_innovations_demo.py

Or, if you have Ollama running and want to see karaṇa with a
real local LLM (recommended ≥14B for synthesis-heavy use):

    OLLAMA_HOST=http://localhost:11434 \\
    PATHA_KARANA_MODEL=qwen2.5:14b-instruct \\
    uv run python examples/three_innovations_demo.py --karana hybrid

What you'll see:

  Section 1: Synthesis-intent routing
            — "how much have I spent on bikes?" bypasses Phase 1
            — gaṇita queries the belief store directly
            — pure deterministic arithmetic, ZERO LLM tokens at recall

  Section 2: Hebbian retrieval (no benchmark lift, no regression,
            real for repeat-query workloads)
            — co-retrieval edges accumulate from real usage

  Section 3: Filesystem-native ingest
            — `patha import obsidian-vault` brings pre-existing
              writing into the belief store
            — frontmatter dates, wikilinks, tags all preserved

The demo creates a temp store at /tmp/patha-three-innovations-demo/
and tears it down at the end.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from textwrap import dedent

import patha
from patha.belief.karana import HybridKaranaExtractor, OllamaKaranaExtractor
from patha.importers import import_obsidian_vault


def section(title: str) -> None:
    print()
    print("═" * 70)
    print(f"  {title}")
    print("═" * 70)


def subsection(title: str) -> None:
    print()
    print(f"── {title} " + "─" * (66 - len(title)))


# ─── Section 1: Hebbian-cluster-aware retrieval ──────────────────────


def demo_hebbian(tmpdir: Path) -> None:
    section("Innovation #1 — Hebbian-cluster-aware retrieval")
    print(dedent("""
        Patha's belief layer accumulates Hebbian co-retrieval edges
        between beliefs that surface in the same query. That signal,
        recorded since v0.7, is now READ at runtime: after Phase 1
        retrieves the candidate set, we walk each seed's strongest
        Hebbian neighbors and add them to the candidates.
    """).strip())

    mem = patha.Memory(
        path=tmpdir / "hebbian.jsonl",
        enable_phase1=False,  # demo deliberately bypasses Phase 1
        hebbian_expansion=True,
        hebbian_session_seed_weight=0.0,
    )

    # Ingest two related and one unrelated belief in different sessions
    a = mem.remember("I am training for a triathlon",
                     session_id="training", asserted_at=datetime(2024, 1, 1))
    b = mem.remember("I bought a road bike in Lisbon",
                     session_id="bike-purchase",
                     asserted_at=datetime(2024, 2, 1))
    c = mem.remember("My favourite color is blue",
                     session_id="random",
                     asserted_at=datetime(2024, 3, 1))

    subsection("Step 1 — Phase 1 returns BOTH triathlon and bike together once")
    # Force-feed both belief ids into a single 'phase 1 result' so the
    # plasticity hook records co-retrieval. In real usage this is the
    # natural outcome of a query like "what's my fitness situation?".
    layer = mem._patha.belief_layer
    bid_a = a["belief_id"]
    bid_b = b["belief_id"]
    bid_c = c["belief_id"]
    layer.hebbian.record_coretrieval([bid_a, bid_b])
    print(f"   Hebbian edge weight (triathlon ↔ bike) = "
          f"{layer.hebbian.weight(bid_a, bid_b):.3f}")
    print(f"   Hebbian edge weight (triathlon ↔ blue) = "
          f"{layer.hebbian.weight(bid_a, bid_c):.3f}  (still 0)")

    subsection("Step 2 — A new query that hits ONE of them now surfaces the cluster")
    # Mock Phase 1: return ONLY the triathlon proposition.
    triathlon_pid = layer.store.get(bid_a).source_proposition_id
    mem._patha._phase1_retrieve = lambda q, k: [triathlon_pid]
    rec = mem.recall("am I training?")
    surfaced = [c["proposition"] for c in rec.current]
    print(f"   surfaced beliefs: {len(surfaced)}")
    for s in surfaced:
        print(f"     · {s}")
    if any("bike" in s.lower() for s in surfaced):
        print()
        print("   ✓ The bike belief came along via Hebbian expansion,")
        print("     even though Phase 1 only returned the triathlon one.")
    else:
        print("   (Hebbian expansion didn't fire — debug: check edge weights)")


# ─── Section 2: Vedic karaṇa LLM ingest-time extraction ──────────────


def demo_karana(tmpdir: Path, mode: str, model: str) -> None:
    section(f"Innovation #2 — Vedic karaṇa ingest-time extraction "
            f"(--karana {mode})")
    print(dedent("""
        The Vedic word *karaṇa* means ritual preparation: work in
        advance so the moment of performance can be deterministic.
        Patha applies the same idea to the gaṇita layer:

          - INGEST: a small local LLM (when --karana ollama) reads
            each new belief and emits structured tuples.
          - RECALL: pure deterministic arithmetic over the preserved
            tuple index. ZERO LLM tokens at recall.

        This is the inverse of mainstream RAG (which spends tokens
        every recall). Most users ask the same things many times;
        spending tokens once at ingest is a strict win.
    """).strip())

    karana = None
    if mode == "ollama":
        karana = OllamaKaranaExtractor(model=model, timeout_s=60.0)
        print()
        print(f"   Using Ollama (free-form extraction) at {karana.host}, model={karana.model}")
    elif mode == "hybrid":
        karana = HybridKaranaExtractor(model=model, timeout_s=90.0)
        print()
        print(f"   Using HYBRID (regex × LLM tagging) at {karana.host}, model={karana.model}")
        print(f"   Recall preserved: regex catches every $X; LLM only labels semantically.")

    mem = patha.Memory(
        path=tmpdir / "karana.jsonl",
        enable_phase1=False,
        karana_extractor=karana,
    )

    subsection("Step 1 — Ingest 4 bike-related expense statements")
    facts = [
        "I bought a $50 saddle for my bike",
        "I got a $75 helmet for the bike",
        "$30 for new bike lights",
        "I spent $30 on bike gloves",
    ]
    for f in facts:
        mem.remember(f, session_id="bike-shopping",
                     asserted_at=datetime(2024, 1, 1))
        print(f"   · {f}")

    print()
    print(f"   gaṇita index size: {len(mem._ganita_index)} tuples extracted")
    if karana is not None:
        print(f"   ingest-time LLM calls: {karana.calls} "
              f"({karana.total_latency_s:.1f}s total)")

    subsection("Step 2 — Aggregation question, NO LLM at recall")
    rec = mem.recall("how much total did I spend on bike-related expenses?")
    if rec.ganita is not None:
        print(f"   answer:           ${rec.ganita.value:.2f} {rec.ganita.unit}")
        print(f"   contributing ids: "
              f"{len(rec.ganita.contributing_belief_ids)} beliefs")
        print(f"   explanation:      {rec.ganita.explanation}")
        if karana is not None and karana.calls > 0:
            print(f"   recall-time LLM calls: 0 (deterministic arithmetic)")
    else:
        print("   gaṇita didn't trigger — extractor produced no relevant tuples.")
        print("   With --karana regex this happens on dense conversational text;")
        print("   re-run with --karana ollama for the LLM-quality extraction.")


# ─── Section 3: Filesystem-native ingest ─────────────────────────────


def demo_obsidian(tmpdir: Path) -> None:
    section("Innovation #3 — Obsidian / folder-watcher import")
    print(dedent("""
        Many users have years of writing in Obsidian vaults, plain
        Markdown folders, or .txt files. Patha's `patha import` walks
        them and creates beliefs. Frontmatter `date` becomes
        `asserted_at`. Wikilinks and #tags become entity hints.
    """).strip())

    vault = tmpdir / "myvault"
    vault.mkdir(exist_ok=True)
    (vault / "journal").mkdir(exist_ok=True)

    (vault / "journal" / "2024-01-15.md").write_text(dedent("""\
        ---
        date: 2024-01-15
        tags: cycling, lisbon
        ---
        # Riding around Lisbon

        Bought a $50 saddle for the [[bike]] today. Tomorrow I'll get
        a helmet too.

        #cycling
    """))

    (vault / "journal" / "2024-02-01.md").write_text(dedent("""\
        ---
        date: 2024-02-01
        ---
        Got the [[helmet]] for $75. Very happy with it.
    """))

    (vault / ".obsidian").mkdir(exist_ok=True)
    (vault / ".obsidian" / "config").write_text(
        "this file should be skipped"
    )

    subsection(f"Step 1 — point Patha at {vault}")
    mem = patha.Memory(
        path=tmpdir / "obsidian.jsonl",
        enable_phase1=False,
    )
    stats = import_obsidian_vault(vault, mem)
    print(f"   files seen:      {stats.files_seen}")
    print(f"   files imported:  {stats.files_imported}")
    print(f"   beliefs added:   {stats.beliefs_added}")
    print(f"   files skipped:   {stats.files_skipped}")

    subsection("Step 2 — beliefs are now searchable")
    rec = mem.recall("what bike gear did I buy?")
    for c in rec.current:
        print(f"   · {c['asserted_at'][:10]}: {c['proposition'][:60]}…")


# ─── Main ────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--karana", choices=["regex", "ollama", "hybrid"], default="regex",
        help="Karaṇa extractor: regex (zero-deps baseline), ollama "
             "(LLM extracts everything), or hybrid (regex finds every "
             "$X amount, LLM only labels — best recall on dense text). "
             "Both LLM modes require Ollama running with a model pulled.",
    )
    ap.add_argument(
        "--karana-model", default="qwen2.5:7b-instruct",
        help="Model tag when --karana ollama. Default qwen2.5:7b-instruct.",
    )
    args = ap.parse_args()

    tmpdir = Path(tempfile.mkdtemp(prefix="patha-three-innovations-"))
    try:
        demo_hebbian(tmpdir)
        demo_karana(tmpdir, args.karana, args.karana_model)
        demo_obsidian(tmpdir)
        print()
        print("═" * 70)
        print("  All three innovations demonstrated. Temp dir:", tmpdir)
        print("═" * 70)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
