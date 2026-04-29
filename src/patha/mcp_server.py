"""Patha MCP server — stdio transport, four tools, persistent belief store.

Plugs Patha into Claude Desktop / Claude Code / Cursor / Zed / Goose.
Any MCP-compatible client can add one config entry and get persistent,
contradicting, evolving memory across sessions.

Storage:
    ~/.patha/beliefs.jsonl        (event log, append-only)
    Override with PATHA_STORE_PATH env var.

Detector:
    Defaults to 'stub' for instant startup (no model downloads).
    Set PATHA_DETECTOR=full-stack-v7 for the production detector
    (NLI + adhyāsa + numerical + sequential). Triggers a one-time
    ~1.7 GB model download on first use.

Tools exposed to the MCP client:
    patha_ingest    — remember a new proposition, resolves contradictions
    patha_query     — ask what the user currently believes about X
    patha_history   — every belief (current + superseded) mentioning a term
    patha_stats     — store counts + plasticity state

Usage from a Claude Desktop config:

    {
      "mcpServers": {
        "patha": {
          "command": "uvx",
          "args": ["patha-mcp"],
          "env": { "PATHA_DETECTOR": "full-stack-v7" }
        }
      }
    }

Or using a local checkout:

    {
      "mcpServers": {
        "patha": {
          "command": "uv",
          "args": ["run", "--project", "/path/to/Memory", "patha-mcp"]
        }
      }
    }
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from patha.belief import (
    BeliefLayer,
    BeliefStore,
    DirectAnswerer,
    make_detector,
)
from patha.integrated import IntegratedPatha


# ─── Configuration ──────────────────────────────────────────────────

DEFAULT_DATA_DIR = Path(
    os.environ.get("PATHA_STORE_PATH", str(Path.home() / ".patha"))
)
DEFAULT_DETECTOR = os.environ.get("PATHA_DETECTOR", "stub")
# Semantic pre-filter: when set, `patha_query` narrows the belief
# store to the top-K topically relevant beliefs before running
# supersession and summary. Cuts false-positive supersessions on
# large stores. Disable with PATHA_SEMANTIC_FILTER=off.
DEFAULT_SEMANTIC_FILTER_K = int(os.environ.get("PATHA_SEMANTIC_FILTER_K", "40"))
SEMANTIC_FILTER_ENABLED = (
    os.environ.get("PATHA_SEMANTIC_FILTER", "on").lower() != "off"
)
# Phase 1 — the 7-view Vedic retrieval pillar. On by default because
# it IS Patha; running only the semantic filter is a watered-down
# version. Disable with PATHA_PHASE1=off if you specifically want
# cosine-only behavior (benchmarking, offline profiling).
#
# Indexes are built lazily on first query, not at MCP startup — so
# Claude Desktop's server-start stays instant. The first query after
# a Claude Desktop restart pays a one-time indexing cost (~3s per
# 100 beliefs on CPU).
PHASE1_ENABLED = os.environ.get("PATHA_PHASE1", "on").lower() != "off"

# Hebbian-cluster-aware retrieval expands the Phase-1 candidate set
# via the accumulated co-retrieval graph. Targets multi-session
# retrieval: beliefs that have surfaced together repeatedly co-surface
# again. PATHA_HEBBIAN=off to disable for ablations.
HEBBIAN_ENABLED = os.environ.get("PATHA_HEBBIAN", "on").lower() != "off"

# Gaṇita layer — sidecar JSONL index of (entity, attribute, value, unit)
# tuples extracted at ingest. Recall-time aggregation is procedural
# (no LLM tokens). PATHA_GANITA=off to disable.
GANITA_ENABLED = os.environ.get("PATHA_GANITA", "on").lower() != "off"

# Karaṇa extractor — choose what reads each ingested belief and emits
# tuples for the gaṇita index:
#   "regex" (default) — zero-dependency baseline; works on toy facts
#   "ollama"          — local LLM (qwen2.5:7b-instruct) via Ollama;
#                       handles dense conversational text. Requires
#                       `ollama serve` + `ollama pull qwen2.5:7b-instruct`.
#   "off"             — skip extraction entirely (still indexes beliefs).
KARANA_MODE = os.environ.get("PATHA_KARANA", "regex").lower()


# ─── Lazy-built singleton integrated instance ───────────────────────
# Built on first tool call so the server can start (and Claude Desktop
# can see the tools) before the NLI model downloads.

_patha: IntegratedPatha | None = None
_semantic_filter = None  # lazy-built on first filtered query
_phase1_retriever = None  # LazyPhase1Retriever — need handle for invalidate()
_ganita_index = None  # GanitaIndex; sidecar JSONL next to beliefs.jsonl
_karana_extractor = None  # KaranaExtractor used at ingest


def _build_karana_extractor():
    """Build the configured karaṇa extractor. None if PATHA_KARANA=off."""
    if KARANA_MODE == "off":
        return None
    if KARANA_MODE == "ollama":
        from patha.belief.karana import OllamaKaranaExtractor
        return OllamaKaranaExtractor()
    # default = regex baseline
    from patha.belief.karana import RegexKaranaExtractor
    return RegexKaranaExtractor()


def _get_patha() -> IntegratedPatha:
    global _patha, _phase1_retriever, _ganita_index, _karana_extractor
    if _patha is None:
        DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
        store_path = DEFAULT_DATA_DIR / "beliefs.jsonl"
        belief_store = BeliefStore(persistence_path=store_path)
        layer = BeliefLayer(
            store=belief_store,
            detector=make_detector(DEFAULT_DETECTOR),
        )
        answerer = DirectAnswerer(layer.store)

        phase1_retrieve = None
        if PHASE1_ENABLED:
            try:
                from patha.phase1_bridge import LazyPhase1Retriever
                _phase1_retriever = LazyPhase1Retriever(belief_store)
                phase1_retrieve = _phase1_retriever
            except Exception as e:
                print(
                    f"[patha-mcp] warn: PATHA_PHASE1 enabled but bridge "
                    f"failed to initialize ({e}); falling back to semantic "
                    f"filter.",
                    file=sys.stderr,
                )

        # Gaṇita sidecar index — aggregation questions answer with no LLM.
        if GANITA_ENABLED:
            try:
                from patha.belief.ganita import GanitaIndex
                ganita_path = store_path.parent / (
                    store_path.stem + ".ganita.jsonl"
                )
                _ganita_index = GanitaIndex(persistence_path=ganita_path)
                _karana_extractor = _build_karana_extractor()
            except Exception as e:
                print(
                    f"[patha-mcp] warn: gaṇita layer disabled ({e}).",
                    file=sys.stderr,
                )
                _ganita_index = None
                _karana_extractor = None

        _patha = IntegratedPatha(
            phase1_retrieve=phase1_retrieve,
            belief_layer=layer,
            direct_answerer=answerer,
            hebbian_expansion=HEBBIAN_ENABLED,
        )
    return _patha


def _invalidate_phase1() -> None:
    """Mark Phase 1 indexes dirty so the next query rebuilds them.
    Called after every successful ingest so newly-added beliefs are
    findable. Cheap — just flips a flag."""
    global _phase1_retriever
    if _phase1_retriever is not None:
        _phase1_retriever.invalidate()


def _get_semantic_filter():
    """Lazy-load the semantic filter only when a filtered query is served.
    Avoids dragging MiniLM weights in for MCP clients that only ingest."""
    global _semantic_filter
    if _semantic_filter is None:
        from patha.belief.semantic_filter import SemanticBeliefFilter
        _semantic_filter = SemanticBeliefFilter()
    return _semantic_filter


# ─── MCP server ─────────────────────────────────────────────────────

mcp = FastMCP(
    name="patha",
    instructions=(
        "Patha is the user's persistent, local-first memory. Use it "
        "proactively, not just when asked.\n\n"
        "USE RULES (follow these every conversation):\n"
        "1. At the start of every new conversation, read the "
        "`patha://beliefs/current` resource OR call `patha_query` with "
        "the topic of the conversation so you have the user's memory in "
        "context before responding.\n"
        "2. Whenever the user states a FACT about themselves, their "
        "life, work, preferences, relationships, projects, commitments, "
        "deadlines, decisions, opinions, or context — call "
        "`patha_ingest` with that fact. Don't ask permission; the tool "
        "is idempotent and handles duplicates. Missed facts are a bug; "
        "over-capturing is fine.\n"
        "3. When the user contradicts something they said earlier (new "
        "job, moved, changed their mind), still call `patha_ingest` — "
        "Patha's belief layer handles supersession correctly and "
        "preserves the old belief as history.\n"
        "4. When the user asks ANY question about their own life, "
        "preferences, past decisions, or context, call `patha_query` "
        "first. Do not answer from conversation context alone; the user "
        "has explicitly installed Patha to be the source of truth about "
        "themselves.\n"
        "5. Use `patha_history` to show the user how a belief evolved "
        "over time (old vs current). Use `patha_stats` for storage "
        "overview.\n\n"
        "Patha stores beliefs in a plain JSONL file at ~/.patha/ the "
        "user owns. Everything is local, nothing is sent to servers. "
        "You do NOT need to warn the user each time you call a tool; "
        "just do it."
    ),
)


@mcp.tool()
def patha_ingest(
    proposition: str,
    asserted_at: str | None = None,
    session_id: str | None = None,
    source_id: str | None = None,
    context: str | None = None,
) -> dict[str, Any]:
    """Remember a single proposition. Routes through the full belief
    pipeline (contradiction detection, supersession, plasticity).

    Args:
        proposition: the claim to remember, in natural language.
        asserted_at: ISO timestamp (defaults to now).
        session_id: session/bucket id (defaults to today's date).
        source_id: optional upstream identifier for traceability.
        context: optional context tag ("work", "health", ...) for
            context-scoped queries.

    Returns:
        {
          "action": "added" | "reinforced" | "superseded",
          "belief_id": "...",
          "affected_belief_ids": [...],
          "contradictions_detected": int
        }
    """
    patha = _get_patha()
    at = datetime.fromisoformat(asserted_at) if asserted_at else datetime.now()
    session = session_id or at.strftime("%Y-%m-%d")
    src = source_id or f"mcp-{session}-{int(at.timestamp())}"

    ev = patha.ingest(
        proposition=proposition,
        asserted_at=at,
        asserted_in_session=session,
        source_proposition_id=src,
        context=context,
    )
    # Mark the Phase 1 index dirty so the next query rebuilds with
    # the new belief included. Cheap flag-flip.
    _invalidate_phase1()

    # Karaṇa pass: extract numerical tuples for the gaṇita index.
    # Recall-time aggregation questions (sum, count, average) answer
    # over this index with zero LLM tokens.
    if _ganita_index is not None and _karana_extractor is not None:
        try:
            tuples = _karana_extractor.extract(
                proposition,
                belief_id=ev.new_belief.id,
                time=at.isoformat(),
            )
            _ganita_index.add_many(tuples)
        except Exception as e:
            # Never fail an ingest because the karaṇa layer hiccupped.
            print(f"[patha-mcp] karaṇa extract warning: {e}", file=sys.stderr)

    return {
        "action": ev.action,
        "belief_id": ev.new_belief.id,
        "proposition": ev.new_belief.proposition,
        "affected_belief_ids": list(ev.affected_belief_ids),
        "contradictions_detected": ev.contradictions_detected,
    }


@mcp.tool()
def patha_query(
    question: str,
    at_time: str | None = None,
    include_history: bool = False,
    semantic_filter: bool | None = None,
) -> dict[str, Any]:
    """Ask what the user currently believes about a topic.

    Args:
        question: the query text.
        at_time: ISO timestamp to query at (defaults to now).
        include_history: if true, also return superseded beliefs.
        semantic_filter: if true (default: follows PATHA_SEMANTIC_FILTER env
            var, normally on), the belief store is narrowed to the top-K
            topically relevant beliefs before supersession and summary.
            This prevents unrelated-topic beliefs from diluting the answer
            and cuts false-positive supersessions on large stores.

    Returns:
        {
          "strategy": "direct_answer" | "structured" | "raw",
          "answer": "..." | null,
          "summary": "...",
          "current": [{"id", "proposition", "asserted_at"}, ...],
          "history": [{"id", "proposition", "asserted_at"}, ...] if include_history,
          "tokens_in_summary": int,
          "semantic_filter_applied": bool,
          "semantic_filter_kept": int  # how many beliefs survived the filter
        }
    """
    patha = _get_patha()
    at = datetime.fromisoformat(at_time) if at_time else datetime.now()

    use_filter = (
        semantic_filter if semantic_filter is not None else SEMANTIC_FILTER_ENABLED
    )
    candidate_belief_ids = None
    filter_kept = 0
    if use_filter:
        all_beliefs = list(patha.belief_layer.store.all())
        if all_beliefs:
            sfilter = _get_semantic_filter()
            candidate_belief_ids = sfilter.top_k(
                query=question,
                beliefs=all_beliefs,
                k=DEFAULT_SEMANTIC_FILTER_K,
            )
            filter_kept = len(candidate_belief_ids)

    response = patha.query(
        question,
        at_time=at,
        include_history=include_history,
        candidate_belief_ids=candidate_belief_ids,
    )

    qr = response.retrieval_result
    current = [
        {
            "id": b.id,
            "proposition": b.proposition,
            "asserted_at": b.asserted_at.isoformat(),
            "confidence": b.confidence,
        }
        for b in (qr.current if qr else [])
    ]
    history = [
        {
            "id": b.id,
            "proposition": b.proposition,
            "asserted_at": b.asserted_at.isoformat(),
            "confidence": b.confidence,
        }
        for b in (qr.history if qr and include_history else [])
    ]

    # Gaṇita pass — try procedural arithmetic for aggregation questions
    # ("how much total", "how many"). Returns a structured result so the
    # client can show the deterministic answer alongside the LLM-bound
    # summary. Restricted to the retrieved beliefs to avoid summing
    # unrelated currency mentions across the whole store.
    ganita_block = None
    if _ganita_index is not None:
        try:
            from patha.belief.ganita import answer_aggregation_question
            retrieved_ids = None
            qr_local = response.retrieval_result
            if qr_local is not None:
                retrieved_ids = {b.id for b in qr_local.current}
                if include_history:
                    retrieved_ids |= {b.id for b in qr_local.history}
            gres = answer_aggregation_question(
                question, _ganita_index,
                restrict_to_belief_ids=retrieved_ids,
            )
            if gres is not None:
                ganita_block = {
                    "operator": gres.operator,
                    "value": gres.value,
                    "unit": gres.unit,
                    "explanation": gres.explanation,
                    "contributing_belief_ids": list(
                        gres.contributing_belief_ids
                    ),
                }
        except Exception:
            ganita_block = None

    return {
        "strategy": response.strategy,
        "answer": response.answer or None,
        "summary": response.prompt,
        "current": current,
        "history": history,
        "tokens_in_summary": response.tokens_in,
        "semantic_filter_applied": use_filter,
        "semantic_filter_kept": filter_kept,
        "ganita": ganita_block,
    }


@mcp.tool()
def patha_history(term: str) -> dict[str, Any]:
    """Find every belief (current OR superseded) mentioning ``term``.

    Args:
        term: substring to match (case-insensitive).

    Returns:
        {
          "term": "...",
          "matches": [
            {"id", "proposition", "status", "asserted_at", "confidence"},
            ...
          ]
        }
    """
    patha = _get_patha()
    store = patha.belief_layer.store
    needle = term.lower()
    matching = [b for b in store.all() if needle in b.proposition.lower()]
    matching.sort(key=lambda b: b.asserted_at)
    return {
        "term": term,
        "matches": [
            {
                "id": b.id,
                "proposition": b.proposition,
                "status": b.status.value,
                "asserted_at": b.asserted_at.isoformat(),
                "confidence": b.confidence,
            }
            for b in matching
        ],
    }


@mcp.tool()
def patha_stats() -> dict[str, Any]:
    """Return belief-store statistics: counts, plasticity state, data path."""
    patha = _get_patha()
    store = patha.belief_layer.store
    layer = patha.belief_layer
    hebbian_weights = getattr(layer.hebbian, "_weights", {})
    ganita_n = len(_ganita_index) if _ganita_index is not None else 0
    return {
        "data_dir": str(DEFAULT_DATA_DIR),
        "detector": DEFAULT_DETECTOR,
        "phase1_enabled": PHASE1_ENABLED,
        "hebbian_expansion": HEBBIAN_ENABLED,
        "karana_mode": KARANA_MODE,
        "total_beliefs": len(store),
        "current": len(store.current()),
        "superseded": len(store.superseded()),
        "disputed": len(store.disputed()),
        "archived": len(store.archived()),
        "hebbian_edges": len(hebbian_weights),
        "ganita_tuples": ganita_n,
        "ingest_tick": getattr(layer, "_ingest_tick", 0),
    }


# ─── Resources ──────────────────────────────────────────────────────
# Resources are addressable data Claude (or any MCP client) can read
# without a tool call. Exposing the current belief summary here means
# a client can surface memory at conversation start with no roundtrip.

@mcp.resource("patha://beliefs/current")
def _beliefs_current() -> str:
    """Current belief summary (all non-superseded beliefs)."""
    patha = _get_patha()
    store = patha.belief_layer.store
    current = list(store.current())
    if not current:
        return (
            "No beliefs yet. Patha is installed and ready; ingest facts "
            "as the user tells you about themselves."
        )
    lines = [
        f"Patha — user's current beliefs ({len(current)} total). "
        f"Stored locally at ~/.patha/beliefs.jsonl.",
        "",
    ]
    for b in current:
        date = b.asserted_at.strftime("%Y-%m-%d") if b.asserted_at else "-"
        lines.append(f"- [{date}] {b.proposition}")
    return "\n".join(lines)


@mcp.resource("patha://beliefs/all")
def _beliefs_all() -> str:
    """All beliefs including superseded ones, sorted newest-first."""
    patha = _get_patha()
    store = patha.belief_layer.store
    current = list(store.current())
    superseded = list(store.superseded())
    if not current and not superseded:
        return "No beliefs yet."
    lines = [f"Patha — all beliefs ({len(current)} current, "
             f"{len(superseded)} superseded).", ""]
    lines.append("## Current")
    for b in current:
        date = b.asserted_at.strftime("%Y-%m-%d") if b.asserted_at else "-"
        lines.append(f"- [{date}] {b.proposition}")
    if superseded:
        lines.append("")
        lines.append("## Superseded (no longer current)")
        for b in sorted(superseded, key=lambda x: x.asserted_at, reverse=True):
            date = b.asserted_at.strftime("%Y-%m-%d") if b.asserted_at else "-"
            lines.append(f"- [{date}] ~~{b.proposition}~~")
    return "\n".join(lines)


@mcp.resource("patha://stats")
def _resource_stats() -> str:
    """Human-readable belief store stats."""
    patha = _get_patha()
    layer = patha.belief_layer
    store = layer.store
    return (
        f"Patha belief store:\n"
        f"  {len(store)} total beliefs\n"
        f"  {len(store.current())} current\n"
        f"  {len(store.superseded())} superseded\n"
        f"  Detector: {DEFAULT_DETECTOR}\n"
        f"  Location: {DEFAULT_DATA_DIR}/beliefs.jsonl\n"
        f"  Phase 1 retrieval: "
        f"{'on' if PHASE1_ENABLED else 'off'}\n"
    )


# ─── Entry point ────────────────────────────────────────────────────

def main() -> int:
    """Run the MCP server on stdio. Called by the `patha-mcp` script."""
    mcp.run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
