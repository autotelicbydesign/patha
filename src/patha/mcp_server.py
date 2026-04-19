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


# ─── Lazy-built singleton integrated instance ───────────────────────
# Built on first tool call so the server can start (and Claude Desktop
# can see the tools) before the NLI model downloads.

_patha: IntegratedPatha | None = None


def _get_patha() -> IntegratedPatha:
    global _patha
    if _patha is None:
        DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
        store_path = DEFAULT_DATA_DIR / "beliefs.jsonl"
        layer = BeliefLayer(
            store=BeliefStore(persistence_path=store_path),
            detector=make_detector(DEFAULT_DETECTOR),
        )
        answerer = DirectAnswerer(layer.store)
        _patha = IntegratedPatha(
            phase1_retrieve=None,
            belief_layer=layer,
            direct_answerer=answerer,
        )
    return _patha


# ─── MCP server ─────────────────────────────────────────────────────

mcp = FastMCP(
    name="patha",
    instructions=(
        "Patha is a local-first AI memory system with contradiction "
        "detection and non-destructive supersession. Use `patha_ingest` "
        "to remember things the user tells you about themselves, their "
        "preferences, commitments, or context. Use `patha_query` to ask "
        "what the user currently believes about a topic (pass "
        "`include_history=true` to also see what they used to believe). "
        "Use `patha_history` to find every mention of a term. Use "
        "`patha_stats` to see how much Patha has remembered."
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
) -> dict[str, Any]:
    """Ask what the user currently believes about a topic.

    Args:
        question: the query text.
        at_time: ISO timestamp to query at (defaults to now).
        include_history: if true, also return superseded beliefs.

    Returns:
        {
          "strategy": "direct_answer" | "structured" | "raw",
          "answer": "..." | null,
          "summary": "...",
          "current": [{"id", "proposition", "asserted_at"}, ...],
          "history": [{"id", "proposition", "asserted_at"}, ...] if include_history,
          "tokens_in_summary": int
        }
    """
    patha = _get_patha()
    at = datetime.fromisoformat(at_time) if at_time else datetime.now()

    response = patha.query(
        question, at_time=at, include_history=include_history,
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

    return {
        "strategy": response.strategy,
        "answer": response.answer or None,
        "summary": response.prompt,
        "current": current,
        "history": history,
        "tokens_in_summary": response.tokens_in,
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
    return {
        "data_dir": str(DEFAULT_DATA_DIR),
        "detector": DEFAULT_DETECTOR,
        "total_beliefs": len(store),
        "current": len(store.current()),
        "superseded": len(store.superseded()),
        "disputed": len(store.disputed()),
        "archived": len(store.archived()),
        "hebbian_edges": len(hebbian_weights),
        "ingest_tick": getattr(layer, "_ingest_tick", 0),
    }


# ─── Entry point ────────────────────────────────────────────────────

def main() -> int:
    """Run the MCP server on stdio. Called by the `patha-mcp` script."""
    mcp.run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
