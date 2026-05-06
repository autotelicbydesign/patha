"""Patha MCP server — stdio transport, four tools, persistent belief store.

Plugs Patha into Claude Desktop / Claude Code / Cursor / Zed / Goose.
Any MCP-compatible client can add one config entry and get persistent,
contradicting, evolving memory across sessions.

Storage:
    ~/.patha/beliefs.jsonl        (event log, append-only)
    Override with PATHA_STORE_PATH env var.

Detector:
    Defaults to 'stub' for instant startup (no model downloads).
    Set PATHA_DETECTOR=full-stack-v8 for the production detector
    (NLI + adhyāsa + numerical + sequential + learned-classifier).
    Triggers a one-time ~1.7 GB model download on first use.

Tools exposed to the MCP client:
    patha_ingest    — remember a new proposition, resolves contradictions
    patha_query     — ask what the user currently believes about X
    patha_history   — every belief (current + superseded) mentioning a term
    patha_stats     — store counts + plasticity state

All tool errors are caught and returned as structured `{"error": "..."}`
responses; we never let an exception kill the JSON-RPC frame and trigger
the dreaded "Server disconnected" in Claude Desktop.

Usage from a Claude Desktop config:

    {
      "mcpServers": {
        "patha": {
          "command": "uvx",
          "args": ["patha-mcp"],
          "env": { "PATHA_DETECTOR": "full-stack-v8" }
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

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field

from patha.belief import (
    BeliefLayer,
    BeliefStore,
    DirectAnswerer,
    make_detector,
)
from patha.belief.ganita import GanitaIndex, answer_aggregation_question
from patha.integrated import IntegratedPatha


# ─── Configuration ──────────────────────────────────────────────────

DEFAULT_DATA_DIR = Path(
    os.environ.get("PATHA_STORE_PATH", str(Path.home() / ".patha"))
)
DEFAULT_DETECTOR = os.environ.get("PATHA_DETECTOR", "stub")
DEFAULT_SEMANTIC_FILTER_K = int(os.environ.get("PATHA_SEMANTIC_FILTER_K", "40"))
SEMANTIC_FILTER_ENABLED = (
    os.environ.get("PATHA_SEMANTIC_FILTER", "on").lower() != "off"
)
PHASE1_ENABLED = os.environ.get("PATHA_PHASE1", "on").lower() != "off"
HEBBIAN_ENABLED = os.environ.get("PATHA_HEBBIAN", "on").lower() != "off"
GANITA_ENABLED = os.environ.get("PATHA_GANITA", "on").lower() != "off"
KARANA_MODE = os.environ.get("PATHA_KARANA", "regex").lower()

# Hard input limits — keep oversized strings out of the belief store.
# A "proposition" is meant to be a single fact, not a whole document;
# anything over 4 KB is almost certainly the model dumping context
# rather than asserting one belief. The query/term limits guard against
# unbounded substring scans.
MAX_PROPOSITION_CHARS = 4_000
MAX_QUESTION_CHARS = 2_000
MAX_TERM_CHARS = 200
MAX_CONTEXT_CHARS = 200
MAX_HISTORY_LIMIT = 200


# ─── Lazy-built singleton integrated instance ───────────────────────
# Built on first tool call so the server can start (and Claude Desktop
# can see the tools) before the detector loads model weights.

_patha: IntegratedPatha | None = None
_semantic_filter = None
_phase1_retriever = None
_ganita_index: GanitaIndex | None = None
_karana_extractor = None


def _build_karana_extractor():
    """Build the configured karaṇa extractor. None if PATHA_KARANA=off."""
    if KARANA_MODE == "off":
        return None
    if KARANA_MODE == "ollama":
        from patha.belief.karana import OllamaKaranaExtractor
        return OllamaKaranaExtractor()
    from patha.belief.karana import RegexKaranaExtractor
    return RegexKaranaExtractor()


def _get_patha() -> IntegratedPatha:
    """Lazy-build the IntegratedPatha singleton on first use."""
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
                    f"failed to initialize ({type(e).__name__}: {e}); "
                    f"falling back to semantic filter.",
                    file=sys.stderr,
                )

        if GANITA_ENABLED:
            try:
                ganita_path = store_path.parent / (
                    store_path.stem + ".ganita.jsonl"
                )
                _ganita_index = GanitaIndex(persistence_path=ganita_path)
                _karana_extractor = _build_karana_extractor()
            except Exception as e:
                print(
                    f"[patha-mcp] warn: gaṇita layer disabled "
                    f"({type(e).__name__}: {e}).",
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
    findable on the next query. Cheap — just flips a flag.
    """
    if _phase1_retriever is not None:
        _phase1_retriever.invalidate()


def _get_semantic_filter():
    """Lazy-load the semantic filter only when a filtered query runs.

    Avoids dragging MiniLM weights in for clients that only ingest.
    """
    global _semantic_filter
    if _semantic_filter is None:
        from patha.belief.semantic_filter import SemanticBeliefFilter
        _semantic_filter = SemanticBeliefFilter()
    return _semantic_filter


# ─── Helpers ────────────────────────────────────────────────────────


def _parse_iso_or_none(value: Optional[str]) -> Optional[datetime]:
    """Best-effort ISO-8601 parse. Returns None if value is None or blank.

    Raises ValueError with a clear message on malformed input so the
    caller can return a structured error to the client (and the JSON-RPC
    frame stays alive).
    """
    if value is None or not value.strip():
        return None
    s = value.strip()
    # Tolerate trailing 'Z' (UTC) which Python <3.11 doesn't accept.
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"Invalid ISO-8601 timestamp: {value!r} "
            f"(expected e.g. '2026-05-06T14:30:00' or "
            f"'2026-05-06T14:30:00+00:00'). {e}"
        ) from e


def _structured_error(
    code: str,
    message: str,
    *,
    details: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Return a structured error response.

    We never let an exception bubble out of a tool — that crashes the
    stdio JSON-RPC frame and the client shows "Server disconnected".
    Every tool catches exceptions and returns one of these instead.
    """
    out = {"error": {"code": code, "message": message}}
    if details:
        out["error"]["details"] = details
    return out


# ─── Pydantic input models ──────────────────────────────────────────
# Pydantic validates inputs (length, non-empty, ISO-shape) BEFORE the
# tool body runs, so malformed inputs return clear validation errors
# instead of mid-tool crashes.


class IngestInput(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True, validate_assignment=True, extra="forbid",
    )
    proposition: str = Field(
        ...,
        min_length=1,
        max_length=MAX_PROPOSITION_CHARS,
        description=(
            "The fact to remember, as a single declarative sentence in "
            "natural language. Examples: 'I am vegetarian', 'I bought a "
            "$50 saddle for my bike', 'I just moved to Melbourne'. "
            "Short, atomic, present-tense ideal."
        ),
    )
    asserted_at: Optional[str] = Field(
        default=None,
        description=(
            "ISO-8601 timestamp the user asserted this belief at "
            "(defaults to now). Examples: '2026-05-06T14:30:00' or "
            "'2026-05-06T14:30:00+00:00'."
        ),
    )
    session_id: Optional[str] = Field(
        default=None, max_length=120,
        description="Session/bucket id (defaults to today's date YYYY-MM-DD).",
    )
    source_id: Optional[str] = Field(
        default=None, max_length=200,
        description="Optional upstream identifier for traceability.",
    )
    context: Optional[str] = Field(
        default=None, max_length=MAX_CONTEXT_CHARS,
        description=(
            "Optional context tag ('work', 'health', 'travel', ...) for "
            "context-scoped queries. Use sparingly — the belief layer "
            "indexes by content, not by tag."
        ),
    )


class QueryInput(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True, validate_assignment=True, extra="forbid",
    )
    question: str = Field(
        ...,
        min_length=1,
        max_length=MAX_QUESTION_CHARS,
        description=(
            "The user's question or topic, in natural language. "
            "Examples: 'what do I currently eat?', 'how much have I "
            "spent on bikes?', 'where do I live?'."
        ),
    )
    at_time: Optional[str] = Field(
        default=None,
        description=(
            "ISO-8601 timestamp to query at (defaults to now). Used to "
            "answer 'what did I believe on date X?' counterfactuals."
        ),
    )
    include_history: bool = Field(
        default=False,
        description=(
            "If true, also return superseded beliefs in the response. "
            "Default false to keep responses focused on current state."
        ),
    )
    semantic_filter: Optional[bool] = Field(
        default=None,
        description=(
            "If true (default: follows PATHA_SEMANTIC_FILTER env, on by "
            "default), narrow the belief store to top-K topically "
            "relevant beliefs before supersession. Improves answer focus "
            "on large stores; set false to query the entire store."
        ),
    )


class HistoryInput(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True, validate_assignment=True, extra="forbid",
    )
    term: str = Field(
        ...,
        min_length=1,
        max_length=MAX_TERM_CHARS,
        description=(
            "Substring to match in the belief proposition text "
            "(case-insensitive). Examples: 'sushi', 'Lisbon', 'job'."
        ),
    )
    limit: int = Field(
        default=50, ge=1, le=MAX_HISTORY_LIMIT,
        description=(
            f"Maximum results to return. Default 50; max {MAX_HISTORY_LIMIT}. "
            f"Use the `next_offset` field for pagination."
        ),
    )
    offset: int = Field(
        default=0, ge=0,
        description="Number of results to skip for pagination. Default 0.",
    )


# ─── MCP server ─────────────────────────────────────────────────────

mcp = FastMCP(
    name="patha_mcp",
    instructions=(
        "Patha is the user's persistent, local-first memory. Use it "
        "PROACTIVELY, not just when asked.\n\n"
        "USE RULES (apply every conversation):\n"
        "1. At the start of every new conversation, read the "
        "`patha://beliefs/current` resource OR call `patha_query` with "
        "the topic of the conversation so you have the user's memory in "
        "context before responding.\n"
        "2. Whenever the user states a FACT about themselves — their "
        "life, work, preferences, relationships, projects, commitments, "
        "deadlines, decisions, opinions, or context — call "
        "`patha_ingest` with that fact. Don't ask permission; the tool "
        "is idempotent and handles duplicates. Missed facts are a bug; "
        "over-capturing is fine.\n"
        "3. When the user contradicts something they said earlier (new "
        "job, moved, changed their mind), call `patha_ingest` with the "
        "new claim — Patha's belief layer detects the contradiction, "
        "supersedes the old belief, and preserves it as history.\n"
        "4. When the user asks ANY question about their own life, "
        "preferences, past decisions, or context, call `patha_query` "
        "first. Do not answer from conversation context alone; the user "
        "has explicitly installed Patha to be the source of truth about "
        "themselves.\n"
        "5. Use `patha_history` to show how a belief evolved over time. "
        "Use `patha_stats` for storage overview.\n\n"
        "Patha stores beliefs in a plain JSONL file at ~/.patha/ that "
        "the user owns. Everything is local; nothing is sent to "
        "servers. You do NOT need to warn the user each time you call "
        "a Patha tool; just do it.\n\n"
        "If a tool returns an `{\"error\": ...}` object, the server is "
        "still alive — just relay the error message to the user and "
        "continue."
    ),
)


@mcp.tool(
    name="patha_ingest",
    annotations={
        "title": "Remember a fact",
        "readOnlyHint": False,
        "destructiveHint": False,  # supersession is non-destructive
        "idempotentHint": True,    # duplicate ingests reinforce, no harm
        "openWorldHint": False,
    },
)
def patha_ingest(params: IngestInput) -> dict[str, Any]:
    """Remember a single proposition. Routes through the full belief
    pipeline (contradiction detection, supersession, plasticity, gaṇita
    tuple extraction).

    The tool is **idempotent**: ingesting the same proposition twice
    reinforces the existing belief rather than creating a duplicate.
    Contradictions trigger non-destructive supersession — the old
    belief moves to history and remains queryable via `patha_history`.

    Returns:
        On success:
        {
          "action": "added" | "reinforced" | "superseded",
          "belief_id": str,
          "proposition": str,
          "affected_belief_ids": list[str],
          "contradictions_detected": int
        }
        On failure: {"error": {"code": str, "message": str, ...}}
    """
    try:
        try:
            at = _parse_iso_or_none(params.asserted_at) or datetime.now()
        except ValueError as e:
            return _structured_error("invalid_timestamp", str(e))

        session = params.session_id or at.strftime("%Y-%m-%d")
        src = params.source_id or f"mcp-{session}-{int(at.timestamp())}"

        patha = _get_patha()
        ev = patha.ingest(
            proposition=params.proposition,
            asserted_at=at,
            asserted_in_session=session,
            source_proposition_id=src,
            context=params.context,
        )
        _invalidate_phase1()

        # Karaṇa pass: extract numerical tuples for the gaṇita index.
        # Recall-time aggregation answers over this index with no LLM tokens.
        if _ganita_index is not None and _karana_extractor is not None:
            try:
                tuples = _karana_extractor.extract(
                    params.proposition,
                    belief_id=ev.new_belief.id,
                    time=at.isoformat(),
                )
                _ganita_index.add_many(tuples)
            except Exception as e:
                # Never fail an ingest because the karaṇa layer hiccupped.
                print(
                    f"[patha-mcp] karaṇa extract warning: "
                    f"{type(e).__name__}: {e}",
                    file=sys.stderr,
                )

        return {
            "action": ev.action,
            "belief_id": ev.new_belief.id,
            "proposition": ev.new_belief.proposition,
            "affected_belief_ids": list(ev.affected_belief_ids),
            "contradictions_detected": ev.contradictions_detected,
        }
    except Exception as e:
        # Last-resort guard: any exception we didn't anticipate becomes
        # a structured error, not a JSON-RPC frame crash.
        return _structured_error(
            "ingest_failed",
            f"Could not ingest belief: {type(e).__name__}: {e}",
        )


@mcp.tool(
    name="patha_query",
    annotations={
        "title": "Ask what the user currently believes",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
def patha_query(params: QueryInput) -> dict[str, Any]:
    """Ask what the user currently believes about a topic.

    Routes through Phase 1 retrieval (Vedic 7-view + songline graph) →
    Belief Layer (current-state filter, supersession-aware) → Gaṇita
    sidecar (deterministic synthesis answers for "how much/many"
    questions, with **zero LLM tokens at recall**).

    Returns:
        On success:
        {
          "strategy": "direct_answer" | "structured" | "raw" | "ganita",
          "answer": str | null,        # natural-language direct answer if any
          "summary": str,              # the prompt to feed an LLM
          "current": [{"id", "proposition", "asserted_at", "confidence"}],
          "history": [...],            # only when include_history=True
          "ganita": {                  # only on synthesis questions
              "operator", "value", "unit", "explanation",
              "contributing_belief_ids"
          } | null,
          "tokens_in_summary": int,
          "semantic_filter_applied": bool,
          "semantic_filter_kept": int
        }
        On failure: {"error": {"code": str, "message": str, ...}}
    """
    try:
        try:
            at = _parse_iso_or_none(params.at_time) or datetime.now()
        except ValueError as e:
            return _structured_error("invalid_timestamp", str(e))

        patha = _get_patha()

        use_filter = (
            params.semantic_filter
            if params.semantic_filter is not None
            else SEMANTIC_FILTER_ENABLED
        )
        candidate_belief_ids = None
        filter_kept = 0
        if use_filter:
            all_beliefs = list(patha.belief_layer.store.all())
            if all_beliefs:
                sfilter = _get_semantic_filter()
                candidate_belief_ids = sfilter.top_k(
                    query=params.question,
                    beliefs=all_beliefs,
                    k=DEFAULT_SEMANTIC_FILTER_K,
                )
                filter_kept = len(candidate_belief_ids)

        response = patha.query(
            params.question,
            at_time=at,
            include_history=params.include_history,
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
            for b in (qr.history if qr and params.include_history else [])
        ]

        ganita_block = None
        if _ganita_index is not None:
            try:
                retrieved_ids = None
                if qr is not None:
                    retrieved_ids = {b.id for b in qr.current}
                    if params.include_history:
                        retrieved_ids |= {b.id for b in qr.history}
                gres = answer_aggregation_question(
                    params.question, _ganita_index,
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
            except Exception as e:
                # Gaṇita is a bonus path; if it fails, return the
                # belief-layer answer without it. Don't error the tool.
                print(
                    f"[patha-mcp] gaṇita query warning: "
                    f"{type(e).__name__}: {e}",
                    file=sys.stderr,
                )
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
    except Exception as e:
        return _structured_error(
            "query_failed",
            f"Could not query: {type(e).__name__}: {e}",
        )


@mcp.tool(
    name="patha_history",
    annotations={
        "title": "List beliefs mentioning a term (current + superseded)",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
def patha_history(params: HistoryInput) -> dict[str, Any]:
    """Find every belief (current OR superseded) whose proposition
    contains ``term`` (case-insensitive).

    Returns paginated results so a search across a large store doesn't
    flood the response.

    Returns:
        On success:
        {
          "term": str,
          "total": int,           # total matches across the whole store
          "count": int,           # number returned in this page
          "offset": int,
          "next_offset": int | null,   # null = no more pages
          "has_more": bool,
          "matches": [
            {"id", "proposition", "status", "asserted_at", "confidence"},
            ...
          ]
        }
        On failure: {"error": {...}}
    """
    try:
        patha = _get_patha()
        store = patha.belief_layer.store
        needle = params.term.lower()
        all_matches = [
            b for b in store.all() if needle in b.proposition.lower()
        ]
        all_matches.sort(key=lambda b: b.asserted_at)

        total = len(all_matches)
        page = all_matches[params.offset : params.offset + params.limit]
        end = params.offset + len(page)
        has_more = end < total

        return {
            "term": params.term,
            "total": total,
            "count": len(page),
            "offset": params.offset,
            "next_offset": end if has_more else None,
            "has_more": has_more,
            "matches": [
                {
                    "id": b.id,
                    "proposition": b.proposition,
                    "status": b.status.value,
                    "asserted_at": b.asserted_at.isoformat(),
                    "confidence": b.confidence,
                }
                for b in page
            ],
        }
    except Exception as e:
        return _structured_error(
            "history_failed",
            f"Could not search history: {type(e).__name__}: {e}",
        )


@mcp.tool(
    name="patha_stats",
    annotations={
        "title": "Belief-store stats: counts, plasticity state, data path",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
def patha_stats() -> dict[str, Any]:
    """Return belief-store statistics: counts, plasticity state, data path.

    Returns:
        On success:
        {
          "data_dir": str,
          "detector": str,
          "phase1_enabled": bool,
          "hebbian_expansion": bool,
          "karana_mode": str,
          "total_beliefs": int,
          "current": int,
          "superseded": int,
          "disputed": int,
          "archived": int,
          "hebbian_edges": int,
          "ganita_tuples": int,
          "ingest_tick": int
        }
        On failure: {"error": {...}}
    """
    try:
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
    except Exception as e:
        return _structured_error(
            "stats_failed",
            f"Could not read stats: {type(e).__name__}: {e}",
        )


# ─── Resources ──────────────────────────────────────────────────────
# Resources are addressable data Claude (or any MCP client) can read
# without a tool call. Exposing the current belief summary here means
# a client can surface memory at conversation start with no roundtrip.


@mcp.resource("patha://beliefs/current")
def _beliefs_current() -> str:
    """Current belief summary (all non-superseded beliefs)."""
    try:
        patha = _get_patha()
        store = patha.belief_layer.store
        current = list(store.current())
        if not current:
            return (
                "No beliefs yet. Patha is installed and ready; ingest "
                "facts as the user tells you about themselves."
            )
        lines = [
            f"Patha — user's current beliefs ({len(current)} total). "
            f"Stored locally at ~/.patha/beliefs.jsonl.",
            "",
        ]
        for b in current:
            date = (
                b.asserted_at.strftime("%Y-%m-%d") if b.asserted_at else "-"
            )
            lines.append(f"- [{date}] {b.proposition}")
        return "\n".join(lines)
    except Exception as e:
        return f"(patha resource error: {type(e).__name__}: {e})"


@mcp.resource("patha://beliefs/all")
def _beliefs_all() -> str:
    """All beliefs including superseded ones, sorted newest-first."""
    try:
        patha = _get_patha()
        store = patha.belief_layer.store
        current = list(store.current())
        superseded = list(store.superseded())
        if not current and not superseded:
            return "No beliefs yet."
        lines = [
            f"Patha — all beliefs ({len(current)} current, "
            f"{len(superseded)} superseded).",
            "",
            "## Current",
        ]
        for b in current:
            date = (
                b.asserted_at.strftime("%Y-%m-%d") if b.asserted_at else "-"
            )
            lines.append(f"- [{date}] {b.proposition}")
        if superseded:
            lines.append("")
            lines.append("## Superseded (no longer current)")
            for b in sorted(
                superseded, key=lambda x: x.asserted_at, reverse=True,
            ):
                date = (
                    b.asserted_at.strftime("%Y-%m-%d")
                    if b.asserted_at else "-"
                )
                lines.append(f"- [{date}] ~~{b.proposition}~~")
        return "\n".join(lines)
    except Exception as e:
        return f"(patha resource error: {type(e).__name__}: {e})"


@mcp.resource("patha://stats")
def _resource_stats() -> str:
    """Human-readable belief store stats."""
    try:
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
    except Exception as e:
        return f"(patha resource error: {type(e).__name__}: {e})"


# ─── Entry point ────────────────────────────────────────────────────


def main() -> int:
    """Run the MCP server on stdio. Called by the `patha-mcp` script.

    NEVER print to stdout from this process: stdio MCP uses stdout for
    JSON-RPC framing; any unsolicited stdout corrupts the protocol and
    causes the client to disconnect. All diagnostics use stderr.
    """
    mcp.run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
