"""Direct-answer compression (D7 Option C).

When a query is a belief LOOKUP — "what do I currently believe about X?",
"where do I live now?", "am I still avoiding raw fish?" — Patha can
answer from belief state without invoking the downstream LLM at all.

This is the Phase 2 move that makes the token-economy claim true:
memory systems that increase token use are search interfaces with
nicer names; memory systems that REDUCE tokens are actual memory.

Flow:
  1. Classify the query. Is it a lookup that the belief store can
     answer directly, or a generation query that needs the LLM?
  2. For lookups: retrieve the relevant current belief(s), render a
     structured answer. Zero LLM tokens.
  3. For generation: return None. Caller falls back to the standard
     retrieve -> rerank -> LLM path.

Design rationale (see docs/phase_2_spec.md §4.3 and §9 D7):
  - No new ML is needed for the classifier — regex + small rule set
    handles the bulk of belief-lookup queries cleanly.
  - Every direct answer includes provenance: the belief ids and
    timestamps. Users and callers can trace the answer back to its
    evidence.
  - Fallback is the default. We return None when unsure, not a wrong
    confident answer.

Impact: on BeliefEval, the 15 'current_belief' questions are all
lookup-shaped. With direct-answer compression, they can be answered
with 0 LLM tokens instead of ~500-2000 for naive RAG. This is the
compression claim.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from patha.belief.store import BeliefStore
from patha.belief.types import Belief, BeliefId, ResolutionStatus


# ─── Query classification ────────────────────────────────────────────

# Patterns that mark a query as a belief lookup. Deliberately narrow —
# false positives (classifying generation queries as lookups) are much
# worse than false negatives (falling back to LLM unnecessarily).

_LOOKUP_PATTERNS: list[re.Pattern] = [
    # "what do i currently believe/think about X"
    re.compile(
        r"\bwhat\s+(?:do\s+)?(?:i|you|the\s+user)\s+(?:currently|now|still)\s+"
        r"(?:believe|think|feel|prefer|do|like|eat|drink|use|own)",
        re.IGNORECASE,
    ),
    # "where/what/how does the user X now/currently"
    re.compile(
        r"\b(?:what|where|how|when|who)\s+(?:does|is)\s+(?:the\s+user|they|i|you)\b[^?]*\b(?:now|currently|today)\b",
        re.IGNORECASE,
    ),
    # "is/does the user still X"
    re.compile(
        r"\b(?:is|does|do)\s+(?:the\s+user|they|i|you)\s+(?:still|currently|now)\b",
        re.IGNORECASE,
    ),
    # "what is the user's current X"
    re.compile(
        r"\bwhat\s+is\s+(?:the\s+user'?s?|my|your)\s+current\b",
        re.IGNORECASE,
    ),
    # "what <noun> does/is X currently/now Y" — catches "What car does the user drive now"
    re.compile(
        r"\bwhat\s+\w+\s+(?:does|is)\s+.{1,40}\b(?:currently|now)\b",
        re.IGNORECASE,
    ),
    # "who|what is currently/now the X" — "Who is currently the lead?"
    re.compile(
        r"\b(?:who|what|where|how)\s+(?:is|are)\s+(?:currently|now)\b",
        re.IGNORECASE,
    ),
    # "how old / when did X change"
    re.compile(
        r"\bwhen\s+did\s+(?:the\s+user'?s?|my|your)\b.*\b(?:change|move|switch|start|stop)\b",
        re.IGNORECASE,
    ),
    # "when did the user change ..."
    re.compile(
        r"\bwhen\s+did\s+.{1,40}\b(?:change|move|switch|start|stop)\b",
        re.IGNORECASE,
    ),
    # "does the user currently eat X?"
    re.compile(
        r"\bdoes\s+(?:the\s+user|they|i|you)\s+(?:currently|still|now)\b",
        re.IGNORECASE,
    ),
]


def is_belief_lookup(query: str) -> bool:
    """Return True if the query looks like a belief lookup.

    Conservative: classifies only queries that match one of the known
    patterns. Everything else falls back to LLM generation.
    """
    for pat in _LOOKUP_PATTERNS:
        if pat.search(query):
            return True
    return False


# ─── Direct answer rendering ─────────────────────────────────────────

@dataclass(frozen=True)
class DirectAnswer:
    """A direct answer produced from belief state without LLM inference.

    Attributes
    ----------
    text
        The answer string, ready to be returned to the user or inserted
        into a response.
    belief_ids
        The ids of the beliefs this answer was synthesised from.
        Provenance: callers can trace every claim back to its record.
    tokens_used
        Estimated tokens in ``text`` (rough 4-chars-per-token heuristic).
        Zero LLM tokens were spent — this is the rendered-output cost,
        which is all the downstream caller pays.
    include_history
        Whether the answer mentions superseded predecessors.
    """

    text: str
    belief_ids: list[BeliefId]
    tokens_used: int
    include_history: bool


class DirectAnswerer:
    """Renders belief-state lookups into direct answers without an LLM.

    Parameters
    ----------
    store
        The BeliefStore to query.
    include_history_by_default
        If True, append a short lineage note to every answer ('Earlier:
        <superseded belief>'). If False (default), history is surfaced
        only when the query explicitly asks when something changed.
    """

    def __init__(
        self,
        store: BeliefStore,
        *,
        include_history_by_default: bool = False,
    ) -> None:
        self._store = store
        self._include_history_default = include_history_by_default

    # ── main entry ──────────────────────────────────────────────────

    def try_answer(
        self,
        query: str,
        candidate_belief_ids: list[BeliefId],
        *,
        at_time: datetime | None = None,
    ) -> DirectAnswer | None:
        """Try to answer a query directly from belief state.

        Returns None if the query is not a belief lookup, or if no
        current belief among the candidates is available. Callers treat
        None as "fall back to LLM".
        """
        if not is_belief_lookup(query):
            return None

        include_history = (
            self._include_history_default
            or self._asks_about_change(query)
        )
        t = at_time or datetime.now()

        # Filter to current, validity-valid beliefs among the candidates
        current: list[Belief] = []
        for bid in candidate_belief_ids:
            b = self._store.get(bid)
            if b is None:
                continue
            if not b.is_current:
                continue
            if not b.validity.is_valid_at(t):
                continue
            current.append(b)

        if not current:
            return None

        # Build the answer text
        lines = [f"- {b.proposition}" for b in current]
        disputed = [b for b in current if b.is_disputed]
        if disputed:
            lines.append(
                f"  (note: {len(disputed)} disputed belief(s); "
                "resolution pending)"
            )
        if include_history:
            history: list[Belief] = []
            seen: set[BeliefId] = set()
            for b in current:
                for ancestor in self._store.lineage(b.id):
                    if ancestor.id == b.id or ancestor.id in seen:
                        continue
                    seen.add(ancestor.id)
                    history.append(ancestor)
            if history:
                lines.append("")
                lines.append("Earlier beliefs:")
                for b in history:
                    date_str = b.asserted_at.strftime("%Y-%m-%d")
                    lines.append(f"  ~ [{date_str}] {b.proposition}")

        text = "\n".join(lines)
        tokens = max(1, len(text) // 4)
        return DirectAnswer(
            text=text,
            belief_ids=[b.id for b in current],
            tokens_used=tokens,
            include_history=include_history,
        )

    # ── helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _asks_about_change(query: str) -> bool:
        q = query.lower()
        return any(
            cue in q
            for cue in ("when did", "used to", "previously", "before", "history")
        )
