"""End-to-end Patha: Phase 1 retrieval + Phase 2 belief layer.

Combines the two phases into a single system that can ingest
conversations and answer questions with:

  1. Phase 1 — multi-view retrieval (RRF + BM25 + songline + reranking)
  2. Phase 2 — belief-layer filtering (current-only, non-destructive
     supersession, validity windows, optional history)
  3. Compression routing — direct-answer for lookup queries, structured
     summary for reasoning queries, raw propositions for open-ended

This module is the 'use Patha as a memory system' entry point. It does
not modify Phase 1 at all — the integration is additive.

Correspondence model:
  Each ingested conversation turn becomes:
    - one or more propositions in the Phase 1 proposition store
      (chunked, 7-view-indexed, BM25-indexed, songline-linked)
    - one belief in the Phase 2 BeliefStore, with
      source_proposition_id linking back

At query time:
    retrieve(query) -> Phase 1 candidate chunk_ids
       -> map chunk_id -> belief_id via source_proposition_id
       -> Phase 2 .query(belief_ids, at_time) filters + enriches
       -> direct_answer.try_answer() attempts no-LLM response
       -> fall back to structured summary or raw propositions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from patha.belief.contradiction import (
    ContradictionDetector,
    StubContradictionDetector,
)
from patha.belief.direct_answer import DirectAnswer, DirectAnswerer
from patha.belief.layer import BeliefLayer, BeliefQueryResult, IngestEvent
from patha.belief.raw_archive import RawArchive
from patha.belief.store import BeliefStore
from patha.belief.types import PropositionId


# ─── Outcomes ────────────────────────────────────────────────────────

ResponseStrategy = Literal["direct_answer", "structured", "raw"]


@dataclass
class IntegratedResponse:
    """Output of a full query through the integrated system.

    Attributes
    ----------
    query
        The original question.
    strategy
        Which compression strategy produced the response:
          - 'direct_answer' : no LLM call made (Phase 2 answered alone)
          - 'structured'    : structured belief summary to send to LLM
          - 'raw'           : raw retrieved propositions to send to LLM
    prompt
        The text to send to a downstream LLM (empty string if direct_answer).
    answer
        Populated only when strategy == 'direct_answer'. Empty otherwise.
    belief_ids
        Beliefs surfaced for this query (current ones from Phase 2).
    source_proposition_ids
        Underlying Phase 1 proposition ids, for traceability.
    tokens_in
        Estimated LLM input tokens (0 for direct_answer).
    retrieval_result
        The raw Phase 2 query result, for advanced callers.
    """

    query: str
    strategy: ResponseStrategy
    prompt: str
    answer: str
    belief_ids: list[str]
    source_proposition_ids: list[PropositionId]
    tokens_in: int
    retrieval_result: BeliefQueryResult | None = None


# ─── Integrated system ───────────────────────────────────────────────

class IntegratedPatha:
    """End-to-end Patha: Phase 1 retrieval + Phase 2 belief layer.

    Parameters
    ----------
    phase1_retrieve
        Callable that takes a query and returns a list of candidate
        proposition ids (in rank order). Usually wraps
        `patha.retrieval.pipeline.retrieve` + `.top_ids`.
        Pass None for pure Phase 2 testing — in that mode, candidates
        default to the entire belief store (tiny-scale only).
    belief_layer
        The Phase 2 BeliefLayer. Create with a persistent store if you
        want state across sessions.
    direct_answerer
        Optional DirectAnswerer. Created automatically from
        belief_layer.store if not provided.
    system_prompt
        System prompt prefixed to LLM-bound responses (structured or raw).
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You are a careful assistant that answers questions about the "
        "user based on their stated memory. Be concise and factual. "
        "When the user's beliefs have changed over time, respect the "
        "current state and flag the change if relevant."
    )

    def __init__(
        self,
        phase1_retrieve=None,
        belief_layer: BeliefLayer | None = None,
        direct_answerer: DirectAnswerer | None = None,
        raw_archive: RawArchive | None = None,
        *,
        system_prompt: str | None = None,
    ) -> None:
        self._phase1_retrieve = phase1_retrieve
        self.belief_layer = belief_layer if belief_layer is not None else BeliefLayer(
            store=BeliefStore(), detector=StubContradictionDetector()
        )
        self.direct_answerer = (
            direct_answerer
            if direct_answerer is not None
            else DirectAnswerer(self.belief_layer.store)
        )
        # Raw Archive (v0.3, wired into ingest flow in v0.5).
        # When provided, every ingest automatically creates a raw-turn
        # record and links the resulting proposition_id to it, so full
        # provenance is preserved from the start.
        self.raw_archive = raw_archive
        # Per-session turn counter for the raw archive
        self._turn_counters: dict[str, int] = {}
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

    # ── ingestion ───────────────────────────────────────────────────

    def ingest(
        self,
        proposition: str,
        *,
        asserted_at: datetime,
        asserted_in_session: str,
        source_proposition_id: PropositionId,
        speaker: str = "user",
        raw_content: str | None = None,
        source_name: str = "integrated",
    ) -> IngestEvent:
        """Ingest one proposition into the belief layer (and archive).

        Phase 1 (proposition store) ingestion is the caller's
        responsibility — call `patha.indexing.ingest.ingest_session`
        separately to populate the multi-view index. This method
        updates Phase 2 and, if a raw_archive is wired, also records
        the raw turn for full provenance.

        Parameters
        ----------
        speaker
            Who uttered the proposition. Default 'user'.
        raw_content
            The original verbatim text. Defaults to the proposition
            itself — only differs when the proposition is a distilled
            atomic claim from a longer utterance.
        source_name
            Archive source label (e.g., 'slack-dm', 'voice-memo').
        """
        # Raw archive: record the turn and link the proposition id
        if self.raw_archive is not None:
            idx = self._turn_counters.get(asserted_in_session, 0)
            turn = self.raw_archive.add_turn(
                session_id=asserted_in_session,
                turn_index=idx,
                speaker=speaker,
                content=raw_content if raw_content is not None else proposition,
                timestamp=asserted_at,
                source_name=source_name,
            )
            self._turn_counters[asserted_in_session] = idx + 1
            self.raw_archive.link_proposition(
                raw_turn_id=turn.id,
                proposition_id=source_proposition_id,
            )

        return self.belief_layer.ingest(
            proposition=proposition,
            asserted_at=asserted_at,
            asserted_in_session=asserted_in_session,
            source_proposition_id=source_proposition_id,
        )

    # ── query path ──────────────────────────────────────────────────

    def query(
        self,
        question: str,
        *,
        at_time: datetime | None = None,
        phase1_top_k: int = 20,
        include_history: bool = False,
    ) -> IntegratedResponse:
        """Answer a question by running Phase 1 + Phase 2 end-to-end.

        Parameters
        ----------
        question
            Natural-language query.
        at_time
            Evaluate validity as of this time. None = now.
        phase1_top_k
            How many candidates Phase 1 should surface. Phase 2 filters
            these to current beliefs.
        include_history
            If True, structured and direct-answer responses include
            supersession lineage. If False (default), only current
            state is returned.
        """
        # 1. Phase 1 retrieval — candidate propositions
        if self._phase1_retrieve is not None:
            proposition_ids = list(self._phase1_retrieve(question, phase1_top_k))
        else:
            # Fallback: candidate = every current belief's source proposition
            proposition_ids = [
                b.source_proposition_id
                for b in self.belief_layer.store.current()
            ]

        # 2. Map Phase 1 proposition ids → Phase 2 belief ids
        belief_ids: list[str] = []
        for pid in proposition_ids:
            belief = self.belief_layer.store.by_proposition(pid)
            if belief is not None:
                belief_ids.append(belief.id)

        # 3. Phase 2 belief-layer filter
        result = self.belief_layer.query(
            belief_ids,
            at_time=at_time,
            include_history=include_history,
        )

        # 4. Compression routing: try direct-answer first
        direct = self.direct_answerer.try_answer(
            question, belief_ids, at_time=at_time
        )
        if direct is not None:
            return IntegratedResponse(
                query=question,
                strategy="direct_answer",
                prompt="",  # no LLM call
                answer=direct.text,
                belief_ids=direct.belief_ids,
                source_proposition_ids=[
                    b.source_proposition_id for b in result.current
                    if b.id in direct.belief_ids
                ],
                tokens_in=0,
                retrieval_result=result,
            )

        # 5. Structured summary (Option B) for non-lookup queries.
        # Option A (raw) is used only when structured produces nothing.
        if result.current:
            summary = self.belief_layer.render_summary(
                result, include_history=include_history
            )
            prompt = (
                self.system_prompt
                + "\n\nCurrent beliefs:\n"
                + summary
                + f"\n\nQuestion: {question}\nAnswer:"
            )
            return IntegratedResponse(
                query=question,
                strategy="structured",
                prompt=prompt,
                answer="",
                belief_ids=[b.id for b in result.current],
                source_proposition_ids=[
                    b.source_proposition_id for b in result.current
                ],
                tokens_in=_approx_tokens(prompt),
                retrieval_result=result,
            )

        # 6. Raw fallback — no current beliefs; Phase 1 candidates only.
        raw_props = []
        for pid in proposition_ids[:phase1_top_k]:
            belief = self.belief_layer.store.by_proposition(pid)
            if belief is not None:
                raw_props.append(belief.proposition)

        prompt = (
            self.system_prompt
            + "\n\nContext:\n"
            + "\n".join(f"- {p}" for p in raw_props)
            + f"\n\nQuestion: {question}\nAnswer:"
        )
        return IntegratedResponse(
            query=question,
            strategy="raw",
            prompt=prompt,
            answer="",
            belief_ids=[],
            source_proposition_ids=list(proposition_ids[:phase1_top_k]),
            tokens_in=_approx_tokens(prompt),
            retrieval_result=result,
        )


# ─── helpers ─────────────────────────────────────────────────────────

def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)
