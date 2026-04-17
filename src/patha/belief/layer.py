"""BeliefLayer — the public API of the Phase 2 belief layer.

Ties together contradiction detection and the belief store. Given a new
proposition and a set of candidate existing beliefs, the layer decides
how to update the store:

  - neutral to everything          → add as a new belief
  - entails an existing belief     → reinforce the existing one
  - contradicts an existing belief → new supersedes old (non-destructive)

For retrieval queries, the layer filters Phase 1 retrieval output to
return only current beliefs by default, with optional history.

v0.1 design decisions (per docs/phase_2_spec.md §9):
  - D1 contradiction detection: NLI (swap-able via DI of detector)
  - D2 detection timing:        query-time only (ingest is cheap, no O(N²))
  - D3 supersession policy:     neutral, non-destructive (AGM Preservation)
  - D4 validity:                explicit-only; inference deferred to v0.2
  - D7 compression:             Option B — structured belief summary

Deferred to v0.2:
  - Ingest-time contradiction detection with a 30-day sliding window
  - Confidence-weighted supersession (currently pure temporal ordering)
  - Rule-based validity extraction (HeidelTime/SUTime)
  - Direct-answer compression (Option C) for lookup queries
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from patha.belief.contradiction import (
    ContradictionDetector,
    StubContradictionDetector,
)
from patha.belief.store import BeliefStore
from patha.belief.types import (
    Belief,
    BeliefId,
    ContradictionLabel,
    PropositionId,
    Validity,
)


# ─── Ingest outcomes ─────────────────────────────────────────────────

IngestAction = Literal["added", "reinforced", "superseded"]


@dataclass(frozen=True)
class IngestEvent:
    """Outcome of ingesting one proposition into the belief layer.

    Attributes
    ----------
    action
        What the layer did:
          - "added":       new belief created, no relation to existing
          - "reinforced":  new assertion matched an existing belief;
                           confidence bumped, no new supersession edge
          - "superseded":  new assertion contradicts one or more existing
                           beliefs; new belief supersedes them
    new_belief
        The belief that was added (always present).
    affected_belief_ids
        For "reinforced" or "superseded": the ids of existing beliefs
        that were touched. Empty for "added".
    contradictions_detected
        The number of contradiction checks that returned CONTRADICTS.
        Useful for debugging + metrics.
    """

    action: IngestAction
    new_belief: Belief
    affected_belief_ids: tuple[BeliefId, ...] = ()
    contradictions_detected: int = 0


# ─── Retrieval outcomes ──────────────────────────────────────────────

@dataclass(frozen=True)
class BeliefQueryResult:
    """A belief layer query result.

    Attributes
    ----------
    current
        Current (non-superseded) beliefs matching the query.
    history
        Superseded beliefs related to the current ones, if requested.
        Empty unless ``include_history=True`` was passed to query().
    tokens_in_summary
        Rough token count of the structured summary this result would
        render to if sent to an LLM. Used for the token-economy
        evaluation axis — lets callers compare against naive RAG.
    """

    current: list[Belief]
    history: list[Belief]
    tokens_in_summary: int


# ─── BeliefLayer ─────────────────────────────────────────────────────

class BeliefLayer:
    """Top-level API for the Patha belief layer.

    Composition, not inheritance:
      - A ContradictionDetector does the NLI work.
      - A BeliefStore holds and indexes beliefs.

    The layer is stateless apart from the store — same layer instance
    can be used across sessions.

    Parameters
    ----------
    store
        The backing BeliefStore. Pass a persistence-enabled one for
        durable state.
    detector
        The contradiction detector. Defaults to the stub for
        zero-download behaviour; swap in NLIContradictionDetector for
        production.
    contradiction_threshold
        Minimum confidence for a CONTRADICTS verdict to trigger
        supersession. Below this, we treat it as NEUTRAL. Default 0.75
        — stricter than the detector's own threshold because we are
        about to change stored state on this signal.
    entailment_threshold
        Minimum confidence for ENTAILS to trigger reinforcement.
        Default 0.70.
    """

    def __init__(
        self,
        store: BeliefStore | None = None,
        detector: ContradictionDetector | None = None,
        *,
        contradiction_threshold: float = 0.75,
        entailment_threshold: float = 0.70,
    ) -> None:
        self.store = store if store is not None else BeliefStore()
        self.detector = detector if detector is not None else StubContradictionDetector()
        self._contradiction_threshold = contradiction_threshold
        self._entailment_threshold = entailment_threshold

    # ── ingest ──────────────────────────────────────────────────────

    def ingest(
        self,
        proposition: str,
        *,
        asserted_at: datetime,
        asserted_in_session: str,
        source_proposition_id: PropositionId,
        validity: Validity | None = None,
        confidence: float = 1.0,
        candidate_belief_ids: list[BeliefId] | None = None,
    ) -> IngestEvent:
        """Ingest a new proposition into the belief layer.

        If ``candidate_belief_ids`` is provided, contradiction checks are
        only run against those ids. Otherwise, the new proposition is
        checked against every current belief in the store (O(N) — acceptable
        for v0.1; sliding-window scoping lands in v0.2 per D2).

        Returns an IngestEvent describing what the layer did.
        """
        # Create the new belief first so it has an id we can reference
        # in supersede/reinforce relations.
        new = self.store.add(
            proposition=proposition,
            asserted_at=asserted_at,
            asserted_in_session=asserted_in_session,
            source_proposition_id=source_proposition_id,
            validity=validity,
            confidence=confidence,
        )

        # Candidate beliefs to check against
        if candidate_belief_ids is None:
            candidates = [b for b in self.store.current() if b.id != new.id]
        else:
            candidates = [
                b for bid in candidate_belief_ids
                if (b := self.store.get(bid)) is not None
                and b.is_current
                and b.id != new.id
            ]

        if not candidates:
            return IngestEvent(action="added", new_belief=new)

        # Batch contradiction check.
        # Premise = candidate (older), Hypothesis = new (recent).
        # An NLI model interprets "does the new claim contradict the older one?"
        pairs = [(c.proposition, new.proposition) for c in candidates]
        results = self.detector.detect_batch(pairs)

        superseded_ids: list[BeliefId] = []
        reinforced_id: BeliefId | None = None
        contradictions_detected = 0

        for candidate, result in zip(candidates, results):
            if (
                result.label == ContradictionLabel.CONTRADICTS
                and result.confidence >= self._contradiction_threshold
            ):
                self.store.supersede(candidate.id, new.id)
                superseded_ids.append(candidate.id)
                contradictions_detected += 1
            elif (
                result.label == ContradictionLabel.ENTAILS
                and result.confidence >= self._entailment_threshold
                and reinforced_id is None
            ):
                # Reinforce only the first entailment match — we don't
                # want to multiply-reinforce a single user repetition.
                self.store.reinforce(candidate.id, new.id)
                reinforced_id = candidate.id

        if superseded_ids:
            return IngestEvent(
                action="superseded",
                new_belief=new,
                affected_belief_ids=tuple(superseded_ids),
                contradictions_detected=contradictions_detected,
            )
        if reinforced_id is not None:
            return IngestEvent(
                action="reinforced",
                new_belief=new,
                affected_belief_ids=(reinforced_id,),
                contradictions_detected=0,
            )
        return IngestEvent(action="added", new_belief=new)

    # ── query ───────────────────────────────────────────────────────

    def query(
        self,
        candidate_belief_ids: list[BeliefId],
        *,
        at_time: datetime | None = None,
        include_history: bool = False,
    ) -> BeliefQueryResult:
        """Filter a list of candidate beliefs to current-state view.

        Typical usage: Phase 1 retrieval surfaces candidate beliefs; the
        layer filters them down to only the current ones (and optionally
        their supersession history).

        Parameters
        ----------
        candidate_belief_ids
            IDs of beliefs surfaced by Phase 1 retrieval.
        at_time
            The time to evaluate validity at. None means "now".
        include_history
            If True, superseded ancestors of each current belief are
            returned in .history.

        Returns
        -------
        BeliefQueryResult
        """
        t = at_time if at_time is not None else datetime.now()

        current: list[Belief] = []
        seen: set[BeliefId] = set()

        for bid in candidate_belief_ids:
            if bid in seen:
                continue
            seen.add(bid)
            b = self.store.get(bid)
            if b is None:
                continue
            if not b.is_current:
                # Skip the superseded candidate; its successor(s) will
                # surface instead if retrieval also picked them up.
                continue
            if not b.validity.is_valid_at(t):
                continue
            current.append(b)

        history: list[Belief] = []
        if include_history:
            history_seen: set[BeliefId] = set()
            for b in current:
                for ancestor in self.store.lineage(b.id):
                    if ancestor.id == b.id:
                        continue
                    if ancestor.id in history_seen:
                        continue
                    history_seen.add(ancestor.id)
                    history.append(ancestor)

        tokens = _estimate_summary_tokens(current, history)
        return BeliefQueryResult(
            current=current, history=history, tokens_in_summary=tokens
        )

    # ── rendering (D7 compression: Option B — structured summary) ───

    def render_summary(
        self,
        result: BeliefQueryResult,
        *,
        include_history: bool = False,
    ) -> str:
        """Render a query result as a compact text summary for an LLM.

        This is the Option B compression strategy: send the structured
        belief state, not the raw proposition soup. A lookup that would
        have been 12 propositions at ~20 tokens each becomes one current
        belief line and (optionally) a brief lineage.
        """
        lines: list[str] = []
        if not result.current:
            return "(no current belief)"
        for b in result.current:
            line = f"- [{_fmt_date(b.asserted_at)}] {b.proposition}"
            if b.reinforced_by:
                line += f" (reinforced {len(b.reinforced_by)}×)"
            lines.append(line)
        if include_history and result.history:
            lines.append("")
            lines.append("Earlier beliefs (superseded):")
            for b in result.history:
                lines.append(f"  ~ [{_fmt_date(b.asserted_at)}] {b.proposition}")
        return "\n".join(lines)


# ─── helpers ─────────────────────────────────────────────────────────

def _estimate_summary_tokens(
    current: list[Belief], history: list[Belief]
) -> int:
    """Rough token estimate for a structured-summary render.

    Not calibrated to a specific tokenizer — a ~4-chars-per-token
    heuristic. Use for relative comparisons (compression ratios), not
    for precise billing.
    """
    chars = 0
    for b in current:
        chars += len(b.proposition) + 20  # date prefix + formatting
    for b in history:
        chars += len(b.proposition) + 20
    return max(1, chars // 4)


def _fmt_date(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")
