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
from patha.belief.plasticity import (
    HebbianAssociation,
    HomeostaticRegulation,
    LongTermDepression,
    SynapticPruning,
)
from patha.belief.pramana import detect_pramana
from patha.belief.store import BeliefStore
from patha.belief.types import (
    Belief,
    BeliefId,
    ContradictionLabel,
    Pramana,
    PropositionId,
    Validity,
)
from patha.belief.validity_extraction import (
    extract_validity,
    extract_validity_with_fallback,
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


# ─── Plasticity configuration ────────────────────────────────────────

@dataclass
class PlasticityConfig:
    """Configuration for the neuroplasticity mechanisms that fire during
    normal BeliefLayer operation.

    Each mechanism is opt-out, not opt-in, so the default v0.3 layer is
    dynamic out of the box. Disable individually for ablations.

    Attributes
    ----------
    enabled
        Master switch. If False, no plasticity fires regardless of
        sub-flags. Useful for exact-parity comparisons with v0.1/v0.2.
    ltd_on_query
        Apply LTD (time-based decay) to beliefs whose confidence hasn't
        been updated recently. Fires once per query() call, scoped to
        the candidate beliefs touched by that query.
    ltd_half_life_days
        Half-life for LTD decay. 365 days = confidence halves per year
        of inactivity (toward the floor).
    ltd_floor
        Lower bound for decayed confidence. 0.1 prevents beliefs from
        vanishing entirely; retrieval can still surface them, they're
        just less confident.
    hebbian_on_query
        Record Hebbian co-retrieval edges between beliefs that surface
        in the same query. Strengthens associations over time.
    hebbian_learning_rate
        Per-co-retrieval weight bump. 0.05 means 20 co-retrievals to
        reach weight 1.0.
    homeostasis_on_ingest
        Apply homeostatic regulation every ``homeostasis_interval_ingests``
        calls. Normalises current-belief confidences so no belief
        dominates through reinforcement.
    homeostasis_interval_ingests
        How many ingests between homeostatic re-normalisations. Default
        100 — frequent enough to prevent runaway, rare enough to keep
        ingest cheap.
    homeostasis_target_mean
        Target mean confidence after normalisation. 0.75 keeps beliefs
        reasonably confident while leaving headroom for new evidence.
    pruning_on_ingest
        Apply synaptic pruning every ``pruning_interval_ingests`` calls.
    pruning_interval_ingests
        How many ingests between pruning sweeps. Default 500 — rarer
        than homeostasis because pruning is archival, less reversible.
    pruning_max_depth
        Archive superseded beliefs more than this many hops from a
        current descendant. Default 10 (matches SynapticPruning default).
    """

    enabled: bool = True
    ltd_on_query: bool = True
    ltd_half_life_days: float = 365.0
    ltd_floor: float = 0.1
    hebbian_on_query: bool = True
    hebbian_learning_rate: float = 0.05
    homeostasis_on_ingest: bool = True
    homeostasis_interval_ingests: int = 100
    homeostasis_target_mean: float = 0.75
    pruning_on_ingest: bool = True
    pruning_interval_ingests: int = 500
    pruning_max_depth: int = 10


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
        auto_extract_validity: bool = True,
        validity_llm_generate=None,
        plasticity: PlasticityConfig | None = None,
        ingest_window_days: int | None = None,
        confidence_weighted_supersession: bool = False,
        confidence_margin: float = 0.2,
    ) -> None:
        self.store = store if store is not None else BeliefStore()
        self.detector = detector if detector is not None else StubContradictionDetector()
        self._contradiction_threshold = contradiction_threshold
        self._entailment_threshold = entailment_threshold
        self._auto_extract_validity = auto_extract_validity
        self._validity_llm_generate = validity_llm_generate
        self._ingest_window_days = ingest_window_days
        self._confidence_weighted = confidence_weighted_supersession
        self._confidence_margin = confidence_margin

        # Plasticity mechanisms. Instantiated up-front so callers can
        # inspect state between queries (e.g., hebbian.related(belief_id)).
        self.plasticity = plasticity if plasticity is not None else PlasticityConfig()
        self._ltd = LongTermDepression(
            half_life_days=self.plasticity.ltd_half_life_days,
            floor=self.plasticity.ltd_floor,
        )
        self.hebbian = HebbianAssociation(
            learning_rate=self.plasticity.hebbian_learning_rate,
        )
        self._homeostasis = HomeostaticRegulation(
            target_mean=self.plasticity.homeostasis_target_mean,
        )
        self._pruning = SynapticPruning(
            max_depth=self.plasticity.pruning_max_depth,
        )
        # Ingest tick — advances every .ingest() call; used to schedule
        # periodic homeostasis and pruning without running them on every ingest.
        self._ingest_tick = 0

    # ── ingest ──────────────────────────────────────────────────────

    def ingest(
        self,
        proposition: str,
        *,
        asserted_at: datetime,
        asserted_in_session: str,
        source_proposition_id: PropositionId,
        validity: Validity | None = None,
        confidence: float | None = None,
        candidate_belief_ids: list[BeliefId] | None = None,
        pramana: Pramana | None = None,
        source_id: str | None = None,
    ) -> IngestEvent:
        """Ingest a new proposition into the belief layer.

        If ``candidate_belief_ids`` is provided, contradiction checks are
        only run against those ids. Otherwise, the new proposition is
        checked against every current belief in the store (O(N) — acceptable
        for v0.1; sliding-window scoping lands in v0.2 per D2).

        Returns an IngestEvent describing what the layer did.
        """
        # Advance the ingest tick first — plasticity schedules depend
        # on it, and we want them to fire even on early-return paths
        # (e.g., the first-ever ingest with no candidates).
        self._ingest_tick += 1

        # If no explicit validity was passed, try to extract one from
        # the proposition text. Rule-based first; LLM fallback for
        # implicit durations if a validity_llm_generate was configured.
        # Falls back to permanent default inside BeliefStore.add() when
        # extraction returns None.
        if validity is None and self._auto_extract_validity:
            validity = extract_validity_with_fallback(
                proposition,
                asserted_at=asserted_at,
                llm_generate=self._validity_llm_generate,
            )

        # Pramāṇa auto-detection unless explicitly provided.
        if pramana is None:
            inferred = detect_pramana(proposition)
            pramana = inferred.pramana

        # Create the new belief first so it has an id we can reference
        # in supersede/reinforce relations. When confidence is None, the
        # store applies a pramāṇa-weighted default (PRATYAKṢA=1.0,
        # SHABDA=0.6×source_reliability, etc.).
        new = self.store.add(
            proposition=proposition,
            asserted_at=asserted_at,
            asserted_in_session=asserted_in_session,
            source_proposition_id=source_proposition_id,
            validity=validity,
            confidence=confidence,
            pramana=pramana,
            source_id=source_id,
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

        # D2 hybrid: sliding-window scope. When ingest_window_days is
        # set, only check against beliefs asserted within the window.
        # Bounds contradiction-check cost at scale (O(window) instead
        # of O(N)). Older beliefs still get checked at query time via
        # the belief layer's retrieve path.
        if self._ingest_window_days is not None:
            from datetime import timedelta
            window_start = asserted_at - timedelta(
                days=self._ingest_window_days
            )
            candidates = [c for c in candidates if c.asserted_at >= window_start]

        if not candidates:
            self._run_scheduled_plasticity()
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
                # Pramāṇa-aware resolution: the store decides whether
                # this is a temporal supersession or a sublation based
                # on which belief carries stronger pramāṇa (and, when
                # confidence_weighted_supersession is set, confidence
                # scores too).
                self.store.resolve_contradiction(
                    candidate.id,
                    new.id,
                    confidence_weighted=self._confidence_weighted,
                    confidence_margin=self._confidence_margin,
                )
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

        # Plasticity: scheduled maintenance (homeostasis, pruning)
        self._run_scheduled_plasticity()

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

    # ── plasticity hooks ────────────────────────────────────────────

    def _run_scheduled_plasticity(self) -> None:
        """Fire homeostasis and pruning on their configured intervals.

        Called once per ingest. No-op if plasticity is disabled.
        """
        if not self.plasticity.enabled:
            return
        if (
            self.plasticity.homeostasis_on_ingest
            and self._ingest_tick % self.plasticity.homeostasis_interval_ingests == 0
        ):
            self._homeostasis.apply(self.store)
        if (
            self.plasticity.pruning_on_ingest
            and self._ingest_tick % self.plasticity.pruning_interval_ingests == 0
        ):
            self._pruning.prune(self.store)

    def _apply_query_plasticity(
        self,
        touched_beliefs: list[Belief],
        at_time: datetime,
    ) -> None:
        """Apply query-time plasticity: LTD decay + Hebbian co-retrieval.

        Called by query() on the beliefs it actually surfaces.
        """
        if not self.plasticity.enabled:
            return
        if self.plasticity.ltd_on_query and touched_beliefs:
            self._ltd.apply_to_store(
                self.store, now=at_time, beliefs=touched_beliefs
            )
        if self.plasticity.hebbian_on_query and len(touched_beliefs) >= 2:
            self.hebbian.record_coretrieval(
                [b.id for b in touched_beliefs]
            )

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

        # Plasticity: LTD decay on surfaced beliefs, Hebbian co-retrieval
        # edges. Applied before history walk so decayed confidences flow
        # through to the summary.
        self._apply_query_plasticity(current, t)

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
