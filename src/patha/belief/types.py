"""Core data types for the Patha belief layer.

These are the foundational types used across the belief layer modules
(contradiction, supersession, validity, store). The types are deliberately
conservative for v0.1 — each field has a narrow, well-defined meaning.
Extensions (probabilistic confidence, causal links, multi-agent attribution)
are deferred until the simple version proves itself.

Design references:
- Belief vs belief set separation: Hansson (1999)
- AGM revision postulates (Success, Consistency, Preservation): Alchourrón,
  Gärdenfors, Makinson (1985)
- See docs/phase_2_spec.md §2.2 for the full data model rationale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Literal


# ─── Type aliases ────────────────────────────────────────────────────

BeliefId = str
"""Opaque, stable identifier for a Belief. Assigned at creation."""

PropositionId = str
"""Identifier of the underlying Phase-1 proposition row this belief was derived from."""


# ─── Contradiction detection outputs ─────────────────────────────────

class ContradictionLabel(str, Enum):
    """Three-way classification of the relation between two propositions.

    Mirrors the standard NLI label set (ENTAILS, CONTRADICTS, NEUTRAL)
    so that NLI models map directly onto this enum. ENTAILS currently
    unused by downstream logic but recorded for future use (could
    drive `supports` edges in a later version).
    """

    CONTRADICTS = "contradicts"
    ENTAILS = "entails"
    NEUTRAL = "neutral"


@dataclass(frozen=True)
class ContradictionResult:
    """Output of a pairwise contradiction check.

    Attributes
    ----------
    label
        Three-way classification.
    confidence
        Score in [0, 1]. For NLI models this is the softmax probability
        of the winning class. Not a calibrated probability unless the
        detector is calibrated; treat as a ranking signal.
    rationale
        Optional short explanation. NLI models typically leave this
        None; LLM-fallback judges may populate it.
    """

    label: ContradictionLabel
    confidence: float
    rationale: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0, 1]; got {self.confidence}"
            )


# ─── Validity (temporal lifespan of a belief) ────────────────────────

ValidityMode = Literal["permanent", "dated_range", "duration", "decay"]
"""How a belief's lifespan is represented.

- permanent:    always valid; never expires.
- dated_range:  valid between an explicit start and end timestamp.
- duration:     valid for a half-life span (inferred from content).
- decay:        default mode for unmarked beliefs; confidence decays
                over a configurable half-life with no hard expiry.

v0.1 ships permanent + dated_range only. duration and decay modes are
specified but their inference logic lands in v0.2.
"""


@dataclass(frozen=True)
class Validity:
    """Temporal validity window for a belief.

    Defaults to permanent if nothing is specified — safest behaviour,
    avoids expiring beliefs that were merely never annotated.

    Attributes
    ----------
    mode
        One of the ValidityMode literals.
    start
        Optional start timestamp. If None, belief is valid from
        assertion time.
    end
        Optional explicit end timestamp. Required for dated_range.
    half_life_days
        For duration and decay modes: the half-life in days. None in
        permanent/dated_range modes.
    source
        Where the validity came from: "explicit" (user-provided),
        "inferred" (extracted from content), or "default" (fallback).
        v0.1 assigns "explicit" for user-provided windows and "default"
        otherwise.
    """

    mode: ValidityMode = "permanent"
    start: datetime | None = None
    end: datetime | None = None
    half_life_days: float | None = None
    source: Literal["explicit", "inferred", "default"] = "default"

    def __post_init__(self) -> None:
        if self.mode == "dated_range" and self.end is None:
            raise ValueError("dated_range validity requires an end timestamp")
        if self.mode in ("duration", "decay") and self.half_life_days is None:
            raise ValueError(
                f"{self.mode} validity requires half_life_days"
            )
        if self.half_life_days is not None and self.half_life_days <= 0:
            raise ValueError(
                f"half_life_days must be positive; got {self.half_life_days}"
            )

    def is_valid_at(self, t: datetime) -> bool:
        """Return whether this validity window includes timestamp t.

        For decay mode, returns True as long as we're not before start —
        decay affects confidence, not hard validity. The confidence
        curve is applied by the store, not here.
        """
        if self.start is not None and t < self.start:
            return False
        if self.mode == "permanent":
            return True
        if self.mode == "dated_range":
            # end is guaranteed non-None by __post_init__
            assert self.end is not None
            return t <= self.end
        if self.mode == "duration":
            # For v0.1 we treat duration as a hard end at start + 3 * half_life
            # (~88% decayed by then). In v0.2 this becomes a soft decay curve.
            assert self.start is not None and self.half_life_days is not None
            from datetime import timedelta
            hard_end = self.start + timedelta(days=3 * self.half_life_days)
            return t <= hard_end
        if self.mode == "decay":
            return True  # decay affects confidence, not validity
        # Should be unreachable given ValidityMode literal typing
        raise ValueError(f"unknown validity mode: {self.mode}")


# ─── Belief (the main object) ────────────────────────────────────────

# ─── Pramāṇa epistemic strength ──────────────────────────────────────

# Default epistemic strength of each pramāṇa, 0-1. Used for:
#   - Default initial confidence of a belief asserted via that pramāṇa
#   - Hierarchy in contradiction resolution (pratyakṣa trumps śabda for
#     directly observable things; the 'bādhā' relation in Nyāya)
#
# These are conservative defaults. A production system can override
# them per-domain (e.g., historical claims flip the hierarchy — for
# events outside perception, testimony outranks direct observation
# because the observer wasn't there).
#
# Values reflect the Nyāya ordering for directly-observable domains:
# pratyakṣa > anumāna > arthāpatti > śabda > anupalabdhi > upamāna.

PRAMANA_STRENGTH: dict[str, float] = {
    "pratyaksa": 1.00,    # direct perception — strongest
    "anumana": 0.80,      # inference
    "arthapatti": 0.70,   # postulation from circumstance
    "anupalabdhi": 0.65,  # absence-based inference
    "shabda": 0.60,       # testimony — valid only if source is āpta
    "upamana": 0.55,      # comparison / analogy
    # UNKNOWN = 1.0 because a bare self-assertion without epistemic
    # markers is treated as implicit direct report — the user knows
    # their own state unless they hedge. This also preserves the v0.1/v0.2
    # behaviour where beliefs default to confidence 1.0.
    "unknown": 1.00,
}


class Pramana(str, Enum):
    """Source of valid knowledge, per the Nyāya and Mīmāṃsā traditions.

    Patha uses pramāṇa as a first-class property of every belief: it
    records *how* the claim came to be known, not just *that* it was
    asserted. Source-independence weighting (see BeliefStore.reinforce)
    uses pramāṇa diversity — two reinforcements via the same pramāṇa
    count less than reinforcements via different ones, because epistemic
    robustness comes from cross-corroboration across kinds of evidence,
    not from repetition of one kind.

    The six classical pramāṇas:

      PRATYAKṢA   — direct perception ("I saw it happen")
      ANUMANA     — inference ("if A, then B; A, so B")
      UPAMANA     — comparison / analogy ("like X but smaller")
      SHABDA      — testimony ("the doctor told me")
      ARTHAPATTI  — postulation / inference from circumstance
                    ("he's alive and not here, so he must be elsewhere")
      ANUPALABDHI — absence-based inference ("I don't see it, so it
                    isn't there")
      UNKNOWN     — fallback when the layer can't infer the pramāṇa
                    (e.g., bare assertions without linguistic markers).

    These are tagged onto beliefs at ingest. Auto-detection is
    rule-based and deliberately conservative; callers can override
    with an explicit pramana= argument when higher confidence is
    available.
    """

    PRATYAKSA = "pratyaksa"
    ANUMANA = "anumana"
    UPAMANA = "upamana"
    SHABDA = "shabda"
    ARTHAPATTI = "arthapatti"
    ANUPALABDHI = "anupalabdhi"
    UNKNOWN = "unknown"


class ResolutionStatus(str, Enum):
    """The relationship status of a belief relative to others in its cluster.

    Moves beyond the binary current/superseded view to reflect the
    multi-outcome resolution policies a belief maintenance system
    actually needs (Hansson 1999 on non-prioritised revision):

      CURRENT      — this belief holds. Nothing supersedes it.
      SUPERSEDED   — a later belief replaced this one. Non-destructive;
                     still queryable through history.
      COEXISTS     — this belief holds alongside another related belief
                     that is neither a strict supersession nor a
                     strict entailment (e.g., two preferences held
                     simultaneously: 'I like sushi' + 'I like steak').
      DISPUTED     — a contradiction was detected but the system
                     does not yet know which side wins. Both are
                     surfaced; neither is authoritative.
      AMBIGUOUS    — a contradiction signal was seen but confidence
                     is too low to act on. Flagged for later review.
      BADHITA      — sublated: contradicted by a stronger pramāṇa.
                     Distinct from SUPERSEDED (temporally replaced) —
                     this belief was RIGHT at the time it was asserted,
                     but a stronger pramāṇa has since contradicted it.
                     "The doctor told me X, then my own blood test
                     showed not-X" — the doctor's claim is bādhita.
      ARCHIVED     — pruned via SynapticPruning. Still stored, not
                     returned by default even in history walks.
    """

    CURRENT = "current"
    SUPERSEDED = "superseded"
    COEXISTS = "coexists"
    DISPUTED = "disputed"
    AMBIGUOUS = "ambiguous"
    BADHITA = "badhita"
    ARCHIVED = "archived"


@dataclass
class Belief:
    """A proposition enriched with belief-layer metadata.

    Beliefs are the derived-set layer on top of raw propositions. One
    proposition becomes one belief at assertion time; subsequent
    re-assertions of the same content reinforce (bump confidence) rather
    than create duplicates.

    Supersession is non-destructive: when B supersedes A,
    A.superseded_by appends B.id and B.supersedes appends A.id. A is
    still stored and queryable, just not returned by default.

    Attributes
    ----------
    id
        Stable unique id for this belief.
    proposition
        The textual content (the claim).
    asserted_at
        When this belief entered the system. (When the user said it.)
    observed_at
        Optional: when the referenced event occurred (if different from
        asserted_at). "I moved to Sofia last month" — asserted now,
        observed last month.
    asserted_in_session
        Opaque session identifier where it was asserted.
    source_id
        Identifier of the external source (speaker, document, session
        cluster). Used for source-independence weighting of
        reinforcements. Defaults to asserted_in_session.
    confidence
        Current confidence in [0, 1]. Starts at 1.0 for explicit user
        assertions; adjusted by reinforcement (+), decay (-), or conflict
        resolution over time.
    status
        ResolutionStatus. Derived from the supersession/dispute graph
        by the store; stored denormalised here for fast filtering.
    validity
        Temporal lifespan.
    supersedes
        BeliefIds this belief replaces.
    superseded_by
        BeliefIds that replace this belief. Non-empty means this is a
        historical/archival belief — still queryable, not current.
    coexists_with
        BeliefIds that hold simultaneously with this one. Symmetric.
    disputed_with
        BeliefIds that appear to contradict this one but have not been
        resolved via supersession. Symmetric.
    reinforced_by
        BeliefIds of later assertions that confirmed this belief without
        contradicting it. Proxy for 'number of times the user said this.'
    reinforcement_sources
        Set of distinct source_ids that have reinforced this belief.
        Used for source-independence weighting.
    reinforcement_pramanas
        Set of distinct pramāṇa types that have reinforced this belief.
        Used for epistemic-diversity weighting: a belief reinforced via
        perception AND testimony is more robust than one reinforced
        twice via the same kind of knowledge.
    source_proposition_id
        Link back to the underlying Phase-1 proposition row.
    pramana
        Source of valid knowledge for this belief (Pramana enum).
        Defaults to UNKNOWN when not specified. Auto-detection via
        linguistic cues is applied by BeliefLayer.ingest unless
        explicitly overridden.
    context
        Optional string labelling the conversational context this
        belief belongs to (e.g., "work", "health", "family"). Two
        beliefs with different contexts are NOT considered to
        contradict by default — "I'm available" in work context vs.
        personal context are both current. None = context-independent
        (applies across all contexts). v0.4 addition.
    """

    id: BeliefId
    proposition: str
    asserted_at: datetime
    asserted_in_session: str
    source_proposition_id: PropositionId
    confidence: float = 1.0
    status: ResolutionStatus = ResolutionStatus.CURRENT
    validity: Validity = field(default_factory=Validity)
    pramana: Pramana = Pramana.UNKNOWN
    observed_at: datetime | None = None
    source_id: str | None = None
    context: str | None = None
    # Saṁskāra → Vāsanā layered confidence (v0.4).
    # samskara_count: number of distinct-source reinforcements this
    # belief has received. Used to decide when to crystallise surface
    # confidence into deep (vāsanā) confidence.
    # deep_confidence: slow-moving layer below surface .confidence.
    # Rises when samskara_count crosses the establishment threshold;
    # decays at 1/10 the surface decay rate. Defaults to None until
    # established.
    samskara_count: int = 0
    deep_confidence: float | None = None
    supersedes: list[BeliefId] = field(default_factory=list)
    superseded_by: list[BeliefId] = field(default_factory=list)
    coexists_with: list[BeliefId] = field(default_factory=list)
    disputed_with: list[BeliefId] = field(default_factory=list)
    reinforced_by: list[BeliefId] = field(default_factory=list)
    reinforcement_sources: list[str] = field(default_factory=list)
    reinforcement_pramanas: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0, 1]; got {self.confidence}"
            )
        if self.source_id is None:
            self.source_id = self.asserted_in_session

    @property
    def is_current(self) -> bool:
        """A belief is current iff nothing has superseded it and it is not archived."""
        return (
            len(self.superseded_by) == 0
            and self.status != ResolutionStatus.ARCHIVED
        )

    @property
    def is_superseded(self) -> bool:
        """At least one later belief has replaced this one."""
        return len(self.superseded_by) > 0

    @property
    def is_disputed(self) -> bool:
        """This belief has unresolved contradiction with another."""
        return len(self.disputed_with) > 0

    @property
    def is_coexisting(self) -> bool:
        """This belief holds alongside related non-contradictory beliefs."""
        return len(self.coexists_with) > 0

    @property
    def is_vasana_established(self) -> bool:
        """True iff this belief's samskāra chain has crystallised into
        a deep (vāsanā) confidence. Surface confidence can now swing
        without losing the user's long-held position entirely."""
        return self.deep_confidence is not None

    @property
    def effective_confidence(self) -> float:
        """The confidence to report in most UIs: the deep confidence
        when established, otherwise the surface confidence.

        Surface confidence can dip below deep during decay and recover
        through reinforcement — the deep layer prevents the belief
        from vanishing between reinforcements."""
        if self.deep_confidence is not None:
            return max(self.confidence, self.deep_confidence)
        return self.confidence
