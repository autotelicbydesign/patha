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

@dataclass
class Belief:
    """A proposition enriched with belief-layer metadata.

    Beliefs are the derived-set layer on top of raw propositions. One
    proposition becomes one belief at assertion time; subsequent
    re-assertions of the same content reinforce (bump confidence) rather
    than create duplicates.

    Supersession is non-destructive in v0.1: when Belief B supersedes A,
    A.superseded_by appends B.id and B.supersedes appends A.id. A is
    still stored and queryable, just not returned by default.

    Attributes
    ----------
    id
        Stable unique id for this belief.
    proposition
        The textual content (the claim).
    asserted_at
        When this belief entered the system.
    asserted_in_session
        Opaque session identifier where it was asserted.
    confidence
        Current confidence in [0, 1]. Starts at 1.0 for explicit user
        assertions; adjusted by reinforcement (+), decay (-), or conflict
        resolution over time.
    validity
        Temporal lifespan.
    supersedes
        BeliefIds this belief replaces.
    superseded_by
        BeliefIds that replace this belief. Non-empty means this is a
        historical/archival belief — still queryable, not current.
    reinforced_by
        BeliefIds of later assertions that confirmed this belief without
        contradicting it. Proxy for "number of times the user said this."
    source_proposition_id
        Link back to the underlying Phase-1 proposition row.
    """

    id: BeliefId
    proposition: str
    asserted_at: datetime
    asserted_in_session: str
    source_proposition_id: PropositionId
    confidence: float = 1.0
    validity: Validity = field(default_factory=Validity)
    supersedes: list[BeliefId] = field(default_factory=list)
    superseded_by: list[BeliefId] = field(default_factory=list)
    reinforced_by: list[BeliefId] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0, 1]; got {self.confidence}"
            )

    @property
    def is_current(self) -> bool:
        """A belief is current iff nothing has superseded it."""
        return len(self.superseded_by) == 0

    @property
    def is_superseded(self) -> bool:
        """Inverse of is_current. Exists for readability at call sites."""
        return not self.is_current
