"""Patha belief layer (Phase 2).

The belief layer sits above Phase 1 retrieval and tracks how propositions
relate, update, and expire over time. It implements:

- Contradiction detection (when two propositions conflict)
- Supersession (non-destructive replacement of an old belief by a newer one)
- Temporal validity (beliefs have lifespans — permanent, dated, or decaying)

See docs/phase_2_spec.md for the design rationale and the open decisions
(D1-D7). v0.1 uses conservative defaults chosen from the literature survey.
"""

from patha.belief.contradiction import (
    ContradictionDetector,
    NLIContradictionDetector,
    StubContradictionDetector,
)
from patha.belief.layer import (
    BeliefLayer,
    BeliefQueryResult,
    IngestAction,
    IngestEvent,
)
from patha.belief.store import BeliefStore
from patha.belief.types import (
    Belief,
    BeliefId,
    ContradictionLabel,
    ContradictionResult,
    PropositionId,
    Validity,
    ValidityMode,
)

__all__ = [
    # Types
    "Belief",
    "BeliefId",
    "ContradictionLabel",
    "ContradictionResult",
    "PropositionId",
    "Validity",
    "ValidityMode",
    # Contradiction detection
    "ContradictionDetector",
    "NLIContradictionDetector",
    "StubContradictionDetector",
    # Store
    "BeliefStore",
    # Top-level layer
    "BeliefLayer",
    "BeliefQueryResult",
    "IngestAction",
    "IngestEvent",
]
