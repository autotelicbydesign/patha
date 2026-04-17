"""Patha belief layer (Phase 2).

The belief layer sits above Phase 1 retrieval and tracks how propositions
relate, update, and expire over time. It implements:

- Contradiction detection (when two propositions conflict)
- Supersession (non-destructive replacement of an old belief by a newer one)
- Temporal validity (beliefs have lifespans — permanent, dated, or decaying)

See docs/phase_2_spec.md for the design rationale and the open decisions
(D1-D7). v0.1 uses conservative defaults chosen from the literature survey.
"""

from patha.belief.types import (
    Belief,
    BeliefId,
    ContradictionLabel,
    ContradictionResult,
    Validity,
    ValidityMode,
)

__all__ = [
    "Belief",
    "BeliefId",
    "ContradictionLabel",
    "ContradictionResult",
    "Validity",
    "ValidityMode",
]
