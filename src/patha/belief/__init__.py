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
from patha.belief.llm_judge import (
    HybridContradictionDetector,
    LLMJudge,
    PromptLLMJudge,
    StubLLMJudge,
)
from patha.belief.layer import (
    BeliefLayer,
    BeliefQueryResult,
    IngestAction,
    IngestEvent,
)
from patha.belief.plasticity import (
    HebbianAssociation,
    HomeostaticRegulation,
    LongTermDepression,
    LongTermPotentiation,
    SynapticPruning,
)
from patha.belief.store import BeliefStore
from patha.belief.validity_extraction import extract_validity
from patha.belief.types import (
    Belief,
    BeliefId,
    ContradictionLabel,
    ContradictionResult,
    PropositionId,
    ResolutionStatus,
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
    "ResolutionStatus",
    "Validity",
    "ValidityMode",
    # Contradiction detection
    "ContradictionDetector",
    "NLIContradictionDetector",
    "StubContradictionDetector",
    # LLM-judge fallback (D1 Option D)
    "HybridContradictionDetector",
    "LLMJudge",
    "PromptLLMJudge",
    "StubLLMJudge",
    # Store
    "BeliefStore",
    # Plasticity mechanisms (the neuroplasticity layer)
    "LongTermPotentiation",
    "LongTermDepression",
    "SynapticPruning",
    "HomeostaticRegulation",
    "HebbianAssociation",
    # Validity extraction
    "extract_validity",
    # Top-level layer
    "BeliefLayer",
    "BeliefQueryResult",
    "IngestAction",
    "IngestEvent",
]
