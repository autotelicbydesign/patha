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
from patha.belief.direct_answer import (
    DirectAnswer,
    DirectAnswerer,
    is_belief_lookup,
)
from patha.belief.llm_judge import (
    HybridContradictionDetector,
    LLMJudge,
    PromptLLMJudge,
    StubLLMJudge,
)
from patha.belief.ollama_judge import OllamaLLMJudge
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
from patha.belief.abhava import AbhavaInference, AbhavaKind, classify_abhava
from patha.belief.adhyasa import (
    AdhyasaResult,
    HandCuratedOntology,
    IsAOntology,
    check_superimposition,
)
from patha.belief.adhyasa_detector import AdhyasaAwareDetector

# WordNet ontology is optional — only available when nltk is installed.
# Import lazily to avoid ImportError on `from patha.belief import ...`.
try:
    from patha.belief.wordnet_ontology import WordNetOntology
except ImportError:
    WordNetOntology = None  # type: ignore[assignment,misc]
from patha.belief.counterfactual import order_sensitivity, replay_in_order
from patha.belief.pramana import detect_pramana
from patha.belief.raw_archive import RawArchive, RawTurn
from patha.belief.vritti import VrittiClass, vritti_label, vritti_of
from patha.belief.types import (
    Belief,
    BeliefId,
    ContradictionLabel,
    ContradictionResult,
    Pramana,
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
    "Pramana",
    "PropositionId",
    "ResolutionStatus",
    "Validity",
    "ValidityMode",
    # Pramāṇa detection
    "detect_pramana",
    # Abhāva (Nyāya four-fold negation)
    "AbhavaKind",
    "AbhavaInference",
    "classify_abhava",
    # Counterfactual / order-sensitive belief operations (quantum cognition)
    "replay_in_order",
    "order_sensitivity",
    # Adhyāsa (superimposition-based contradiction detection)
    "AdhyasaResult",
    "AdhyasaAwareDetector",
    "HandCuratedOntology",
    "IsAOntology",
    "WordNetOntology",  # may be None if nltk not installed
    "check_superimposition",
    # Raw archive (provenance substrate)
    "RawArchive",
    "RawTurn",
    # Vṛtti classification (Patañjali's cognitive-mode taxonomy)
    "VrittiClass",
    "vritti_of",
    "vritti_label",
    # Contradiction detection
    "ContradictionDetector",
    "NLIContradictionDetector",
    "StubContradictionDetector",
    # LLM-judge fallback (D1 Option D)
    "HybridContradictionDetector",
    "LLMJudge",
    "PromptLLMJudge",
    "StubLLMJudge",
    "OllamaLLMJudge",
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
    # Direct-answer compression (D7 Option C)
    "DirectAnswer",
    "DirectAnswerer",
    "is_belief_lookup",
    # Plasticity mechanisms (the neuroplasticity layer)
    "LongTermPotentiation",
    "LongTermDepression",
    "SynapticPruning",
    "HomeostaticRegulation",
    "HebbianAssociation",
    # Top-level layer
    "BeliefLayer",
    "BeliefQueryResult",
    "IngestAction",
    "IngestEvent",
]
