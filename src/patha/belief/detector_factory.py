"""Shared detector factory.

CLI, MCP server, Streamlit viewer, and eval harnesses all need to
construct ContradictionDetectors by name. Centralising the factory
here means every entry point names detectors identically.

Usage:
    from patha.belief.detector_factory import make_detector

    det = make_detector("full-stack-v7")   # production default
    det = make_detector("stub")            # fast CI default

Detectors available:
    stub            — heuristic only, no model downloads
    nli             — DeBERTa-v3-large MNLI (~1.7 GB on first run)
    adhyasa-nli     — adhyāsa rewrite pre-pass + NLI
    full-stack      — numerical + adhyāsa + NLI (v0.6)
    full-stack-v7   — full-stack + sequential-event detector (v0.7)
"""

from __future__ import annotations

from patha.belief.adhyasa_detector import AdhyasaAwareDetector
from patha.belief.contradiction import (
    ContradictionDetector,
    NLIContradictionDetector,
    StubContradictionDetector,
)
from patha.belief.numerical_detector import NumericalAwareDetector
from patha.belief.sequential_detector import SequentialEventDetector


AVAILABLE_DETECTORS: tuple[str, ...] = (
    "stub",
    "nli",
    "adhyasa-nli",
    "full-stack",
    "full-stack-v7",
    "full-stack-v8",
)


def make_detector(name: str) -> ContradictionDetector:
    """Construct a named detector. Raises ValueError on unknown name."""
    if name == "stub":
        return StubContradictionDetector()
    if name == "nli":
        return NLIContradictionDetector()
    if name == "adhyasa-nli":
        return AdhyasaAwareDetector(inner=NLIContradictionDetector())
    if name == "full-stack":
        return NumericalAwareDetector(
            inner=AdhyasaAwareDetector(inner=NLIContradictionDetector())
        )
    if name == "full-stack-v7":
        return NumericalAwareDetector(
            inner=SequentialEventDetector(
                inner=AdhyasaAwareDetector(
                    inner=NLIContradictionDetector()
                )
            )
        )
    if name == "full-stack-v8":
        # v0.8: wraps full-stack-v7 with the learned supersession
        # classifier as an additive check. At threshold 0.7 the
        # classifier only fires when very confident, letting the
        # regex + NLI stack handle the uncertain band. Bundled
        # trained model lives at patha/belief/_models/.
        from patha.belief.learned_supersession import LearnedSupersessionDetector
        return LearnedSupersessionDetector(
            inner=NumericalAwareDetector(
                inner=SequentialEventDetector(
                    inner=AdhyasaAwareDetector(
                        inner=NLIContradictionDetector()
                    )
                )
            ),
            threshold=0.7,
        )
    raise ValueError(
        f"unknown detector {name!r}; choose from {AVAILABLE_DETECTORS}"
    )


def describe_detector(name: str) -> str:
    """One-line human description of a named detector."""
    return {
        "stub": "heuristic, no model (fast, CI)",
        "nli": "DeBERTa-v3-large MNLI (~1.7 GB first run)",
        "adhyasa-nli": "adhyāsa lexical rewrite + NLI",
        "full-stack": "numerical + adhyāsa + NLI (v0.6)",
        "full-stack-v7": "full-stack + sequential-event supersession (v0.7)",
        "full-stack-v8": "full-stack-v7 + learned classifier, 0% FPR (v0.8)",
    }.get(name, "unknown detector")
