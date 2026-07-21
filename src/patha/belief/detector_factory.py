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
    "full-stack-v9",
    "full-stack-v10",
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
    if name == "full-stack-v9":
        # v0.11: v8 + the two fixes from the EvolutionEval held-out
        # reveal (docs/benchmarks.md). SymmetricNLI wraps the NLI core
        # ONLY (direction-dependent outer detectors stay one-way);
        # RevisionPatternDetector adds resumption / settlement /
        # arrangement families. v7/v8 remain frozen so published
        # numbers stay reproducible.
        from patha.belief.learned_supersession import LearnedSupersessionDetector
        from patha.belief.revision_patterns import RevisionPatternDetector
        from patha.belief.symmetric_detector import SymmetricContradictionDetector
        return LearnedSupersessionDetector(
            inner=NumericalAwareDetector(
                inner=RevisionPatternDetector(
                    inner=SequentialEventDetector(
                        inner=AdhyasaAwareDetector(
                            inner=SymmetricContradictionDetector(
                                inner=NLIContradictionDetector(),
                            )
                        )
                    )
                )
            ),
            threshold=0.7,
        )
    if name == "full-stack-v10":
        # v0.12: v9 + the supersession-PRECISION veto built from the
        # 113 false claims harvested from rubric-v2 artifacts (dev
        # precision 0.475 / held-out 0.230 while recall was 0.885 /
        # 1.000). RefinementVetoDetector sits OUTERMOST (it must see
        # final labels) and can only downgrade CONTRADICTS -> NEUTRAL:
        # no-shared-locus, fulfilled-intention, initiation-progress,
        # and new-regime-facet classes, with reversal-evidence KEEP
        # overrides running first. v7/v8/v9 remain frozen.
        from patha.belief.refinement_veto import RefinementVetoDetector
        return RefinementVetoDetector(inner=make_detector("full-stack-v9"))
    raise ValueError(
        f"unknown detector {name!r}; choose from {AVAILABLE_DETECTORS}"
    )


def describe_detector(name: str) -> str:
    """One-line human description of a named detector."""
    return {
        "stub": "heuristic, no model (fast, CI)",
        "full-stack-v10": (
            "full-stack-v9 + refinement veto (supersession precision: "
            "locus/intention/initiation/regime classes suppressed unless "
            "the new belief carries reversal evidence)"
        ),
        "nli": "DeBERTa-v3-large MNLI (~1.7 GB first run)",
        "adhyasa-nli": "adhyāsa lexical rewrite + NLI",
        "full-stack": "numerical + adhyāsa + NLI (v0.6)",
        "full-stack-v7": "full-stack + sequential-event supersession (v0.7)",
        "full-stack-v8": "full-stack-v7 + learned classifier, 0% FPR (v0.8)",
        "full-stack-v9": (
            "full-stack-v8 + symmetric NLI + revision patterns "
            "(resumption/settlement/arrangement) — the held-out fixes (v0.11)"
        ),
    }.get(name, "unknown detector")
