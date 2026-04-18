"""AdhyasaAwareDetector — composable wrapper that adds adhyāsa
rewrite-and-retest to any inner ContradictionDetector.

Usage:
    # NLI alone misses sushi/raw-fish. Wrap it in adhyāsa:
    inner = NLIContradictionDetector()
    detector = AdhyasaAwareDetector(inner=inner)

    # Or combine with the LLM-judge hybrid for even stronger coverage:
    hybrid = HybridContradictionDetector(
        primary=NLIContradictionDetector(),
        llm=llm_judge,
    )
    detector = AdhyasaAwareDetector(inner=hybrid)

Mechanism:
    For each (p1, p2) pair:
      1. Check superimposition via `check_superimposition`.
      2. If detected, run inner detector on (p1, p2_rewritten) where
         p2_rewritten substitutes p1's lexeme for its synonym.
      3. Otherwise, run inner detector on (p1, p2) as usual.

    The rewrite gives NLI a fair lexical anchor (same word in both
    propositions) so it can resolve the contradiction that the
    original surface form obscured.

Metrics:
    The wrapper tracks .adhyasa_rewrites (how often rewrite fired) and
    .inner_calls (total pairs sent to the inner detector). Useful for
    cost accounting and ablation comparisons.

Caveats:
    - The rewrite can sometimes introduce grammatical awkwardness
      (e.g., "I am avoiding sushi on my doctor's advice" is fine;
      "I have switched from sushi to sashimi" becomes "I have switched
      from sushi to sushi" if both terms are in the same class). The
      check_superimposition implementation filters out identity
      rewrites; anything weirder is the caller's responsibility.
    - The rewrite does NOT verify semantic preservation. If the
      ontology is wrong (e.g., 'dog' → 'wolf'), the rewrite will
      propagate that error. Use a conservative ontology.
"""

from __future__ import annotations

from patha.belief.adhyasa import (
    HandCuratedOntology,
    IsAOntology,
    check_superimposition,
)
from patha.belief.contradiction import ContradictionDetector
from patha.belief.types import ContradictionResult


class AdhyasaAwareDetector:
    """Wraps any inner detector with an adhyāsa rewrite pass.

    Satisfies the ContradictionDetector protocol so it can be passed
    anywhere an inner detector could.

    Parameters
    ----------
    inner
        The base contradiction detector (NLI, hybrid, or any protocol-
        conforming implementation).
    ontology
        Equivalence-class source. Defaults to HandCuratedOntology.
    """

    def __init__(
        self,
        inner: ContradictionDetector,
        *,
        ontology: IsAOntology | None = None,
    ) -> None:
        self._inner = inner
        self._ontology = ontology if ontology is not None else HandCuratedOntology()
        # Metrics
        self.adhyasa_rewrites = 0
        self.inner_calls = 0

    def detect(self, p1: str, p2: str) -> ContradictionResult:
        return self.detect_batch([(p1, p2)])[0]

    def detect_batch(
        self, pairs: list[tuple[str, str]]
    ) -> list[ContradictionResult]:
        if not pairs:
            return []

        # First pass: rewrite pairs where superimposition is detected
        effective_pairs: list[tuple[str, str]] = []
        for p1, p2 in pairs:
            res = check_superimposition(p1, p2, ontology=self._ontology)
            if res.superimposition_detected and res.rewritten_p2 is not None:
                self.adhyasa_rewrites += 1
                effective_pairs.append((p1, res.rewritten_p2))
            else:
                effective_pairs.append((p1, p2))

        self.inner_calls += len(effective_pairs)
        return self._inner.detect_batch(effective_pairs)
