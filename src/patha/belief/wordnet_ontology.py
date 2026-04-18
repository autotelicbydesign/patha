"""WordNet-backed IsAOntology for adhyāsa superimposition detection.

Drops in as a replacement for HandCuratedOntology when broad semantic
coverage is needed. Uses NLTK's WordNet corpus (which requires a
one-time `nltk.download('wordnet')`). Optional dependency — Patha core
doesn't require it.

Two modes of equivalence are computed:
  1. Synonyms: terms that share a synset (WordNet's unit of meaning)
  2. Hyponym/hypernym chain within a narrow depth: 'sushi' → 'fish'
     (is-a raw fish) up to the configured hop limit

The narrow-depth constraint is important. Without it, WordNet happily
returns "dog" as equivalent to "entity" (via many layers). For
adhyāsa detection we only want near-identity, not abstract
generalisation.

Usage:
    from patha.belief.wordnet_ontology import WordNetOntology
    from patha.belief import AdhyasaAwareDetector, NLIContradictionDetector

    ont = WordNetOntology(max_depth=2)
    detector = AdhyasaAwareDetector(
        inner=NLIContradictionDetector(),
        ontology=ont,
    )

Failure mode: if NLTK/WordNet isn't installed, constructor raises a
clear ImportError with installation instructions. Callers should catch
that and fall back to HandCuratedOntology.
"""

from __future__ import annotations

from functools import lru_cache


class WordNetOntology:
    """WordNet-backed equivalence-class source for adhyāsa detection.

    Parameters
    ----------
    max_depth
        Maximum hypernym/hyponym hops to walk when building an
        equivalence class. Default 2. Values > 3 produce very broad
        classes that include abstract ancestors (e.g., 'entity'),
        which hurts adhyāsa precision.
    include_hyponyms
        If True (default), includes hyponyms (more-specific terms)
        as well as hypernyms. For adhyāsa, both directions matter:
        'sushi' and 'raw fish' are mutual — one is a specific form
        of the other.
    """

    def __init__(
        self,
        *,
        max_depth: int = 2,
        include_hyponyms: bool = True,
    ) -> None:
        self._max_depth = max_depth
        self._include_hyponyms = include_hyponyms
        # Defer import + validate WordNet is available
        try:
            from nltk.corpus import wordnet as wn
            # Force corpus load to fail early if not downloaded
            _ = wn.synsets("test")
            self._wn = wn
        except (ImportError, LookupError) as e:
            raise ImportError(
                "WordNetOntology requires nltk + WordNet corpus. Install with:\n"
                "  pip install nltk\n"
                "  python -c \"import nltk; nltk.download('wordnet')\"\n"
                f"Underlying error: {e}"
            ) from e

    def equivalent_terms(self, term: str) -> set[str]:
        return self._equivalent_cached(term.lower().strip().replace(" ", "_"))

    @lru_cache(maxsize=1024)
    def _equivalent_cached(self, term: str) -> set[str]:
        """Compute the equivalence class for a term. Cached per instance.

        WordNet synsets are content-addressed by (lemma, POS, sense);
        computing the closure is not cheap. Cache helps.
        """
        synsets = self._wn.synsets(term)
        if not synsets:
            # No WordNet entry → term is its own singleton class
            return {term.replace("_", " ")}

        result: set[str] = set()
        for syn in synsets:
            # Add direct synonyms (lemmas of this synset)
            for lemma in syn.lemma_names():
                result.add(lemma.replace("_", " ").lower())
            # Walk up and (optionally) down by max_depth hops
            self._walk_hypernyms(syn, result, depth=0)
            if self._include_hyponyms:
                self._walk_hyponyms(syn, result, depth=0)

        # Always include the input itself
        result.add(term.replace("_", " "))
        return result

    def _walk_hypernyms(self, syn, result: set[str], depth: int) -> None:
        if depth >= self._max_depth:
            return
        for parent in syn.hypernyms():
            for lemma in parent.lemma_names():
                result.add(lemma.replace("_", " ").lower())
            self._walk_hypernyms(parent, result, depth + 1)

    def _walk_hyponyms(self, syn, result: set[str], depth: int) -> None:
        if depth >= self._max_depth:
            return
        for child in syn.hyponyms():
            for lemma in child.lemma_names():
                result.add(lemma.replace("_", " ").lower())
            self._walk_hyponyms(child, result, depth + 1)
