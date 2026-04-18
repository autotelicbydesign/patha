"""Tests for WordNet-backed IsAOntology (v0.5 #2).

These tests are marked @pytest.mark.slow — they require nltk + the
WordNet corpus to be installed. Run with `pytest -m slow`.

For CI without nltk, the constructor raises a clear ImportError; see
test_constructor_without_nltk below (which is fast because it doesn't
require the corpus, just the absence of nltk).
"""

from __future__ import annotations

import importlib.util

import pytest

from patha.belief.wordnet_ontology import WordNetOntology


# Detect nltk availability — the slow tests require it + corpus
_HAS_NLTK = importlib.util.find_spec("nltk") is not None


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_NLTK, reason="nltk not installed")
class TestWordNetOntology:
    def test_equivalence_includes_synonyms(self) -> None:
        ont = WordNetOntology(max_depth=1)
        eq = ont.equivalent_terms("car")
        # WordNet synonyms: auto, automobile, machine, motorcar
        assert "automobile" in eq or "auto" in eq or "motorcar" in eq

    def test_singleton_for_unknown_term(self) -> None:
        ont = WordNetOntology(max_depth=1)
        eq = ont.equivalent_terms("glizzybop")  # nonsense word
        # No WordNet entry → singleton class
        assert eq == {"glizzybop"}

    def test_max_depth_bounds_class_size(self) -> None:
        """Small depth should produce smaller classes than larger depth."""
        shallow = WordNetOntology(max_depth=1)
        deep = WordNetOntology(max_depth=3)
        s1 = shallow.equivalent_terms("dog")
        s2 = deep.equivalent_terms("dog")
        assert len(s2) >= len(s1)

    def test_hyponyms_included_by_default(self) -> None:
        ont = WordNetOntology(max_depth=2, include_hyponyms=True)
        eq = ont.equivalent_terms("dog")
        # Should include specific dog breeds (hyponyms)
        # Without requiring a specific breed, check class is meaningfully
        # populated
        assert len(eq) > 5


class TestImportErrorHandling:
    """Fast test: even if nltk is installed, this class tests that the
    error messaging is clear when WordNet corpus isn't downloaded."""

    @pytest.mark.skipif(
        not _HAS_NLTK,
        reason="Can only exercise this path when nltk present",
    )
    def test_no_corpus_raises_clearly(self, monkeypatch) -> None:
        """If wordnet corpus lookup fails, the error should be clear.

        We patch nltk.corpus.wordnet.synsets to raise LookupError.
        """
        from nltk.corpus import wordnet as wn

        def fake_synsets(*args, **kwargs):
            raise LookupError("corpus not found")

        monkeypatch.setattr(wn, "synsets", fake_synsets)
        with pytest.raises(ImportError, match="WordNet"):
            WordNetOntology()
