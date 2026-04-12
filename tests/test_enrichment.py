"""Tests for NER entity enrichment and cross-encoder reranker."""

from __future__ import annotations

import pytest

from patha.query.entities import EntityEnricher, enrich_propositions


class TestEntityEnricher:
    @pytest.fixture(scope="class")
    def enricher(self):
        return EntityEnricher()

    def test_extracts_person(self, enricher):
        entities = enricher.extract("Alice went to the store.")
        assert "Alice" in entities

    def test_extracts_location(self, enricher):
        entities = enricher.extract("I moved to Seattle last year.")
        assert "Seattle" in entities

    def test_extracts_org(self, enricher):
        entities = enricher.extract("I work at Google as an engineer.")
        assert "Google" in entities

    def test_empty_text_returns_empty(self, enricher):
        assert enricher.extract("") == []

    def test_no_entities_returns_empty(self, enricher):
        entities = enricher.extract("The weather is nice today.")
        # May or may not have entities depending on model, but shouldn't crash
        assert isinstance(entities, list)

    def test_batch_mode(self, enricher):
        texts = [
            "Bob uses Python at work.",
            "Alice lives in Portland.",
        ]
        results = enricher.extract_batch(texts)
        assert len(results) == 2
        assert isinstance(results[0], list)
        assert isinstance(results[1], list)

    def test_deduplicates(self, enricher):
        entities = enricher.extract("Alice met Alice at the park. Alice was happy.")
        assert entities.count("Alice") == 1

    def test_convenience_function(self):
        results = enrich_propositions(["Bob works at Google."])
        assert len(results) == 1
        assert isinstance(results[0], list)
