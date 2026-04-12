"""Named entity extraction for Patha proposition enrichment.

Extracts named entities from proposition text using spaCy. The entities
unlock the entity-anchored views (v5 ghana, v6 reframed) which would
otherwise degrade to their base forms.

Entity types preserved: PERSON, ORG, GPE, LOC, FAC, EVENT, PRODUCT,
WORK_OF_ART, LANGUAGE, NORP. Numeric types (DATE, TIME, MONEY, etc.)
are excluded — temporal information is handled separately by the
temporal module.

Usage::

    enricher = EntityEnricher()
    entities = enricher.extract("Alice moved to 456 Oak Ave in Seattle.")
    # => ["Alice", "456 Oak Ave", "Seattle"]

    # Batch mode for efficiency:
    batch_entities = enricher.extract_batch([
        "Bob uses Python at work.",
        "Alice lives in Portland.",
    ])
    # => [["Bob", "Python"], ["Alice", "Portland"]]
"""

from __future__ import annotations

import spacy
from spacy.language import Language


# Entity types that are useful for semantic anchoring.
# Exclude numeric types (DATE, TIME, MONEY, QUANTITY, ORDINAL, CARDINAL)
# since those are handled by the temporal module.
_ENTITY_LABELS = frozenset({
    "PERSON", "ORG", "GPE", "LOC", "FAC", "EVENT",
    "PRODUCT", "WORK_OF_ART", "LANGUAGE", "NORP",
})


class EntityEnricher:
    """Extract named entities from proposition text via spaCy.

    Parameters
    ----------
    model_name
        spaCy model to use. Default ``en_core_web_sm`` (fast, good enough
        for named entity extraction). Use ``en_core_web_trf`` for higher
        accuracy if GPU is available.
    """

    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        self._nlp: Language = spacy.load(
            model_name,
            disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"],
        )
        # Only keep NER pipe
        # (some models have different pipeline component names)

    def extract(self, text: str) -> list[str]:
        """Extract entity strings from a single text.

        Returns a deduplicated list of entity strings, preserving first-
        occurrence order.
        """
        doc = self._nlp(text)
        seen: set[str] = set()
        entities: list[str] = []
        for ent in doc.ents:
            if ent.label_ in _ENTITY_LABELS and ent.text not in seen:
                seen.add(ent.text)
                entities.append(ent.text)
        return entities

    def extract_batch(self, texts: list[str]) -> list[list[str]]:
        """Extract entities for a batch of texts efficiently.

        Uses spaCy's ``nlp.pipe()`` for batched processing, which is
        significantly faster than calling ``extract()`` in a loop.
        """
        results: list[list[str]] = []
        for doc in self._nlp.pipe(texts, batch_size=256):
            seen: set[str] = set()
            entities: list[str] = []
            for ent in doc.ents:
                if ent.label_ in _ENTITY_LABELS and ent.text not in seen:
                    seen.add(ent.text)
                    entities.append(ent.text)
            results.append(entities)
        return results


def enrich_propositions(
    prop_texts: list[str],
    enricher: EntityEnricher | None = None,
) -> list[list[str]]:
    """Convenience: extract entities for a list of proposition texts.

    Creates an ``EntityEnricher`` on first call if none is provided.
    Returns a list parallel to ``prop_texts`` where each element is
    the list of entity strings for that proposition.
    """
    if enricher is None:
        enricher = EntityEnricher()
    return enricher.extract_batch(prop_texts)
