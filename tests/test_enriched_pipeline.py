"""Integration test for the enriched pipeline (NER + temporal views)."""

from __future__ import annotations

import pytest

from patha.chunking.propositionizer import propositionize
from patha.chunking.views import VIEW_NAMES, build_views
from patha.indexing.ingest import ingest_session
from patha.indexing.store import InMemoryStore
from patha.models.embedder import StubEmbedder
from patha.query.entities import EntityEnricher


class TestEnrichedIngest:
    """Test that NER enrichment flows through ingest to create real v5/v6 views."""

    @pytest.fixture(scope="class")
    def enricher(self):
        return EntityEnricher()

    def test_enriched_views_differ_from_base(self, enricher):
        """When entities are found, v5 and v6 should differ from v4 and v1."""
        props = propositionize(
            "Alice works at Google in Seattle.",
            session_id="s1", turn_idx=0,
        )
        entities = enricher.extract_batch([p.text for p in props])
        views = build_views(props, entities=entities)

        # If entities were found, v5 should contain entity markers
        if any(entities):
            for i, ents in enumerate(entities):
                if ents:
                    assert views[i]["v5"] != views[i]["v4"], \
                        "v5 (ghana) should differ from v4 (jata) when entities present"
                    assert views[i]["v6"] != views[i]["v1"], \
                        "v6 (reframed) should differ from v1 (pada) when entities present"

    def test_enriched_ingest_stores_entities(self, enricher):
        """Ingest with NER enricher should populate entities in the row."""
        store = InMemoryStore()
        embedder = StubEmbedder(dim=8)

        ids = ingest_session(
            [{"text": "Bob uses Python at Google.", "speaker": "user"}],
            session_id="s1",
            store=store,
            embedder=embedder,
            entity_enricher=enricher,
        )

        assert len(ids) >= 1
        row = store.get(ids[0])
        assert row is not None
        # Entities should be populated
        assert isinstance(row["entities"], list)
        # At minimum, "Bob" or "Google" or "Python" should be extracted
        all_entities = row["entities"]
        found_any = any(
            e in all_entities
            for e in ["Bob", "Google", "Python"]
        )
        assert found_any, f"Expected NER entities, got: {all_entities}"

    def test_temporal_view_with_timestamp(self):
        """v7 should include the timestamp when provided."""
        props = propositionize(
            "I moved to a new apartment.",
            session_id="s1", turn_idx=0, timestamp="2026-01-15",
        )
        views = build_views(
            props,
            timestamps=["2026-01-15"] * len(props),
        )
        for v in views:
            assert "2026-01-15" in v["v7"], \
                f"v7 should contain timestamp, got: {v['v7']}"

    def test_temporal_view_degrades_without_timestamp(self):
        """v7 should degrade to v1 when no timestamp provided."""
        props = propositionize(
            "The weather is nice.",
            session_id="s1", turn_idx=0,
        )
        views = build_views(props)
        for v in views:
            assert v["v7"] == v["v1"], \
                "v7 should degrade to v1 without timestamp"
