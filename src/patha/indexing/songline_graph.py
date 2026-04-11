"""Songline multi-modal graph builder.

Inspired by Australian Aboriginal songlines — the longest continuously
transmitted information on Earth — this module binds every proposition to
five modality channels and builds edges between propositions that share a
modality. The resulting graph is the "landscape" of the songline layer:
retrieval is a weighted walk through it, returning narrative paths instead
of disconnected hits.

Modality channels
-----------------
1. **entities** — shared named entities (most specific, highest weight)
2. **temporal** — shared timestamp or temporal bucket
3. **session** — same session_id (local coherence)
4. **speaker** — same speaker
5. **topic** — same topic cluster ID

Edges are weighted by *specificity*: a shared rare entity weighs more than
a shared common speaker. The weight formula is:

    edge_weight = sum(1 / log2(2 + freq(shared_value))) per channel

where ``freq`` is the global frequency of that value across all props. Rare
values (unique entities) yield high weights; common values (frequent speaker
"alice") yield low weights. This mirrors how songlines encode rarer
landmarks with more elaborate song segments.

The graph is stored as an adjacency list ``{chunk_id: [(neighbor_id, weight,
channel), ...]}`` and can be persisted alongside the store. For now it's
in-memory; the LanceDB edge table adapter comes later.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field


@dataclass
class SonglineGraph:
    """In-memory multi-modal adjacency graph over propositions."""

    # chunk_id -> list of (neighbor_id, weight, channel_name)
    adjacency: dict[str, list[tuple[str, float, str]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    # Per-channel inverted index: channel -> value -> set of chunk_ids
    _channel_index: dict[str, dict[str, set[str]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(set))
    )
    # Global value frequency across all channels (for specificity weighting)
    _value_freq: Counter = field(default_factory=Counter)

    def node_count(self) -> int:
        """Number of unique nodes that have at least one edge."""
        return len(self.adjacency)

    def edge_count(self) -> int:
        """Total directed edges (each undirected edge counted twice)."""
        return sum(len(neighbors) for neighbors in self.adjacency.values())

    def neighbors(self, chunk_id: str) -> list[tuple[str, float, str]]:
        """Return (neighbor_id, weight, channel) for a node."""
        return self.adjacency.get(chunk_id, [])


def _specificity_weight(freq: int) -> float:
    """Higher weight for rarer values. Minimum 0.1 for very common ones."""
    return max(0.1, 1.0 / math.log2(2 + freq))


def build_songline_graph(rows: list[dict]) -> SonglineGraph:
    """Build the songline graph from ingested proposition rows.

    Each row must have at minimum:
        - chunk_id: str
        - session_id: str
        - speaker: str | None
        - timestamp: str | None
        - entities: list[str]
        - topic_cluster: int | None  (optional, may be absent)

    Parameters
    ----------
    rows
        All ingested proposition rows from the store.

    Returns
    -------
    SonglineGraph
        Multi-modal adjacency graph ready for walks.
    """
    graph = SonglineGraph()

    # Phase 1: build per-channel inverted indexes and count value frequencies.
    for row in rows:
        cid = row["chunk_id"]

        # Channel: entities
        for ent in row.get("entities", []):
            graph._channel_index["entity"][ent].add(cid)
            graph._value_freq[("entity", ent)] += 1

        # Channel: temporal (use raw timestamp string as bucket key)
        ts = row.get("timestamp")
        if ts:
            graph._channel_index["temporal"][ts].add(cid)
            graph._value_freq[("temporal", ts)] += 1

        # Channel: session
        sid = row.get("session_id", "")
        if sid:
            graph._channel_index["session"][sid].add(cid)
            graph._value_freq[("session", sid)] += 1

        # Channel: speaker
        spk = row.get("speaker")
        if spk:
            graph._channel_index["speaker"][spk].add(cid)
            graph._value_freq[("speaker", spk)] += 1

        # Channel: topic cluster
        tc = row.get("topic_cluster")
        if tc is not None:
            tc_key = str(tc)
            graph._channel_index["topic"][tc_key].add(cid)
            graph._value_freq[("topic", tc_key)] += 1

    # Phase 2: build edges. For each channel, connect all pairs of nodes
    # that share a value. Weight by specificity of the shared value.
    # To avoid O(n^2) blowup on very common values (e.g. a speaker who
    # appears in 10k props), cap the group size at MAX_GROUP.
    MAX_GROUP = 200

    for channel, value_index in graph._channel_index.items():
        for value, members in value_index.items():
            if len(members) < 2:
                continue
            freq = graph._value_freq[(channel, value)]
            weight = _specificity_weight(freq)

            member_list = sorted(members)  # deterministic order
            if len(member_list) > MAX_GROUP:
                # Keep only the first MAX_GROUP members (by alphabetical
                # chunk_id). This is a pragmatic cap — in production the
                # LanceDB adapter will use approximate neighbor joins.
                member_list = member_list[:MAX_GROUP]

            for i, cid_a in enumerate(member_list):
                for cid_b in member_list[i + 1 :]:
                    graph.adjacency[cid_a].append((cid_b, weight, channel))
                    graph.adjacency[cid_b].append((cid_a, weight, channel))

    return graph
