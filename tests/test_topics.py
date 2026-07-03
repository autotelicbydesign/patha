"""Tests for topic clustering (src/patha/indexing/topics.py).

Pure — hand-built numpy vectors, no embedder, no models. Covers the
determinism guarantees the walker's on-theme gate depends on, plus the
row-adapter and the graph/walker integration points (topic edges,
topic_of accessor, the paraphrase-gate regression).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from patha.indexing.songline_graph import SonglineGraph, build_songline_graph
from patha.indexing.topics import assign_topic_clusters, cluster_topics


def _vec(direction: int, dim: int = 8, wobble: float = 0.0, seed_axis: int | None = None):
    """Unit vector along `direction`, optionally tilted slightly toward
    seed_axis so same-direction vectors are near-but-not-identical."""
    v = np.zeros(dim, dtype=np.float32)
    v[direction] = 1.0
    if wobble and seed_axis is not None:
        v[seed_axis] += wobble
    return (v / np.linalg.norm(v)).tolist()


# ─── cluster_topics core ────────────────────────────────────────────


class TestClusterTopics:
    def test_deterministic_across_runs(self):
        embs = [_vec(0), _vec(0, wobble=0.2, seed_axis=1), _vec(3),
                _vec(3, wobble=0.2, seed_axis=4), _vec(6)]
        a = cluster_topics(embs)
        b = cluster_topics(embs)
        assert a == b

    def test_groups_near_separates_far(self):
        # 3 vectors near axis-0, 2 near axis-3, 1 orthogonal singleton.
        embs = [
            _vec(0), _vec(0, wobble=0.15, seed_axis=1), _vec(0, wobble=0.15, seed_axis=2),
            _vec(3), _vec(3, wobble=0.15, seed_axis=4),
            _vec(6),
        ]
        labels = cluster_topics(embs, similarity_threshold=0.55)
        assert labels[0] == labels[1] == labels[2] == 0  # first-occurrence label
        assert labels[3] == labels[4] == 1
        assert labels[5] is None  # singleton → min_cluster_size filter

    def test_permutation_consistent_partition(self):
        embs = [
            _vec(0), _vec(0, wobble=0.15, seed_axis=1),
            _vec(3), _vec(3, wobble=0.15, seed_axis=4),
        ]
        perm = [2, 0, 3, 1]
        labels_orig = cluster_topics(embs)
        labels_perm = cluster_topics([embs[i] for i in perm])

        def partition(labels, index_map):
            groups: dict = {}
            for pos, lab in enumerate(labels):
                if lab is None:
                    continue
                groups.setdefault(lab, set()).add(index_map[pos])
            return {frozenset(g) for g in groups.values()}

        assert partition(labels_orig, list(range(4))) == partition(labels_perm, perm)

    def test_threshold_extremes(self):
        embs = [_vec(0), _vec(0, wobble=0.15, seed_axis=1), _vec(3)]
        # Impossibly high similarity bar → everything a singleton → None
        assert cluster_topics(embs, similarity_threshold=0.999) == [None, None, None]
        # Zero bar → anything with strictly positive similarity merges.
        # (Exactly-orthogonal vectors sit at distance 1.0 = the cutoff
        # and stay separate; tilt the third vector slightly toward
        # axis-0 so all pairwise sims are > 0.)
        embs_pos = [_vec(0), _vec(0, wobble=0.15, seed_axis=1),
                    _vec(3, wobble=0.1, seed_axis=0)]
        labels = cluster_topics(embs_pos, similarity_threshold=0.0)
        assert labels == [0, 0, 0]

    def test_min_cluster_size(self):
        embs = [_vec(0), _vec(0, wobble=0.15, seed_axis=1), _vec(3)]
        labels = cluster_topics(embs, min_cluster_size=3)
        assert labels == [None, None, None]

    def test_empty_and_single(self):
        assert cluster_topics([]) == []
        assert cluster_topics([_vec(0)]) == [None]

    def test_unnormalized_input_same_partition(self):
        embs = [_vec(0), _vec(0, wobble=0.15, seed_axis=1), _vec(3)]
        fat = [(np.array(e) * 7.3).tolist() for e in embs]
        assert cluster_topics(embs) == cluster_topics(fat)


# ─── Row adapter ────────────────────────────────────────────────────


class TestAssignTopicClusters:
    def test_writes_rows_in_place(self):
        rows = [
            {"chunk_id": "a", "views": {"v1": {"embedding": _vec(0)}}},
            {"chunk_id": "b", "views": {"v1": {"embedding": _vec(0, wobble=0.15, seed_axis=1)}}},
            {"chunk_id": "c", "views": {"v1": {"embedding": _vec(4)}}},
            {"chunk_id": "d"},  # no views at all
        ]
        labels = assign_topic_clusters(rows)
        assert rows[0]["topic_cluster"] == 0
        assert rows[1]["topic_cluster"] == 0
        assert rows[2]["topic_cluster"] is None  # singleton
        assert rows[3]["topic_cluster"] is None  # missing view
        assert labels == [0, 0, None, None]


# ─── Graph integration ──────────────────────────────────────────────


def _row(cid, session, ts, cluster):
    return {
        "chunk_id": cid, "session_id": session, "timestamp": ts,
        "entities": [], "speaker": None, "topic_cluster": cluster,
    }


class TestTopicChannelInGraph:
    def test_topic_only_pair_gets_topic_edge(self):
        # Different sessions, different timestamps, no entities — the
        # ONLY thing shared is the topic cluster.
        rows = [
            _row("c1", "s1", "2026-01-01T09:00:00", 0),
            _row("c2", "s2", "2026-02-01T09:00:00", 0),
            _row("c3", "s3", "2026-03-01T09:00:00", None),
        ]
        g = build_songline_graph(rows)
        channels_c1_c2 = [
            ch for nbr, w, ch in g.neighbors("c1") if nbr == "c2"
        ]
        assert "topic" in channels_c1_c2
        assert g.neighbors("c3") == []  # None cluster → no topic edge

    def test_topic_of_accessor(self):
        rows = [
            _row("c1", "s1", "2026-01-01T09:00:00", 0),
            _row("c2", "s2", "2026-02-01T09:00:00", 0),
        ]
        g = build_songline_graph(rows)
        assert g.topic_of("c1") == g.topic_of("c2") is not None
        assert g.topic_of("nope") is None
        # topic-free graph
        g2 = SonglineGraph(
            adjacency=defaultdict(list),
            _channel_index=defaultdict(lambda: defaultdict(set)),
        )
        assert g2.topic_of("c1") is None
