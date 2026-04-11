"""Songline walker: multi-hop narrative traversal from anchor chunks.

At query time, after the reranker has produced a scored candidate list:

1. Take the top-N chunks as **anchors** (default N=3).
2. From each anchor, run a 3-hop weighted walk through the songline graph,
   preferring edges along *different* modality channels at each hop
   (entity -> temporal -> topic, or whatever permutation scores best).
3. Union the walk nodes back into the candidate pool with a walk-score bonus.

This is the Aboriginal pillar of Patha: retrieval is a *path* through the
landscape, not a point lookup. Multi-session questions on LongMemEval have
gold evidence spread across sessions that no single-vector retrieval can
reach — a songline walk *can*, because it traverses shared entities and
temporal anchors to discover related propositions the original query didn't
directly match.

The walker is a pure function with no side effects. It takes the graph and
returns augmented candidate lists.
"""

from __future__ import annotations

from patha.indexing.songline_graph import SonglineGraph


def _walk_one_anchor(
    anchor_id: str,
    graph: SonglineGraph,
    hops: int,
    max_branch: int,
) -> list[tuple[str, float, list[str]]]:
    """Walk from a single anchor, returning discovered nodes.

    At each hop, we pick the top ``max_branch`` neighbors by edge weight,
    preferring channels not yet used on this path (the "different modality
    at each hop" heuristic). This prevents the walk from circling within a
    single session or speaker cluster.

    Returns
    -------
    list[tuple[str, float, list[str]]]
        Each entry is ``(chunk_id, cumulative_path_weight, [channels_used])``.
        The anchor itself is NOT included.
    """
    # BFS-like walk with channel diversity preference.
    # frontier: list of (chunk_id, cumulative_weight, channels_used_set, channels_used_list)
    frontier = [(anchor_id, 0.0, set(), [])]
    visited = {anchor_id}
    discovered: list[tuple[str, float, list[str]]] = []

    for _hop in range(hops):
        next_frontier = []
        for current_id, cum_weight, used_channels, channel_list in frontier:
            neighbors = graph.neighbors(current_id)
            if not neighbors:
                continue

            # Score each neighbor: base edge weight + diversity bonus for
            # channels not yet used on this path.
            scored = []
            for nid, edge_w, channel in neighbors:
                if nid in visited:
                    continue
                diversity_bonus = 0.5 if channel not in used_channels else 0.0
                scored.append((nid, edge_w + diversity_bonus, channel, edge_w))

            # Sort by score descending, take top max_branch
            scored.sort(key=lambda x: x[1], reverse=True)
            for nid, _combined, channel, raw_w in scored[:max_branch]:
                if nid in visited:
                    continue
                visited.add(nid)
                new_cum = cum_weight + raw_w
                new_channels = used_channels | {channel}
                new_channel_list = channel_list + [channel]
                discovered.append((nid, new_cum, new_channel_list))
                next_frontier.append((nid, new_cum, new_channels, new_channel_list))

        frontier = next_frontier

    return discovered


def songline_walk(
    reranked: list[tuple[str, float]],
    graph: SonglineGraph,
    *,
    num_anchors: int = 3,
    hops: int = 3,
    max_branch: int = 5,
    walk_bonus: float = 0.05,
) -> list[tuple[str, float]]:
    """Augment a reranked candidate list with songline walk discoveries.

    Parameters
    ----------
    reranked
        Scored candidate list from the reranker: ``(chunk_id, score)``,
        sorted descending by score.
    graph
        The songline multi-modal graph built at ingest time.
    num_anchors
        Number of top candidates to use as walk starting points. Default 3.
    hops
        Number of hops per walk. Default 3.
    max_branch
        Maximum neighbors to follow per hop. Default 5.
    walk_bonus
        Score bonus added to walk-discovered nodes per unit of path weight.
        This is intentionally small — the walk is meant to *surface*
        candidates that the reranker then scores, not to override reranker
        scores. Default 0.05.

    Returns
    -------
    list[tuple[str, float]]
        Merged candidate list: original reranked + walk discoveries.
        Walk-discovered nodes get ``walk_bonus * path_weight`` added to
        their score. Nodes already in reranked keep their original score
        (or whichever is higher). Sorted descending by score.
    """
    # Start with existing scores
    scores: dict[str, float] = {cid: score for cid, score in reranked}

    # Walk from top-N anchors
    anchors = reranked[:num_anchors]
    for anchor_id, _anchor_score in anchors:
        discovered = _walk_one_anchor(anchor_id, graph, hops, max_branch)
        for nid, path_weight, _channels in discovered:
            bonus = walk_bonus * path_weight
            if nid in scores:
                # Already in candidate list — keep max of existing and bonus
                scores[nid] = max(scores[nid], scores[nid] + bonus)
            else:
                # New discovery — give it the bonus as its base score
                scores[nid] = bonus

    # Sort descending by score, alphabetical tiebreak for determinism
    merged = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return merged
