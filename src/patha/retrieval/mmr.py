"""Maximal Marginal Relevance (MMR) with session-diversity cap.

Standard MMR selects items that balance relevance to the query with
diversity from already-selected items. Our variant adds a per-session cap
so the final top-K doesn't concentrate on a single conversation session —
critical for LongMemEval's multi-session questions where gold evidence is
spread across sessions.

The session-diversity cap is the Aboriginal songline principle applied to
result selection: a walk should traverse *different places* in the memory
landscape, not circle the same room.
"""

from __future__ import annotations

from collections import Counter

from patha.indexing.store import cosine_similarity


def mmr_rerank(
    candidates: list[tuple[str, float, list[float]]],
    query_vec: list[float],
    k: int,
    *,
    lambda_: float = 0.7,
    session_cap: int = 2,
    session_fn: callable | None = None,
) -> list[tuple[str, float]]:
    """Select top-``k`` items via MMR with an optional per-session cap.

    Parameters
    ----------
    candidates
        List of ``(chunk_id, relevance_score, embedding)`` triples.
    query_vec
        The query embedding for relevance scoring.
    k
        Number of items to select.
    lambda_
        Trade-off: 1.0 = pure relevance, 0.0 = pure diversity.
        Default 0.7 (mild diversity bias).
    session_cap
        Maximum items from any one session in the final selection.
        Default 2. Set to a large number to disable.
    session_fn
        Callable ``(chunk_id) -> session_id``. Default parses the
        ``{session}#t{turn}#p{prop}`` format.

    Returns
    -------
    list[tuple[str, float]]
        Selected ``(chunk_id, mmr_score)`` pairs in selection order.
    """
    if not candidates or k <= 0:
        return []

    if session_fn is None:
        session_fn = lambda cid: cid.split("#")[0]

    selected: list[tuple[str, float]] = []
    selected_vecs: list[list[float]] = []
    session_counts: Counter = Counter()
    remaining = list(candidates)

    while len(selected) < k and remaining:
        best_idx = -1
        best_score = float("-inf")

        for i, (cid, rel, vec) in enumerate(remaining):
            # Session cap check
            sid = session_fn(cid)
            if session_counts[sid] >= session_cap:
                continue

            # MMR score: lambda * relevance - (1-lambda) * max_sim_to_selected
            if selected_vecs:
                max_sim = max(
                    cosine_similarity(vec, sv) for sv in selected_vecs
                )
            else:
                max_sim = 0.0

            mmr = lambda_ * rel - (1 - lambda_) * max_sim
            if mmr > best_score:
                best_score = mmr
                best_idx = i

        if best_idx < 0:
            # All remaining items are session-capped out.
            break

        cid, rel, vec = remaining.pop(best_idx)
        selected.append((cid, best_score))
        selected_vecs.append(vec)
        session_counts[session_fn(cid)] += 1

    return selected
