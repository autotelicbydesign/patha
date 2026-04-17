"""Cross-encoder reranker for the Patha retrieval pipeline.

Currently wraps sentence-transformers CrossEncoder models. The default
model (``cross-encoder/ms-marco-MiniLM-L-6-v2``) is lightweight enough
for CPU inference and provides significant lift over pure bi-encoder
ranking — typically 3-8 points on retrieval benchmarks.

The reranker conforms to the ``Reranker`` callable type expected by
``pipeline.retrieve()``: it takes (query, candidates, store) and returns
re-scored (chunk_id, score) pairs sorted descending.

Later, Qwen3-Reranker-4B (via vLLM) replaces this for the full 99%
config. The interface stays the same.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from patha.indexing.store import Store


class CrossEncoderReranker:
    """Pointwise cross-encoder reranker.

    Uses a sentence-transformers CrossEncoder to score (query, passage)
    pairs. This is the standard approach for pointwise reranking in the
    retrieve-then-rerank paradigm.

    Parameters
    ----------
    model_name
        Any model accepted by ``CrossEncoder()``. Default is
        ``cross-encoder/ms-marco-MiniLM-L-6-v2`` — fast, 6-layer, good
        quality. For higher accuracy: ``cross-encoder/ms-marco-MiniLM-L-12-v2``.
    device
        ``"cpu"``, ``"cuda"``, ``"mps"``, or ``None`` for auto-detect.
    batch_size
        Batch size for ``model.predict()``. Default 64.
    top_n
        How many candidates to return after reranking. Default 100.
        Set to 0 to return all candidates re-scored.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str | None = None,
        batch_size: int = 64,
        top_n: int = 100,
        rrf_blend: float = 0.0,
    ) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name, device=device)
        self._batch_size = batch_size
        self.top_n = top_n
        self._rrf_blend = rrf_blend  # 0.0 = pure CE, 1.0 = pure RRF

    def __call__(
        self,
        query: str,
        candidates: list[tuple[str, float]],
        store: "Store",
    ) -> list[tuple[str, float]]:
        """Rerank candidates by cross-encoder score.

        Parameters
        ----------
        query
            The raw query string.
        candidates
            ``(chunk_id, score)`` pairs from candidate generation.
        store
            The proposition store — used to look up passage text.

        Returns
        -------
        list[tuple[str, float]]
            Re-scored ``(chunk_id, ce_score)`` pairs, sorted descending.
        """
        if not candidates:
            return []

        # Build (query, passage) pairs
        pairs: list[tuple[str, str]] = []
        valid_ids: list[str] = []
        for cid, _ in candidates:
            row = store.get(cid)
            if row is not None:
                # Use v4 (jata triple) for richer context, fall back to v1
                text = row["views"].get("v4", row["views"]["v1"])["text"]
                pairs.append((query, text))
                valid_ids.append(cid)

        if not pairs:
            return candidates

        # Score all pairs
        scores = self._model.predict(
            pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
        )

        # Pair up, optionally blend with original RRF rank
        ce_scores = [float(s) for s in scores]

        if self._rrf_blend > 0.0:
            # Normalize CE scores to [0, 1] range for blending
            ce_min = min(ce_scores) if ce_scores else 0.0
            ce_max = max(ce_scores) if ce_scores else 1.0
            ce_range = ce_max - ce_min if ce_max > ce_min else 1.0

            # Build RRF rank scores: rank 1 → 1.0, rank N → ~0.0
            n = len(valid_ids)
            rrf_rank_scores = {cid: 1.0 - (i / n) for i, (cid, _) in enumerate(candidates) if cid in set(valid_ids)}

            blended = []
            for cid, ce_s in zip(valid_ids, ce_scores):
                ce_norm = (ce_s - ce_min) / ce_range
                rrf_s = rrf_rank_scores.get(cid, 0.0)
                blend = (1.0 - self._rrf_blend) * ce_norm + self._rrf_blend * rrf_s
                blended.append((cid, blend))
            scored = blended
        else:
            scored = list(zip(valid_ids, ce_scores))

        scored.sort(key=lambda x: (-x[1], x[0]))

        if self.top_n > 0:
            return scored[: self.top_n]
        return scored
