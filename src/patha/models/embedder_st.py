"""Sentence-transformers embedder wrapper.

Wraps any ``sentence-transformers`` model behind the ``Embedder`` protocol.
Used as the first real (non-stub) embedder for baseline measurement before
moving to Qwen3-Embedding-4B via vLLM.

Default model: ``all-MiniLM-L6-v2`` — 384-dim, runs on CPU in ~1ms/sentence,
good enough for an honest first R@5 number on LongMemEval S.
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbedder:
    """Embedder backed by a sentence-transformers model.

    Conforms to the ``Embedder`` protocol: exposes ``dim`` and ``embed()``.
    Vectors are L2-normalized by default (sentence-transformers default).

    Parameters
    ----------
    model_name
        Any model name accepted by ``SentenceTransformer()``. Default
        ``all-MiniLM-L6-v2`` (384-dim, very fast on CPU).
    device
        ``"cpu"``, ``"cuda"``, ``"mps"``, or ``None`` for auto-detect.
    batch_size
        Batch size for ``model.encode()``. Default 256.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str | None = None,
        batch_size: int = 256,
    ) -> None:
        self._model = SentenceTransformer(model_name, device=device)
        self._batch_size = batch_size
        self.dim = self._model.get_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return [vec.tolist() for vec in embeddings]
