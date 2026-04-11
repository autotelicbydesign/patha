"""Embedder interface and a deterministic stub implementation.

The ``Embedder`` protocol is the single boundary between Patha's pipeline and
whatever model actually produces vectors. Everything upstream (ingest,
views, chunking) and downstream (store, retrieval, rerank) is written
against this protocol, so the real Qwen3-Embedding-4B wrapper drops in as
a one-class swap when GPU infrastructure is ready.

``StubEmbedder`` is pure Python, has no dependencies, and produces
deterministic pseudo-random unit vectors from a SHA-256 stream. It is not
semantic — it is designed only to let the rest of the pipeline be
wire-complete and end-to-end testable without any heavyweight deps. Same
input always gives the same vector, so exact-text search rounds-trip
perfectly, which is the property our integration tests rely on.
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns a list of strings into a list of unit vectors."""

    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input string, in order.

        Implementations must be deterministic for a given model/config so
        that re-ingestion of the same corpus produces bit-identical indexes.
        Output vectors are expected to be unit-normalized so cosine
        similarity reduces to a dot product downstream.
        """
        ...


class StubEmbedder:
    """Pure-Python deterministic fake embedder for tests and wire-complete runs.

    The same text always maps to the same unit vector (SHA-256 stream
    expanded to ``dim`` bytes, scaled to [-1, 1], then L2-normalized).
    Different texts map to different vectors with overwhelming probability.
    Semantic similarity is *not* preserved — this is a stub.
    """

    def __init__(self, dim: int = 64) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        # Expand SHA-256 into a byte stream long enough for ``dim`` floats.
        stream = bytearray()
        seed = text.encode("utf-8")
        block = hashlib.sha256(seed).digest()
        while len(stream) < self.dim:
            stream.extend(block)
            block = hashlib.sha256(block).digest()
        raw = stream[: self.dim]

        vec = [(b / 127.5) - 1.0 for b in raw]
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            # Degenerate — extremely unlikely given SHA output, but handle.
            return [0.0] * self.dim
        return [v / norm for v in vec]
