"""Proposition store interface and an in-memory implementation.

The ``Store`` protocol is the single interface between Patha's ingestion
and retrieval paths and whatever persistence layer actually holds the
indexed rows. Two implementations will exist:

- ``InMemoryStore`` (this module) — pure-Python dict-backed store used
  for tests and wire-complete runs on small corpora.
- ``LanceStore`` (later) — LanceDB-backed production store with columnar
  storage, payload filtering, and native hybrid search.

Both conform to the same protocol so every caller upstream is persistence-
agnostic. Row schema is a plain dict with the following keys:

    {
        "chunk_id":    str,            # "{session_id}#t{turn}#p{prop}"
        "session_id":  str,
        "turn_idx":    int,
        "prop_idx":    int,
        "text":        str,            # verbatim proposition
        "speaker":     str | None,
        "timestamp":   str | None,
        "entities":    list[str],      # may be empty
        "views": {                     # 7 Vedic patha views
            "v1": {"text": str, "embedding": list[float]},
            ...
            "v7": {"text": str, "embedding": list[float]},
        },
    }
"""

from __future__ import annotations

import math
from typing import Iterable, Iterator, Protocol, runtime_checkable


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors. Zero if either is degenerate."""
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} vs {len(b)}")
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


@runtime_checkable
class Store(Protocol):
    """Persistence interface for proposition rows and their multi-view embeddings."""

    def upsert(self, rows: list[dict]) -> None:
        """Insert or replace rows by ``chunk_id``."""
        ...

    def get(self, chunk_id: str) -> dict | None:
        """Return a single row by chunk_id, or ``None`` if absent."""
        ...

    def all_rows(self) -> Iterable[dict]:
        """Iterate over every row in the store. Order is implementation-defined."""
        ...

    def count(self) -> int:
        """Number of rows currently in the store."""
        ...

    def search_view(
        self,
        view: str,
        query_vec: list[float],
        k: int,
    ) -> list[tuple[str, float]]:
        """Return top-``k`` ``(chunk_id, score)`` pairs against a single view.

        ``view`` is one of ``v1`` .. ``v7``. Score is cosine similarity; higher
        is better. Ties are broken by insertion order (deterministic).
        """
        ...


class InMemoryStore:
    """Dict-backed store for tests and wire-complete small-corpus runs.

    Not suitable for LongMemEval-scale corpora — use ``LanceStore`` for that.
    This implementation is O(n) per search call, which is fine for unit tests
    and the 10-session smoke fixture but nothing larger.
    """

    def __init__(self) -> None:
        # Insertion-ordered dict; Python 3.7+ guarantees order preservation.
        self._rows: dict[str, dict] = {}

    def upsert(self, rows: list[dict]) -> None:
        for row in rows:
            cid = row["chunk_id"]
            self._rows[cid] = row

    def get(self, chunk_id: str) -> dict | None:
        return self._rows.get(chunk_id)

    def all_rows(self) -> Iterator[dict]:
        return iter(self._rows.values())

    def count(self) -> int:
        return len(self._rows)

    def search_view(
        self,
        view: str,
        query_vec: list[float],
        k: int,
    ) -> list[tuple[str, float]]:
        if k <= 0:
            return []
        scored: list[tuple[str, float]] = []
        for cid, row in self._rows.items():
            emb = row["views"][view]["embedding"]
            scored.append((cid, cosine_similarity(query_vec, emb)))
        # Stable sort on (-score, insertion_order). Python's sort is stable,
        # and we iterated ``self._rows`` in insertion order above, so a
        # simple descending-score sort preserves insertion order on ties.
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]
