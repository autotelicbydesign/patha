"""BM25 sparse index interface and a pure-Python implementation.

Two implementations will exist:

- ``SimpleBM25`` (this module) — pure-Python TF-IDF-weighted BM25 using
  only stdlib. Good enough for unit tests and small corpora. No deps.
- ``BM25sIndex`` (later) — wrapper around the ``bm25s`` library for
  production speed on LongMemEval-scale corpora.

Both conform to the same ``BM25Index`` protocol so the retrieval layer
is backend-agnostic.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Protocol, runtime_checkable


@runtime_checkable
class BM25Index(Protocol):
    """Anything that supports BM25-style keyword search over a text corpus."""

    def add(self, chunk_id: str, text: str) -> None:
        """Index a single document."""
        ...

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Return top-``k`` ``(chunk_id, score)`` pairs. Higher is better."""
        ...

    def count(self) -> int:
        """Number of indexed documents."""
        ...


# Simple whitespace + punctuation tokenizer. Lowercases. Strips tokens
# shorter than 2 chars. No stemming — the stub is intentionally naive.
_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

_STOPWORDS = frozenset(
    {
        "a", "an", "the", "is", "it", "of", "in", "to", "and", "or",
        "for", "on", "at", "by", "be", "as", "do", "if", "so", "no",
        "not", "are", "was", "were", "been", "has", "had", "have",
        "will", "can", "may", "this", "that", "with", "from",
    }
)


def _tokenize(text: str) -> list[str]:
    return [
        t.lower()
        for t in _TOKEN_RE.findall(text)
        if len(t) >= 2 and t.lower() not in _STOPWORDS
    ]


class SimpleBM25:
    """Pure-Python Okapi BM25 implementation for tests and small corpora.

    Parameters
    ----------
    k1 : float
        Term frequency saturation. Default 1.5.
    b : float
        Document length normalization. Default 0.75.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._docs: dict[str, Counter] = {}  # chunk_id -> term counts
        self._doc_lens: dict[str, int] = {}
        self._df: Counter = Counter()  # document frequency per term
        self._avg_dl: float = 0.0

    def add(self, chunk_id: str, text: str) -> None:
        tokens = _tokenize(text)
        tf = Counter(tokens)
        # If re-adding same chunk_id, undo old df counts first.
        if chunk_id in self._docs:
            for term in self._docs[chunk_id]:
                self._df[term] -= 1
        self._docs[chunk_id] = tf
        self._doc_lens[chunk_id] = len(tokens)
        for term in tf:
            self._df[term] += 1
        total = sum(self._doc_lens.values())
        self._avg_dl = total / len(self._docs) if self._docs else 0.0

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        if k <= 0 or not self._docs:
            return []
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []

        n = len(self._docs)
        scores: list[tuple[str, float]] = []

        for cid, tf in self._docs.items():
            score = 0.0
            dl = self._doc_lens[cid]
            for qt in q_tokens:
                if qt not in tf:
                    continue
                f = tf[qt]
                df = self._df.get(qt, 0)
                # IDF with +0.5 smoothing
                idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
                num = f * (self.k1 + 1)
                denom = f + self.k1 * (1 - self.b + self.b * dl / self._avg_dl)
                score += idf * num / denom
            if score > 0:
                scores.append((cid, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    def count(self) -> int:
        return len(self._docs)
