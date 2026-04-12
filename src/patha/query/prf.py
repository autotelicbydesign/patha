"""Pseudo-Relevance Feedback (PRF) for query expansion.

Implements RM3-style expansion over BM25 top-K hits: the top-K documents
from an initial BM25 retrieval are assumed relevant, and their highest-IDF
terms are appended to the original query with a mixing weight. This
expands the query vocabulary to capture paraphrase variants that pure
embedding might miss.

This is the classic "belt and suspenders" approach: dense retrieval
handles semantic similarity, while PRF-expanded BM25 handles lexical
paraphrase coverage. Together in RRF they are complementary.

Usage::

    expanded = prf_expand(
        query="What programming languages does Bob use?",
        bm25=bm25_index,
        store=store,
        top_k=10,
        num_terms=20,
        weight=0.3,
    )
    # => "What programming languages does Bob use? python rust development ..."
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from patha.indexing.bm25_index import BM25Index
    from patha.indexing.store import Store


# Minimal stopwords — enough to filter noise from expansion terms
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "can", "could", "must", "need",
    "i", "me", "my", "we", "us", "our", "you", "your", "he", "him", "his",
    "she", "her", "it", "its", "they", "them", "their",
    "this", "that", "these", "those", "what", "which", "who", "whom",
    "where", "when", "why", "how",
    "and", "or", "but", "not", "no", "nor", "if", "then", "so", "too",
    "very", "just", "also", "of", "in", "on", "at", "to", "for", "with",
    "by", "from", "up", "about", "into", "over", "after",
})

_WORD_RE = re.compile(r"[a-zA-Z]{2,}")


def _idf(term: str, doc_count: int, total_docs: int) -> float:
    """Compute IDF with smoothing."""
    if total_docs == 0:
        return 0.0
    return math.log((total_docs - doc_count + 0.5) / (doc_count + 0.5) + 1.0)


def prf_expand(
    query: str,
    *,
    bm25: "BM25Index",
    store: "Store",
    top_k: int = 10,
    num_terms: int = 20,
    weight: float = 0.3,
) -> str:
    """Expand query via RM3-style pseudo-relevance feedback.

    Parameters
    ----------
    query
        Original query string.
    bm25
        BM25 index for initial retrieval.
    store
        Proposition store for looking up passage text.
    top_k
        Number of top BM25 hits to use as pseudo-relevant docs.
    num_terms
        Number of expansion terms to add.
    weight
        Mixing weight for expansion terms (0 = no expansion, 1 = only
        expansion terms). Default 0.3.

    Returns
    -------
    str
        Expanded query string with appended terms.
    """
    if weight <= 0 or num_terms <= 0:
        return query

    # Initial BM25 retrieval
    hits = bm25.search(query, k=top_k)
    if not hits:
        return query

    # Collect term frequencies from pseudo-relevant docs
    term_freq: Counter[str] = Counter()
    query_terms = set(_WORD_RE.findall(query.lower()))

    for chunk_id, _score in hits:
        row = store.get(chunk_id)
        if row is None:
            continue
        text = row["text"]
        words = _WORD_RE.findall(text.lower())
        for w in words:
            if w not in _STOPWORDS and w not in query_terms:
                term_freq[w] += 1

    if not term_freq:
        return query

    # Score by TF-IDF-like weighting (using doc frequency from BM25)
    total_docs = bm25.count()
    scored_terms: list[tuple[str, float]] = []
    for term, tf in term_freq.items():
        # Approximate doc frequency from the feedback set
        doc_freq = tf  # simplified: assume each occurrence is from a different doc
        idf_score = _idf(term, min(doc_freq, top_k), total_docs)
        scored_terms.append((term, tf * idf_score))

    # Sort by score descending, take top N
    scored_terms.sort(key=lambda x: -x[1])
    expansion_terms = [t for t, _ in scored_terms[:num_terms]]

    if not expansion_terms:
        return query

    # Append expansion terms to query
    expansion_str = " ".join(expansion_terms)
    return f"{query} {expansion_str}"
