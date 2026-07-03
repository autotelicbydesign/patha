"""Topic clustering — populates the songline graph's topic channel.

The songline graph has always *read* ``row["topic_cluster"]`` when
building its topic channel (`songline_graph.py`), but nothing ever
wrote it — the channel was a placeholder. This module fills it.

Why it matters (dogfood findings F4/F5, `docs/phase_4_dogfood.md`):
abstract themes ("agency", "memory", "vedic") can never appear in the
entity channel — spaCy NER only tags proper nouns — so the narrative
walk had no on-theme *graph structure* to traverse for exactly the
questions it exists to answer. Topic clusters give paraphrased,
same-theme propositions shared edges and give the walker a
cluster-membership signal that is tighter than substring matching.

Design decisions (validated against the codebase before writing):

- **Reuse the v1 ("pada") view embedding.** `build_phase1_indexes`
  already embeds every proposition across the Vedic views; v1 is the
  proposition-alone vector (v2–v4 are contaminated by neighboring
  text). Clustering reads it — NO second embedding pass.
- **Deterministic, no fixed k.** sklearn AgglomerativeClustering with
  average linkage + a cosine distance threshold. No RNG anywhere.
  Labels are canonicalized by first occurrence in row order, so the
  same input always produces the same output (and near-tie merge-order
  differences across sklearn versions can't leak into labels).
- **Average linkage, not single/connected-components** — resists
  transitive chaining into mega-clusters, which matters because the
  walker's on-theme gate trusts cluster membership.
- **Labels are build-local.** They are never persisted or compared
  across rebuilds; the bridge rebuilds wholesale on invalidate.
"""

from __future__ import annotations

import sys
from collections import Counter

import numpy as np

# O(n²) distance-matrix guard. Personal stores are hundreds to a few
# thousand beliefs; beyond this we skip clustering (fail-open: the
# graph simply lacks topic edges, which is today's behavior anyway).
_MAX_N = 5000


def cluster_topics(
    embeddings,
    *,
    similarity_threshold: float = 0.55,
    min_cluster_size: int = 2,
) -> list[int | None]:
    """Cluster unit-ish vectors into topics. Returns one label per input
    (``None`` = unclustered / singleton).

    Parameters
    ----------
    embeddings
        list of vectors (or 2-D array). Re-normalized defensively, so
        un-normalized inputs cluster identically.
    similarity_threshold
        Clusters keep merging while their average pairwise cosine
        similarity stays above this. 0.55 is a MiniLM-tuned default:
        paraphrases land ~0.6–0.85, same-topic-different-fact ~0.4–0.65,
        unrelated <0.3. Higher → more, smaller clusters (fails safe:
        everything a singleton = today's no-topic-channel behavior).
    min_cluster_size
        Clusters smaller than this become ``None`` — a size-1 "cluster"
        can't produce a graph edge or be shared with an anchor anyway.
    """
    n = len(embeddings)
    if n == 0:
        return []
    if n == 1:
        return [None]
    if n > _MAX_N:
        print(
            f"[topics] {n} rows exceeds _MAX_N={_MAX_N}; skipping topic "
            f"clustering (graph will lack topic edges).",
            file=sys.stderr,
        )
        return [None] * n

    X = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X = X / norms

    from sklearn.cluster import AgglomerativeClustering

    raw = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=1.0 - similarity_threshold,
    ).fit_predict(X)

    # Drop under-sized clusters BEFORE canonicalization so the surviving
    # labels are dense and first-occurrence-ordered.
    sizes = Counter(int(r) for r in raw)
    canonical: dict[int, int] = {}
    out: list[int | None] = []
    for r in raw:
        r = int(r)
        if sizes[r] < min_cluster_size:
            out.append(None)
            continue
        if r not in canonical:
            canonical[r] = len(canonical)
        out.append(canonical[r])
    return out


def assign_topic_clusters(
    rows: list[dict],
    *,
    view: str = "v1",
    similarity_threshold: float = 0.55,
    min_cluster_size: int = 2,
) -> list[int | None]:
    """Cluster rows by their ``views[view]["embedding"]`` and write
    ``row["topic_cluster"]`` in place. Returns the labels.

    Rows missing the view (or its embedding) get ``None`` and are
    excluded from the clustering itself, so one malformed row can't
    shift everyone else's labels.
    """
    idx_with_emb: list[int] = []
    embs: list = []
    for i, row in enumerate(rows):
        emb = (row.get("views") or {}).get(view, {}).get("embedding")
        if emb is not None and len(emb) > 0:
            idx_with_emb.append(i)
            embs.append(emb)

    labels_dense = cluster_topics(
        embs,
        similarity_threshold=similarity_threshold,
        min_cluster_size=min_cluster_size,
    )

    labels: list[int | None] = [None] * len(rows)
    for i, lab in zip(idx_with_emb, labels_dense):
        labels[i] = lab
    for row, lab in zip(rows, labels):
        row["topic_cluster"] = lab
    return labels


__all__ = ["cluster_topics", "assign_topic_clusters"]
