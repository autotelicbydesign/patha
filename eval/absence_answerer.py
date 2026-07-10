"""AbsenceEval adapter for the production anupalabdhi path.

Bridges the eval's answerer contract to the real implementation:
ingest the scenario into a fresh Memory, run question detection +
qualified search, map cited belief ids back to proposition indices
(the EvolutionEval belief_id→index pattern).

Phase 1 stays OFF — the absence logic's guarantee is the exhaustive
store scan, not retrieval, so the adapter proves the primitive alone.

Usage:
    uv run python -m eval.absence_eval \\
        --data eval/absence_data/dev_scenarios.jsonl \\
        --answerer eval.absence_answerer:memory_answerer
"""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path


_EMBEDDER = None


def _similarity_fn(question: str, texts: list[str]) -> list[float]:
    """Cosine similarity via the shared MiniLM embedder (lazy, cached
    across scenarios). Used for contrast ranking only — never verdicts."""
    global _EMBEDDER
    if _EMBEDDER is None:
        from patha.models.embedder_st import SentenceTransformerEmbedder
        _EMBEDDER = SentenceTransformerEmbedder()
    import numpy as np
    vecs = _EMBEDDER.embed([question] + texts)
    q, rest = np.asarray(vecs[0]), np.asarray(vecs[1:])
    qn = q / (np.linalg.norm(q) or 1.0)
    rn = rest / np.clip(np.linalg.norm(rest, axis=1, keepdims=True), 1e-9, None)
    return list(rn @ qn)


def memory_answerer(scenario: dict, question: dict, *, detector: str = "stub") -> dict:
    import patha
    from patha.belief.anupalabdhi import (
        answer_absence,
        detect_absence_question,
    )

    with tempfile.TemporaryDirectory(prefix="absence-answerer-") as td:
        mem = patha.Memory(
            path=Path(td) / "beliefs.jsonl",
            detector=detector,
            enable_phase1=False,
        )
        belief_to_idx: dict[str, int] = {}
        for i, prop in enumerate(scenario["propositions"]):
            ev = mem.remember(
                prop["text"],
                asserted_at=datetime.fromisoformat(prop["asserted_at"]),
                session_id=prop.get("session"),
                source_id=f"abs:{scenario['id']}#{i}",
            )
            bid = ev["belief_id"] if isinstance(ev, dict) else ev.new_belief.id
            belief_to_idx[bid] = i

        qi = detect_absence_question(question["q"])
        if qi is None:
            return {
                "route": "retrieval",  # any non-absence route; controls pass
                "verdict": None,
                "kind": None,
                "locus": None,
                "cited_indices": [],
            }
        result = answer_absence(
            qi, store=mem._patha.belief_layer.store,
            similarity_fn=_similarity_fn,
        )
        return {
            "route": "absence",
            "verdict": result.verdict,
            "kind": result.kind.value,
            "locus": result.locus,
            "cited_indices": sorted(
                belief_to_idx[b] for b in result.contrast_ids
                if b in belief_to_idx
            ),
        }
