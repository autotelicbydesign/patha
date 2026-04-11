"""Retrieval metrics for LongMemEval evaluation.

Computes Recall@K at two granularities, matching the official LongMemEval
evaluation protocol:

- **Turn-level R@K**: each retrieved chunk maps to a turn; a hit is when
  any gold turn appears in the top-K retrieved turns.
- **Session-level R@K**: each retrieved chunk maps to a session; a hit is
  when any gold session appears in the top-K retrieved sessions (deduped).

LongMemEval reports turn-level R@K as the primary metric. The "R@5 raw
mode" headline number is turn-level recall_any@5.

Question types are grouped into 5 strata:
- single_session: single-session-user, single-session-assistant,
                   single-session-preference
- multi_session: multi-session
- temporal_reasoning: temporal-reasoning
- knowledge_update: knowledge-update
- abstention: abstention-* (excluded from retrieval R@K per official protocol)
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field


# LongMemEval question_type -> stratum mapping
QUESTION_TYPE_TO_STRATUM: dict[str, str] = {
    "single-session-user": "single_session",
    "single-session-assistant": "single_session",
    "single-session-preference": "single_session",
    "multi-session": "multi_session",
    "temporal-reasoning": "temporal_reasoning",
    "knowledge-update": "knowledge_update",
}

# Strata that participate in retrieval R@K (abstention excluded per official protocol)
RETRIEVAL_STRATA = frozenset({
    "single_session",
    "multi_session",
    "temporal_reasoning",
    "knowledge_update",
})


def _classify_stratum(question_type: str) -> str:
    """Map a LongMemEval question_type to a stratum name."""
    if "abstention" in question_type.lower() or "_abs" in question_type.lower():
        return "abstention"
    return QUESTION_TYPE_TO_STRATUM.get(question_type, "unknown")


@dataclass
class QuestionResult:
    """Retrieval result for a single question."""

    question_id: str
    question_type: str
    stratum: str
    gold_session_ids: list[str]
    retrieved_chunk_ids: list[str]  # top-K, in rank order

    def recall_any_at_k(self, k: int) -> float:
        """1.0 if ANY gold session appears in top-K retrieved sessions, else 0.0.

        Session is extracted from chunk_id by splitting on '#' and taking
        the first segment (our chunk_id format: "{session_id}#t{turn}#p{prop}").
        """
        if not self.gold_session_ids:
            return 0.0
        retrieved_sessions = set()
        for cid in self.retrieved_chunk_ids[:k]:
            sid = cid.split("#")[0]
            retrieved_sessions.add(sid)
        gold = set(self.gold_session_ids)
        return 1.0 if gold & retrieved_sessions else 0.0

    def recall_all_at_k(self, k: int) -> float:
        """1.0 if ALL gold sessions appear in top-K retrieved sessions, else 0.0."""
        if not self.gold_session_ids:
            return 0.0
        retrieved_sessions = set()
        for cid in self.retrieved_chunk_ids[:k]:
            sid = cid.split("#")[0]
            retrieved_sessions.add(sid)
        gold = set(self.gold_session_ids)
        return 1.0 if gold <= retrieved_sessions else 0.0

    def ndcg_at_k(self, k: int) -> float:
        """NDCG@K with binary relevance at the session level."""
        if not self.gold_session_ids:
            return 0.0
        gold = set(self.gold_session_ids)
        relevances = []
        for cid in self.retrieved_chunk_ids[:k]:
            sid = cid.split("#")[0]
            relevances.append(1.0 if sid in gold else 0.0)

        # Pad to k if fewer retrieved
        relevances.extend([0.0] * (k - len(relevances)))

        # DCG
        dcg = relevances[0]
        for i in range(1, len(relevances)):
            dcg += relevances[i] / math.log2(i + 1)

        # Ideal DCG: all 1s first
        num_relevant = min(len(gold), k)
        ideal = [1.0] * num_relevant + [0.0] * (k - num_relevant)
        idcg = ideal[0]
        for i in range(1, len(ideal)):
            idcg += ideal[i] / math.log2(i + 1)

        return dcg / idcg if idcg > 0 else 0.0


@dataclass
class EvalReport:
    """Aggregate metrics over all questions."""

    results: list[QuestionResult] = field(default_factory=list)

    def _retrieval_results(self) -> list[QuestionResult]:
        """Filter to non-abstention questions (per official LongMemEval protocol)."""
        return [r for r in self.results if r.stratum in RETRIEVAL_STRATA]

    def recall_any_at_k(self, k: int) -> float:
        """Macro-averaged recall_any@K over all non-abstention questions."""
        rr = self._retrieval_results()
        if not rr:
            return 0.0
        return sum(r.recall_any_at_k(k) for r in rr) / len(rr)

    def recall_all_at_k(self, k: int) -> float:
        """Macro-averaged recall_all@K over all non-abstention questions."""
        rr = self._retrieval_results()
        if not rr:
            return 0.0
        return sum(r.recall_all_at_k(k) for r in rr) / len(rr)

    def ndcg_at_k(self, k: int) -> float:
        """Macro-averaged NDCG@K over all non-abstention questions."""
        rr = self._retrieval_results()
        if not rr:
            return 0.0
        return sum(r.ndcg_at_k(k) for r in rr) / len(rr)

    def per_stratum_recall_any_at_k(self, k: int) -> dict[str, float]:
        """Recall_any@K broken down by stratum."""
        by_stratum: dict[str, list[float]] = defaultdict(list)
        for r in self._retrieval_results():
            by_stratum[r.stratum].append(r.recall_any_at_k(k))
        return {
            s: sum(v) / len(v) if v else 0.0
            for s, v in sorted(by_stratum.items())
        }

    def per_stratum_recall_all_at_k(self, k: int) -> dict[str, float]:
        """Recall_all@K broken down by stratum."""
        by_stratum: dict[str, list[float]] = defaultdict(list)
        for r in self._retrieval_results():
            by_stratum[r.stratum].append(r.recall_all_at_k(k))
        return {
            s: sum(v) / len(v) if v else 0.0
            for s, v in sorted(by_stratum.items())
        }

    def summary(self, ks: list[int] | None = None) -> dict:
        """Full summary dict suitable for JSON serialization."""
        if ks is None:
            ks = [5, 10]
        out: dict = {
            "total_questions": len(self.results),
            "retrieval_questions": len(self._retrieval_results()),
        }
        for k in ks:
            out[f"recall_any@{k}"] = round(self.recall_any_at_k(k), 4)
            out[f"recall_all@{k}"] = round(self.recall_all_at_k(k), 4)
            out[f"ndcg@{k}"] = round(self.ndcg_at_k(k), 4)
            out[f"per_stratum_recall_any@{k}"] = {
                s: round(v, 4)
                for s, v in self.per_stratum_recall_any_at_k(k).items()
            }
        return out
