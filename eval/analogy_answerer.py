"""AnalogyEval adapter for the production upamāna primitive.

Registers `upamana` as an answerer: episodes are built straight from
the scenario's session-grouped propositions (the same grouping the
production path derives from belief.asserted_in_session), ranked by
rank_analogues with the shared MiniLM embedder.

Usage:
    uv run python -m eval.analogy_eval \\
        --data eval/analogy_data/dev_scenarios.jsonl \\
        --answerer eval.analogy_answerer:upamana_answerer
"""

from __future__ import annotations

from eval.analogy_eval import AnalogyAnswer, ScenarioContext

_EMBEDDER = None


def _embed(texts):
    global _EMBEDDER
    if _EMBEDDER is None:
        from patha.models.embedder_st import SentenceTransformerEmbedder
        _EMBEDDER = SentenceTransformerEmbedder()
    return _EMBEDDER.embed(list(texts))


class _UpamanaAnswerer:
    needs_memory = False

    def __call__(self, ctx: ScenarioContext, question: str) -> AnalogyAnswer:
        from patha.belief.upamana import (
            detect_analogy_question,
            rank_analogues,
        )

        if not detect_analogy_question(question):
            return AnalogyAnswer(routed=False)
        episodes: dict[str, list[str]] = {}
        for p in ctx.scenario["propositions"]:
            episodes.setdefault(p["session"], []).append(p["text"])
        result = rank_analogues(question, episodes, embed_fn=_embed)
        return AnalogyAnswer(
            routed=True,
            analogue_sessions=result.sessions,
            structure_text="; ".join(result.shared_structure),
        )


upamana_answerer = _UpamanaAnswerer()
