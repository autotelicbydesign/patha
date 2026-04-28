"""Patha — local-first AI memory with contradiction detection,
non-destructive supersession, and paraphrase-robust retrieval.

The 3-line developer quickstart:

    >>> import patha
    >>> memory = patha.Memory()
    >>> memory.remember("I live in Lisbon")
    >>> memory.recall("where do I live?").summary
    'Current beliefs:\\n- [2026-04-20] I live in Lisbon'

Or pass the compact summary straight into an LLM prompt:

    >>> import anthropic
    >>> client = anthropic.Anthropic()
    >>> memory = patha.Memory(detector="full-stack-v8")
    >>> user_msg = "What's a good coffee shop near me?"
    >>> memory_context = memory.recall(user_msg).summary   # ~20 tokens
    >>> client.messages.create(
    ...     model="claude-sonnet-4",
    ...     system=f"You are a helpful assistant. User memory:\\n{memory_context}",
    ...     messages=[{"role": "user", "content": user_msg}],
    ...     max_tokens=256,
    ... )

The `Memory` class is the high-level API. For low-level control, import
`BeliefLayer`, `BeliefStore`, `IntegratedPatha`, and the detector
factories directly.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from patha.belief import (
    AVAILABLE_DETECTORS,
    BeliefLayer,
    BeliefStore,
    DirectAnswerer,
    make_detector,
)
from patha.integrated import IntegratedPatha, IntegratedResponse

__version__ = "0.9.3"

__all__ = [
    "Memory",
    "Recall",
    # Low-level exports for power users
    "BeliefLayer",
    "BeliefStore",
    "IntegratedPatha",
    "IntegratedResponse",
    "DirectAnswerer",
    "make_detector",
    "AVAILABLE_DETECTORS",
    "__version__",
]


class Recall:
    """Result of `Memory.recall()`. A thin view over IntegratedResponse.

    Attributes:
        answer:    short direct answer (possibly None if the system couldn't
                   produce one — fall back to `.summary` then).
        summary:   ~20-token structured string meant to drop into an LLM
                   system prompt. Always present.
        current:   list of {id, proposition, asserted_at, confidence} dicts
                   for the currently valid beliefs.
        history:   list of superseded beliefs (only populated when
                   `include_history=True` was passed).
        strategy:  "direct_answer" | "structured" | "raw" | "ganita" — how
                   the response was built.
        tokens:    token count of the summary (for budgeting).
        ganita:    optional GanitaResult — when the question is an
                   aggregation ("how much total", "how many", etc), the
                   procedural-arithmetic layer fills this with the
                   computed value and its contributing source belief ids.
                   None if no aggregation operator was detected.
    """

    __slots__ = (
        "answer", "summary", "current", "history", "strategy", "tokens",
        "ganita",
    )

    def __init__(
        self,
        response: IntegratedResponse,
        include_history: bool,
        ganita_result=None,
    ) -> None:
        self.answer: str | None = response.answer or None
        self.summary: str = response.prompt
        self.strategy: str = response.strategy
        self.tokens: int = response.tokens_in
        self.ganita = ganita_result  # GanitaResult | None
        rr = response.retrieval_result
        self.current = [
            {
                "id": b.id,
                "proposition": b.proposition,
                "asserted_at": b.asserted_at.isoformat() if b.asserted_at else None,
                "confidence": b.confidence,
            }
            for b in (rr.current if rr else [])
        ]
        self.history = [
            {
                "id": b.id,
                "proposition": b.proposition,
                "asserted_at": b.asserted_at.isoformat() if b.asserted_at else None,
                "confidence": b.confidence,
            }
            for b in (rr.history if rr and include_history else [])
        ]

    def __repr__(self) -> str:
        return (
            f"Recall(strategy={self.strategy!r}, tokens={self.tokens}, "
            f"current={len(self.current)}, history={len(self.history)})"
        )


class Memory:
    """A local-first memory store with contradiction detection and
    non-destructive supersession. This is the main developer-facing
    API for Patha.

    Parameters
    ----------
    path:
        Where to persist the belief store as JSONL. Defaults to
        ``~/.patha/beliefs.jsonl``. The file is append-only; concurrent
        readers are safe.
    detector:
        Contradiction detector. One of:

        - ``"stub"`` (default) — fast heuristic, no model downloads.
          Good for CI and light use.
        - ``"nli"`` — DeBERTa-v3-large MNLI (~1.7 GB on first use).
        - ``"adhyasa-nli"`` — NLI + lexical rewriting.
        - ``"full-stack"`` — NLI + adhyasa + numerical change detection.
        - ``"full-stack-v7"`` — full-stack + sequential-event detection.
        - ``"full-stack-v8"`` — full-stack-v7 + learned classifier.
          Recommended for production.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        detector: str = "stub",
        enable_phase1: bool = True,
        phase1_top_k: int = 20,
        enable_ganita: bool = True,
        hebbian_expansion: bool = True,
        hebbian_top_k_per_seed: int = 3,
        hebbian_max_added: int = 30,
        hebbian_session_seed_weight: float = 0.05,
    ) -> None:
        """
        enable_phase1
            If True (default), every `.recall()` routes through the
            full Phase 1 retrieval pipeline (7-view Vedic + BM25 + RRF)
            before Phase 2's belief-layer filter. Disable only for
            tiny stores where the lazy-build cost isn't worth it.
        phase1_top_k
            How many candidates Phase 1 returns to the belief layer.
            Default 20 is fine for focused conversational memory.
            For benchmark-style retrieval where many competing chunks
            exist (LongMemEval), 100–200 gives the belief layer enough
            room to find the answer. Trade-off: bigger summaries.
        enable_ganita
            If True (default), ingest extracts numerical (entity,
            attribute, value, unit) tuples into a sidecar gaṇita index;
            recall() answers aggregation questions ("how much total",
            "how many") by procedural arithmetic over the index.
            In-tradition with Vedic gaṇita / Sulbasūtra arithmetic and
            Aboriginal increase-walks. No LLM involved.
        hebbian_expansion
            If True (default), after Phase 1 retrieval, expand the
            candidate set via the Hebbian co-retrieval graph. Beliefs
            that have surfaced together in past queries co-surface in
            future ones, even if direct cosine similarity to the query
            is borderline. Closes the multi-session retrieval gap.
            Disable for retrieval ablation studies.
        hebbian_top_k_per_seed
            Per Phase-1 seed belief, how many strongest Hebbian
            neighbors to pull in. Default 3.
        hebbian_max_added
            Cap on total beliefs added by Hebbian expansion across all
            seeds. Default 30. Prevents the candidate set ballooning
            in dense graphs.
        hebbian_session_seed_weight
            Initial Hebbian weight for every pair of beliefs asserted
            in the same session. Lets cluster expansion produce signal
            from day-zero, before any queries have run. Default 0.05.
            Set to 0 to disable session seeding.
        """
        path = Path(path) if path is not None else (Path.home() / ".patha" / "beliefs.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._detector_name = detector
        self._phase1_enabled = enable_phase1
        self._phase1_top_k = phase1_top_k
        self._ganita_enabled = enable_ganita

        store = BeliefStore(persistence_path=path)
        layer = BeliefLayer(store=store, detector=make_detector(detector))

        # Gaṇita layer — sidecar JSONL index next to beliefs.jsonl.
        # Lazy-instantiated; no model load required (pure regex).
        self._ganita_index = None
        if enable_ganita:
            from patha.belief.ganita import GanitaIndex
            ganita_path = path.parent / (path.stem + ".ganita.jsonl")
            self._ganita_index = GanitaIndex(persistence_path=ganita_path)

        phase1_retrieve = None
        self._phase1_retriever = None
        if enable_phase1:
            try:
                from patha.phase1_bridge import LazyPhase1Retriever
                from patha.retrieval.pipeline import PipelineConfig
                # MMR caps the pipeline's intermediate output, so it
                # must be at least as big as top_k (and big enough to
                # survive session_cap=2 trimming on LongMemEval-scale
                # data with ~40 sessions per question).
                cfg = PipelineConfig(
                    top_k=phase1_top_k,
                    mmr_k=max(phase1_top_k, 30),
                    mmr_session_cap=max(5, phase1_top_k // 20),
                )
                self._phase1_retriever = LazyPhase1Retriever(
                    store, config=cfg,
                )
                phase1_retrieve = self._phase1_retriever
            except ImportError as e:
                # Phase 1 requires retrieval / embedder modules that
                # may not be installed in minimal environments. That's
                # recoverable — log and fall back to Phase 2 only.
                import sys
                print(
                    f"[patha.Memory] warning: Phase 1 retrieval disabled — "
                    f"missing dependency ({e.name}). Pass "
                    f"enable_phase1=False to silence this warning, or "
                    f"install the missing extra.",
                    file=sys.stderr,
                )
                self._phase1_enabled = False
            # Any other exception (programming error, disk issue,
            # configuration bug) is NOT swallowed — let it propagate
            # so the user sees a real traceback rather than silently
            # getting a degraded pipeline.

        self._patha = IntegratedPatha(
            belief_layer=layer,
            direct_answerer=DirectAnswerer(layer.store),
            phase1_retrieve=phase1_retrieve,
            hebbian_expansion=hebbian_expansion,
            hebbian_top_k_per_seed=hebbian_top_k_per_seed,
            hebbian_max_added=hebbian_max_added,
            hebbian_session_seed_weight=hebbian_session_seed_weight,
        )

    # ─── Primary API ────────────────────────────────────────────────

    def remember(
        self,
        proposition: str,
        *,
        asserted_at: datetime | None = None,
        session_id: str | None = None,
        source_id: str | None = None,
        context: str | None = None,
    ) -> dict[str, Any]:
        """Add a belief. Returns {action, belief_id, affected_belief_ids}.

        ``action`` is one of ``"added"`` (new), ``"reinforced"`` (matches
        an existing belief → bumps confidence), ``"superseded"`` (contradicts
        an existing belief → new becomes current, old moves to history)."""
        at = asserted_at or datetime.now()
        session = session_id or at.strftime("%Y-%m-%d")
        src = source_id or f"api-{session}-{int(at.timestamp() * 1000)}"
        ev = self._patha.ingest(
            proposition=proposition,
            asserted_at=at,
            asserted_in_session=session,
            source_proposition_id=src,
            context=context,
        )
        # Phase 1 index is now stale for the new belief; next recall()
        # rebuilds. Cheap flag-flip.
        if self._phase1_retriever is not None:
            self._phase1_retriever.invalidate()
        # Gaṇita: extract numerical tuples and add to the sidecar index.
        # Pure regex — no model, fast.
        if self._ganita_index is not None:
            from patha.belief.ganita import extract_tuples
            tuples = extract_tuples(
                proposition,
                belief_id=ev.new_belief.id,
                time=at.isoformat(),
            )
            self._ganita_index.add_many(tuples)
        return {
            "action": ev.action,
            "belief_id": ev.new_belief.id,
            "proposition": ev.new_belief.proposition,
            "affected_belief_ids": list(ev.affected_belief_ids),
        }

    def remember_session(
        self,
        turns: list[str],
        *,
        session_id: str,
        asserted_at: datetime | None = None,
    ) -> dict[str, Any]:
        """Ingest a whole conversation session as one belief.

        Use this when you have pre-bundled conversations (transcripts,
        chat logs, LongMemEval haystacks) rather than individual asserted
        facts. All user turns are concatenated into a single belief, which
        gives Phase 1 retrieval the session-level chunk granularity it was
        tuned for. Measurably +53pp over turn-level ingest on
        LongMemEval-KU.

        For the user-asserts-facts case ("I live in Lisbon"), use
        `.remember(single_fact)` — that's what Phase 2's supersession
        machinery is tuned for.
        """
        if not turns:
            return {"action": "skipped", "belief_id": None, "proposition": "",
                    "affected_belief_ids": []}
        concatenated = "\n\n".join(t for t in turns if t and t.strip())
        return self.remember(
            concatenated,
            asserted_at=asserted_at,
            session_id=session_id,
        )

    def recall(
        self,
        question: str,
        *,
        at_time: datetime | None = None,
        include_history: bool = False,
    ) -> Recall:
        """Return a compact memory summary for the question.

        Drop ``.summary`` directly into an LLM system prompt. Uses
        ~20 tokens vs the ~280–325 a naive conversation-history dump
        would take.
        """
        response = self._patha.query(
            question,
            at_time=at_time or datetime.now(),
            include_history=include_history,
            phase1_top_k=self._phase1_top_k,
        )
        # Gaṇita pass: try procedural arithmetic for aggregation
        # questions ("how much total", "how many"). Pure rule-based,
        # no LLM. Restrict to the retrieved beliefs so we only sum
        # topically-relevant numbers (e.g., bike-related expenses,
        # not every currency mention in the haystack).
        ganita_result = None
        if self._ganita_index is not None:
            from patha.belief.ganita import answer_aggregation_question
            retrieved_ids = None
            rr = response.retrieval_result
            if rr is not None:
                retrieved_ids = {b.id for b in rr.current}
                if include_history:
                    retrieved_ids |= {b.id for b in rr.history}
            try:
                ganita_result = answer_aggregation_question(
                    question, self._ganita_index,
                    restrict_to_belief_ids=retrieved_ids,
                )
            except Exception:
                ganita_result = None
        return Recall(
            response, include_history=include_history,
            ganita_result=ganita_result,
        )

    def history(self, term: str) -> list[dict[str, Any]]:
        """Every belief (current and superseded) mentioning ``term``.

        Case-insensitive substring match. Returns a list of dicts with
        ``id``, ``proposition``, ``status``, ``asserted_at``,
        ``confidence``.
        """
        term_lower = term.lower()
        matches = []
        for belief in self._patha.belief_layer.store.all():
            if term_lower in belief.proposition.lower():
                matches.append({
                    "id": belief.id,
                    "proposition": belief.proposition,
                    "status": belief.status.value,
                    "asserted_at": belief.asserted_at.isoformat() if belief.asserted_at else None,
                    "confidence": belief.confidence,
                })
        return matches

    def stats(self) -> dict[str, Any]:
        """Counts + storage info."""
        store = self._patha.belief_layer.store
        return {
            "path": str(self._path),
            "detector": self._detector_name,
            "total": len(store),
            "current": len(store.current()),
            "superseded": len(store.superseded()),
            "disputed": len(store.disputed()),
            "archived": len(store.archived()),
        }

    # ─── Escape hatches for power users ─────────────────────────────

    @property
    def belief_layer(self) -> BeliefLayer:
        """The underlying BeliefLayer — for plasticity, confidence tuning,
        custom contradiction thresholds, etc."""
        return self._patha.belief_layer

    @property
    def store(self) -> BeliefStore:
        """The underlying BeliefStore — raw event-log access."""
        return self._patha.belief_layer.store

    @property
    def path(self) -> Path:
        """The JSONL file path. Copy this file to another machine to
        move your memory."""
        return self._path

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"<Memory path={self._path} detector={self._detector_name!r} "
            f"total={s['total']} current={s['current']} superseded={s['superseded']}>"
        )
