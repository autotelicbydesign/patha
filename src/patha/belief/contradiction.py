"""Pairwise contradiction detection.

Given two propositions, decide whether they CONTRADICT, ENTAIL, or are
NEUTRAL to each other. This is the foundational mechanism of the belief
layer — supersession and validity reasoning both depend on it.

Design (per Phase 2 spec §2.1 and survey §C):

- A `ContradictionDetector` protocol lets us swap mechanisms (NLI model,
  structured predicate comparison, LLM-as-judge, hybrid).
- v0.1 ships one concrete implementation: NLIContradictionDetector,
  wrapping a HuggingFace NLI model fine-tuned for contradiction detection.
  Default model: MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
  (~91% MNLI accuracy, ~50ms/pair on GPU, ~200-500ms/pair on Apple Silicon).
- A StubContradictionDetector is provided for testing and for pipelines
  that want deterministic, model-free behaviour.

Rejected mechanisms for v0.1:
- Structured predicate extraction (OpenIE): extraction F1 on
  conversational text is 60-75%, making it the weakest link.
- LLM-as-judge only: 10-100x cost per pair, flexibility not yet needed.
- Hybrid NLI + LLM fallback: scaffolded for v0.2 once the LLM gateway
  module is ready.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from patha.belief.types import ContradictionLabel, ContradictionResult


# ─── Protocol ────────────────────────────────────────────────────────

@runtime_checkable
class ContradictionDetector(Protocol):
    """Interface for contradiction detection mechanisms.

    Implementations must be safe to call repeatedly and should cache any
    expensive one-time setup (model loading) internally.
    """

    def detect(self, p1: str, p2: str) -> ContradictionResult:
        """Classify the relation between two propositions."""
        ...

    def detect_batch(
        self, pairs: list[tuple[str, str]]
    ) -> list[ContradictionResult]:
        """Classify a batch of proposition pairs.

        Implementations should override for real batched inference;
        a protocol-level fallback that loops over detect() is provided
        by NLIContradictionDetector and StubContradictionDetector.
        """
        ...


# ─── Stub (deterministic, no model) ──────────────────────────────────

class StubContradictionDetector:
    """Keyword-based heuristic detector for tests and CI.

    NOT suitable for production use. Fires CONTRADICTS when both
    propositions contain matching subject cues and at least one carries
    a negation marker that the other lacks. Returns NEUTRAL otherwise.

    Useful for:
    - Unit tests that don't want to download a 1.7 GB NLI model
    - Deterministic reproduction of pipeline behaviour
    - Fast CI runs
    """

    NEGATION_CUES: frozenset[str] = frozenset({
        "not", "no", "never", "none", "neither", "nor",
        "avoid", "avoiding", "stopped", "quit", "former",
        "used to", "don't", "didn't", "doesn't", "isn't", "aren't",
    })

    def detect(self, p1: str, p2: str) -> ContradictionResult:
        l1 = p1.lower()
        l2 = p2.lower()
        neg1 = any(cue in l1 for cue in self.NEGATION_CUES)
        neg2 = any(cue in l2 for cue in self.NEGATION_CUES)

        # Cheap surface-level overlap: do the two share at least one
        # non-trivial content word? Avoids flagging random pairs.
        words1 = {w for w in l1.split() if len(w) > 3}
        words2 = {w for w in l2.split() if len(w) > 3}
        overlap = words1 & words2

        if overlap and (neg1 != neg2):
            return ContradictionResult(
                label=ContradictionLabel.CONTRADICTS,
                confidence=0.6,
                rationale="stub: asymmetric negation with content overlap",
            )
        return ContradictionResult(
            label=ContradictionLabel.NEUTRAL,
            confidence=0.5,
            rationale="stub: no contradiction heuristic fired",
        )

    def detect_batch(
        self, pairs: list[tuple[str, str]]
    ) -> list[ContradictionResult]:
        return [self.detect(p1, p2) for p1, p2 in pairs]


# ─── NLI-based (production) ──────────────────────────────────────────

# Default model: DeBERTa-v3-large fine-tuned on MNLI + FEVER + ANLI +
# Ling + WANLI. ~1.7 GB, strong on contradiction detection specifically.
DEFAULT_NLI_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"

# Smaller alternative for resource-constrained environments.
SMALL_NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

# HuggingFace NLI models conventionally emit 3 logits in this order.
# We verify against the model's id2label at load time.
_EXPECTED_LABELS = {"entailment", "neutral", "contradiction"}


class NLIContradictionDetector:
    """Contradiction detection via a fine-tuned NLI model.

    Parameters
    ----------
    model_name
        HuggingFace model id. Defaults to DeBERTa-v3-large.
    device
        Torch device string: "cpu", "cuda", "mps", or None for auto.
    batch_size
        Max pairs per forward pass in detect_batch().
    confidence_threshold
        Below this confidence, we downgrade a CONTRADICTS verdict to
        NEUTRAL. Prevents over-eager contradiction flagging. Default 0.5.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_NLI_MODEL,
        device: str | None = None,
        batch_size: int = 16,
        confidence_threshold: float = 0.5,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._confidence_threshold = confidence_threshold
        # Lazy-loaded on first detect() / detect_batch() call.
        self._tokenizer = None
        self._model = None
        self._label_map: dict[int, ContradictionLabel] | None = None

    # ── model loading (lazy) ────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        # Imports deferred so the module can be imported without
        # pulling in torch/transformers for type-only use.
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        # Auto-detect device if unspecified
        device = self._device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            self._model_name
        )
        model.eval()
        model.to(device)

        # Build label -> ContradictionLabel mapping from model config
        id2label_raw = {int(k): v.lower() for k, v in model.config.id2label.items()}
        seen_labels = set(id2label_raw.values())
        missing = _EXPECTED_LABELS - seen_labels
        if missing:
            raise ValueError(
                f"NLI model {self._model_name} is missing expected labels: "
                f"{missing}. Found: {seen_labels}"
            )

        label_map: dict[int, ContradictionLabel] = {}
        for idx, name in id2label_raw.items():
            if name == "contradiction":
                label_map[idx] = ContradictionLabel.CONTRADICTS
            elif name == "entailment":
                label_map[idx] = ContradictionLabel.ENTAILS
            elif name == "neutral":
                label_map[idx] = ContradictionLabel.NEUTRAL

        self._tokenizer = tokenizer
        self._model = model
        self._device = device
        self._label_map = label_map

    # ── public API ──────────────────────────────────────────────────

    def detect(self, p1: str, p2: str) -> ContradictionResult:
        return self.detect_batch([(p1, p2)])[0]

    def detect_batch(
        self, pairs: list[tuple[str, str]]
    ) -> list[ContradictionResult]:
        if not pairs:
            return []

        self._ensure_loaded()
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._label_map is not None

        import torch
        import torch.nn.functional as F

        results: list[ContradictionResult] = []
        for start in range(0, len(pairs), self._batch_size):
            batch = pairs[start : start + self._batch_size]
            premises = [p1 for p1, _ in batch]
            hypotheses = [p2 for _, p2 in batch]
            enc = self._tokenizer(
                premises,
                hypotheses,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            )
            enc = {k: v.to(self._device) for k, v in enc.items()}
            with torch.no_grad():
                logits = self._model(**enc).logits
            probs = F.softmax(logits, dim=-1).cpu().numpy()

            for row in probs:
                winner_idx = int(row.argmax())
                label = self._label_map[winner_idx]
                conf = float(row[winner_idx])

                # Below-threshold CONTRADICTS gets downgraded to NEUTRAL.
                # Protects against over-eager contradiction flagging —
                # Mem0 flat's destructive DELETE failure mode.
                if (
                    label == ContradictionLabel.CONTRADICTS
                    and conf < self._confidence_threshold
                ):
                    label = ContradictionLabel.NEUTRAL

                results.append(
                    ContradictionResult(label=label, confidence=conf)
                )

        return results
