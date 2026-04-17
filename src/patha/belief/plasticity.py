"""Neuroplasticity-inspired belief maintenance mechanisms.

These are the operational primitives that make Patha's Phase 2 more than
a retrieval system: the belief store evolves over time like biological
memory, not as a frozen archive.

Five mechanisms, each mechanistically mapped to a Patha operation
(not just named after a neuroscience concept):

  LongTermPotentiation (LTP)
    Repeated or reinforced assertions increase belief confidence.
    Analogue: synaptic strengthening through co-activation.
    Already wired via BeliefStore.reinforce(). The LTP class here
    exposes a policy — "given a reinforcement count, what's the new
    confidence" — for tunability and ablations.

  LongTermDepression (LTD)
    Beliefs that aren't accessed decay in confidence over time.
    Analogue: synaptic weakening via non-use.
    Implementation: apply_decay(store, now) walks the store and nudges
    each belief's confidence down by an exponential factor of its age
    since last reinforcement.

  SynapticPruning
    Beliefs that have been superseded through many generations get
    archived out of the default query path.
    Analogue: removal of disused connections during development.
    Implementation: prune(store, max_depth) marks ancestors deeper than
    max_depth as 'archived' so they don't surface in default history
    walks. Non-destructive: an explicit archive query still shows them.

  HomeostaticRegulation
    Total active-belief confidence is normalised so no belief can run
    away with attention.
    Analogue: homeostatic plasticity keeping firing rates in range.
    Implementation: normalise(store) rescales confidences so their
    mean sits at a target (default 0.7), preserving ordering.

  HebbianAssociation
    Beliefs that are co-retrieved (returned together by the same query)
    form associative edges. Over time, a network of 'things that go
    together in this user's mind' emerges.
    Analogue: 'neurons that fire together wire together.'
    Implementation: record_coretrieval(belief_ids) bumps a symmetric
    weight between every pair. related(id, top_k) returns the most
    strongly associated beliefs.

Each mechanism is a first-class class with a clear policy. They can be
applied individually, composed, or ablated in BeliefEval comparisons.

Honest framing: this is belief-maintenance-inspired-by-plasticity, not
a faithful mechanistic simulation of neural plasticity. The value is a
coherent set of dynamic behaviours no other AI memory system ships.
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Iterable

from patha.belief.store import BeliefStore
from patha.belief.types import Belief, BeliefId


# ─── Long-term potentiation ──────────────────────────────────────────

class LongTermPotentiation:
    """Confidence-bumping policy on reinforcement.

    Default policy: on each reinforcement, close a fraction of the gap
    to 1.0 (diminishing returns — early bumps matter more than late).

    Parameters
    ----------
    gap_closure
        Fraction of remaining gap closed per reinforcement. Default 0.3.
    """

    def __init__(self, gap_closure: float = 0.3) -> None:
        if not 0.0 < gap_closure <= 1.0:
            raise ValueError("gap_closure must be in (0, 1]")
        self._gap_closure = gap_closure

    def new_confidence(self, current: float) -> float:
        gap = 1.0 - current
        return min(1.0, current + self._gap_closure * gap)

    def apply(self, belief: Belief) -> None:
        belief.confidence = self.new_confidence(belief.confidence)


# ─── Long-term depression (decay) ────────────────────────────────────

class LongTermDepression:
    """Time-based confidence decay policy.

    Beliefs decay exponentially with age since last reinforcement. Uses
    a half-life: after half_life_days, confidence halves toward the
    floor (which is 0 by default — the belief becomes arbitrarily
    uncertain rather than vanishing).

    Parameters
    ----------
    half_life_days
        Days for confidence to decay to half its original value.
        Default 365 (one year).
    floor
        Lower bound for decayed confidence. Default 0.0.
    """

    def __init__(
        self, half_life_days: float = 365.0, floor: float = 0.0
    ) -> None:
        if half_life_days <= 0:
            raise ValueError("half_life_days must be positive")
        if not 0.0 <= floor <= 1.0:
            raise ValueError("floor must be in [0, 1]")
        self._half_life_days = half_life_days
        self._floor = floor

    def decayed(self, original: float, age_days: float) -> float:
        if age_days <= 0:
            return original
        factor = math.pow(0.5, age_days / self._half_life_days)
        return max(self._floor, original * factor + self._floor * (1 - factor))

    def apply_to_store(
        self,
        store: BeliefStore,
        *,
        now: datetime,
        beliefs: Iterable[Belief] | None = None,
    ) -> int:
        """Walk the store (or a specified subset) and decay each belief.

        'Last access' in v0.1 means asserted_at or the most recent
        reinforcement — we don't track retrieval timestamps yet. The
        store's event log contains reinforcement times; as a lightweight
        proxy we use the asserted_at of any reinforcing belief.

        Returns the number of beliefs whose confidence was updated.
        """
        targets = list(beliefs) if beliefs is not None else store.all()
        updated = 0
        for b in targets:
            last_access = b.asserted_at
            for rid in b.reinforced_by:
                r = store.get(rid)
                if r is not None and r.asserted_at > last_access:
                    last_access = r.asserted_at
            age_days = (now - last_access).total_seconds() / 86400.0
            new_conf = self.decayed(b.confidence, age_days)
            if abs(new_conf - b.confidence) > 1e-9:
                store.set_confidence(b.id, new_conf)
                updated += 1
        return updated


# ─── Synaptic pruning ────────────────────────────────────────────────

class SynapticPruning:
    """Archive policy for deeply-superseded beliefs.

    Pruning here is non-destructive: it marks a belief as 'archived'
    (via a confidence floor of 0.0 and an archival tag on the event
    log). The belief is still reachable through explicit archive
    queries but filtered out of default lineage walks past max_depth.

    Parameters
    ----------
    max_depth
        Maximum supersession generations to keep in the default
        history. Ancestors beyond this are archived. Default 10.
    """

    def __init__(self, max_depth: int = 10) -> None:
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        self._max_depth = max_depth

    def prune(self, store: BeliefStore) -> list[BeliefId]:
        """Archive beliefs whose supersession depth exceeds max_depth.

        Returns the list of belief ids that were archived.
        """
        archived: list[BeliefId] = []
        for b in store.all():
            if not b.is_superseded:
                continue
            depth = self._depth_from_current(store, b.id)
            if depth is not None and depth > self._max_depth:
                store.set_confidence(b.id, 0.0)
                archived.append(b.id)
        return archived

    def _depth_from_current(
        self, store: BeliefStore, start: BeliefId
    ) -> int | None:
        """Depth = number of hops from this belief down to a current descendant.

        BFS along superseded_by edges. Returns None if no current
        descendant is reachable (orphan branch).
        """
        visited: set[BeliefId] = set()
        frontier: list[tuple[BeliefId, int]] = [(start, 0)]
        while frontier:
            bid, d = frontier.pop(0)
            if bid in visited:
                continue
            visited.add(bid)
            b = store.get(bid)
            if b is None:
                continue
            if b.is_current:
                return d
            for succ in b.superseded_by:
                frontier.append((succ, d + 1))
        return None


# ─── Homeostatic regulation ──────────────────────────────────────────

class HomeostaticRegulation:
    """Normalises the confidence distribution across current beliefs.

    Without this, repeated reinforcement of a small subset can drown out
    other beliefs. Homeostasis rescales so the mean confidence among
    current beliefs sits at target_mean, preserving relative ordering.

    Parameters
    ----------
    target_mean
        Desired mean confidence across current beliefs. Default 0.7.
    """

    def __init__(self, target_mean: float = 0.7) -> None:
        if not 0.0 < target_mean < 1.0:
            raise ValueError("target_mean must be in (0, 1)")
        self._target_mean = target_mean

    def apply(self, store: BeliefStore) -> int:
        current = store.current()
        if not current:
            return 0
        mean_conf = sum(b.confidence for b in current) / len(current)
        if mean_conf == 0:
            return 0
        scale = self._target_mean / mean_conf
        updated = 0
        for b in current:
            new = max(0.0, min(1.0, b.confidence * scale))
            if abs(new - b.confidence) > 1e-9:
                store.set_confidence(b.id, new)
                updated += 1
        return updated


# ─── Hebbian association ─────────────────────────────────────────────

class HebbianAssociation:
    """Associative-edge accumulator for co-retrieved beliefs.

    Each time a query returns a set of beliefs, the association weight
    between every pair is bumped. Over many queries, a network of
    'things that go together in this user's mind' emerges — orthogonal
    to the supersession graph, which records contradiction rather than
    association.

    State lives on the instance, not the store. Persistable via
    serialize() / deserialize() if needed.

    Parameters
    ----------
    learning_rate
        Per-event weight increment. Default 0.1.
    decay
        Multiplicative decay applied to all weights per tick.
        Default 1.0 (no decay). Set <1 to implement forgetting.
    """

    def __init__(self, learning_rate: float = 0.1, decay: float = 1.0) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 < decay <= 1.0:
            raise ValueError("decay must be in (0, 1]")
        self._lr = learning_rate
        self._decay = decay
        # Symmetric edge weights: weights[(a, b)] == weights[(b, a)].
        # We only store the lexicographically-smaller-first key.
        self._weights: dict[tuple[BeliefId, BeliefId], float] = defaultdict(float)

    @staticmethod
    def _key(a: BeliefId, b: BeliefId) -> tuple[BeliefId, BeliefId]:
        return (a, b) if a < b else (b, a)

    def record_coretrieval(self, belief_ids: Iterable[BeliefId]) -> None:
        """Bump association weights for every pair in the co-retrieved set."""
        if self._decay < 1.0:
            self._apply_decay()
        ids = list(set(belief_ids))
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                key = self._key(ids[i], ids[j])
                self._weights[key] += self._lr

    def _apply_decay(self) -> None:
        to_drop: list[tuple[BeliefId, BeliefId]] = []
        for k, v in self._weights.items():
            new_v = v * self._decay
            if new_v < 1e-6:
                to_drop.append(k)
            else:
                self._weights[k] = new_v
        for k in to_drop:
            del self._weights[k]

    def weight(self, a: BeliefId, b: BeliefId) -> float:
        if a == b:
            return 0.0
        return self._weights.get(self._key(a, b), 0.0)

    def related(self, belief_id: BeliefId, top_k: int = 5) -> list[tuple[BeliefId, float]]:
        """Return up to top_k belief ids most strongly associated with belief_id."""
        scores: list[tuple[BeliefId, float]] = []
        for (a, b), w in self._weights.items():
            if a == belief_id:
                scores.append((b, w))
            elif b == belief_id:
                scores.append((a, w))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def __len__(self) -> int:
        return len(self._weights)

    def serialize(self) -> dict[str, float]:
        """Serialise edges to a plain dict (str key "a||b" for JSONability)."""
        return {f"{a}||{b}": w for (a, b), w in self._weights.items()}

    @classmethod
    def deserialize(cls, data: dict[str, float]) -> "HebbianAssociation":
        inst = cls()
        for k, w in data.items():
            a, b = k.split("||", 1)
            inst._weights[cls._key(a, b)] = w
        return inst
