"""Order-sensitive / counterfactual belief operations.

Mental operators in quantum cognition (Busemeyer & Bruza 2012) are
*non-commutative*: applying operator A then B yields a different end
state than B then A. This is empirically robust in humans (Moore 2002
on survey order effects; Pothos & Busemeyer 2013 on conjunction
fallacy).

Patha's belief store is *already* order-sensitive — supersession is
temporally ordered, reinforcement compounds, plasticity ticks advance
on every ingest. This module exposes that as a first-class API:

  replay_in_order(...)         → rebuild the belief state from events
                                  but with a specified permutation
                                  of the original ingest order.
                                  [event-replay mode; decisions frozen]

  reingest_in_order(...)       → [v0.7] fully re-run the contradiction
                                  detector on a reordered sequence of
                                  (proposition, asserted_at) inputs.
                                  Different orderings can produce
                                  genuinely different final beliefs.

  order_sensitivity(...)       → divergence metric between orderings.

Event-replay mode (replay_in_order) is cheap but conservative: it
reuses frozen decisions. Reingest mode (reingest_in_order) is more
expensive (one detector call per proposition) but answers the stronger
counterfactual: if the user had asserted things in a different order,
WOULD Patha have reached a different conclusion?

This is the empirical test for non-commutativity: if
reingest_in_order(order_A) and reingest_in_order(order_B) produce
different current-belief sets, then belief evolution is genuinely
order-sensitive, not just timestamped.

Caveats:
  - replay_in_order needs a persistent event log (JSONL).
  - reingest_in_order needs the original (proposition, asserted_at)
    inputs, plus a ContradictionDetector — typically the same one
    used to produce the original trajectory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from patha.belief.contradiction import ContradictionDetector
from patha.belief.layer import BeliefLayer
from patha.belief.store import BeliefStore
from patha.belief.types import BeliefId


# ─── Internals ─────────────────────────────────────────────────────

def _load_events(path: Path) -> list[dict]:
    events: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events


def _event_belief_id(ev: dict) -> BeliefId | None:
    """Return the belief id associated with an ADD event, else None."""
    if ev.get("type") == "add":
        data = ev.get("belief", {})
        bid = data.get("id")
        if isinstance(bid, str):
            return bid
    return None


# ─── Public API ────────────────────────────────────────────────────

def replay_in_order(
    source_path: Path,
    ordering: list[BeliefId],
) -> BeliefStore:
    """Replay a store's events with a specified belief-add ordering.

    ``ordering`` is a list of belief ids in the desired add-order. Any
    add events for ids not in the list are appended at the end in
    their original order. Non-add events keep their positions
    relative to the add events they reference (so supersede/reinforce
    edges stay consistent).

    Returns a fresh in-memory BeliefStore with the reordered state.
    The original source file is not modified.

    Use case:
        original = [b1, b2, b3]  # asserted in this order
        alt = replay_in_order(path, [b3, b1, b2])
        # alt now has b3 first, b1 second, b2 third — supersede edges
        # that were already recorded still point between the right
        # pairs, but their apparent 'temporal' sequence is different.
    """
    events = _load_events(source_path)

    # Split into add-events (which we can reorder) and others (kept in place)
    add_events: dict[BeliefId, dict] = {}
    non_add_events: list[dict] = []
    for ev in events:
        bid = _event_belief_id(ev)
        if bid is not None:
            add_events[bid] = ev
        else:
            non_add_events.append(ev)

    # Build the new event sequence: reordered adds first, then non-adds
    # in original order. This preserves the relational structure
    # (supersede/reinforce edges) because those events reference ids,
    # and all ids will exist by the time those events replay.
    reordered_adds: list[dict] = []
    for bid in ordering:
        if bid in add_events:
            reordered_adds.append(add_events[bid])
    # Append any add events not explicitly ordered, in their original order
    ordered_set = set(ordering)
    for ev in events:
        bid = _event_belief_id(ev)
        if bid is not None and bid not in ordered_set:
            reordered_adds.append(ev)

    new_store = BeliefStore()
    for ev in reordered_adds + non_add_events:
        new_store._apply_event(ev)

    return new_store


@dataclass(frozen=True)
class CounterfactualInput:
    """One (proposition, time, session) triple for reingest_in_order."""
    proposition: str
    asserted_at: datetime
    asserted_in_session: str = "cf-session"
    source_proposition_id: str | None = None


def reingest_in_order(
    inputs: list[CounterfactualInput],
    detector: ContradictionDetector,
    *,
    contradiction_threshold: float = 0.7,
) -> BeliefLayer:
    """Reingest a sequence of propositions in the given order, running
    the contradiction detector live.

    Unlike ``replay_in_order`` which reuses frozen decisions, this
    function fully re-evaluates contradictions under the new ordering.
    If the detector returns a different verdict on a pair depending on
    which one was seen first, the final state diverges.

    Returns the resulting BeliefLayer (with a fresh BeliefStore).

    Use case:
        original_order = [input_A, input_B, input_C]
        alt_order = [input_C, input_A, input_B]
        layer_1 = reingest_in_order(original_order, detector)
        layer_2 = reingest_in_order(alt_order, detector)
        # Compare layer_1.store.current() vs layer_2.store.current()

    The detector's verdicts are expected to be order-invariant (NLI
    should return the same label on (A, B) and (B, A) up to symmetry)
    — so the order-dependence comes from *which existing belief* the
    detector sees a new one compared against, plus which beliefs are
    then superseded. That cascading is genuinely order-sensitive.
    """
    layer = BeliefLayer(
        store=BeliefStore(),
        detector=detector,
        contradiction_threshold=contradiction_threshold,
    )
    for cf in inputs:
        layer.ingest(
            proposition=cf.proposition,
            asserted_at=cf.asserted_at,
            asserted_in_session=cf.asserted_in_session,
            source_proposition_id=cf.source_proposition_id,
        )
    return layer


def reingest_order_sensitivity(
    inputs: list[CounterfactualInput],
    orderings: list[list[int]],
    detector: ContradictionDetector,
    *,
    contradiction_threshold: float = 0.7,
) -> dict:
    """Measure current-belief divergence across reingest orderings.

    Parameters
    ----------
    inputs
        The full set of propositions available.
    orderings
        Each ordering is a list of indices into ``inputs`` specifying
        the order in which to reingest.
    detector
        The contradiction detector to use for every run.
    contradiction_threshold
        Passed to BeliefLayer.

    Returns a dict:
      divergence: pairwise Jaccard distance between final current-belief
                  text sets (float in [0, 1])
      per_ordering: list of {current_props, superseded_props} per run
      non_commutative: bool — True iff any two orderings produced
                       different current-belief sets
    """
    if len(orderings) < 2:
        raise ValueError("need at least two orderings to compare")

    per: list[dict] = []
    for order in orderings:
        ordered_inputs = [inputs[i] for i in order]
        layer = reingest_in_order(
            ordered_inputs, detector,
            contradiction_threshold=contradiction_threshold,
        )
        current_texts = sorted(b.proposition for b in layer.store.current())
        super_texts = sorted(b.proposition for b in layer.store.superseded())
        per.append({
            "current_props": current_texts,
            "superseded_props": super_texts,
            "ordering": list(order),
        })

    def jaccard_diff(a: list, b: list) -> float:
        sa, sb = set(a), set(b)
        if not sa and not sb:
            return 0.0
        inter = len(sa & sb)
        union = len(sa | sb)
        return 0.0 if union == 0 else 1.0 - inter / union

    total_div = 0.0
    n_pairs = 0
    non_commutative = False
    for i in range(len(per)):
        for j in range(i + 1, len(per)):
            d_current = jaccard_diff(
                per[i]["current_props"], per[j]["current_props"]
            )
            d_super = jaccard_diff(
                per[i]["superseded_props"], per[j]["superseded_props"]
            )
            total_div += (d_current + d_super) / 2
            n_pairs += 1
            if per[i]["current_props"] != per[j]["current_props"]:
                non_commutative = True

    return {
        "divergence": total_div / max(n_pairs, 1),
        "per_ordering": per,
        "non_commutative": non_commutative,
    }


def order_sensitivity(
    source_path: Path,
    orderings: list[list[BeliefId]],
) -> dict:
    """Measure belief-state divergence across multiple ingest orderings.

    For each ordering, replays the store and captures:
      - set of current belief ids
      - set of superseded belief ids

    Returns a dict with:
      divergence: fraction of beliefs whose (current, superseded)
                  status differs between orderings [0, 1]
      per_ordering: for each ordering index, the sets of current/
                    superseded belief ids

    A divergence of 0 means the final state is identical regardless
    of order (classical / commutative behaviour). Non-zero divergence
    means the store's final state genuinely depends on sequence —
    evidence of Patha being 'non-commutative' in belief evolution.
    """
    if len(orderings) < 2:
        raise ValueError("need at least two orderings to compare")

    per: list[dict] = []
    for order in orderings:
        store = replay_in_order(source_path, order)
        per.append({
            "current_ids": sorted(b.id for b in store.current()),
            "superseded_ids": sorted(b.id for b in store.superseded()),
        })

    # Compute pairwise divergence as average over all pairs
    def jaccard_difference(a: list, b: list) -> float:
        sa, sb = set(a), set(b)
        if not sa and not sb:
            return 0.0
        inter = len(sa & sb)
        union = len(sa | sb)
        return 0.0 if union == 0 else 1.0 - inter / union

    total_divergence = 0.0
    n_pairs = 0
    for i in range(len(per)):
        for j in range(i + 1, len(per)):
            d_current = jaccard_difference(
                per[i]["current_ids"], per[j]["current_ids"]
            )
            d_super = jaccard_difference(
                per[i]["superseded_ids"], per[j]["superseded_ids"]
            )
            total_divergence += (d_current + d_super) / 2
            n_pairs += 1

    divergence = total_divergence / max(n_pairs, 1)
    return {
        "divergence": divergence,
        "per_ordering": per,
        "orderings": orderings,
    }
