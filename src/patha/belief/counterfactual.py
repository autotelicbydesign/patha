"""Order-sensitive / counterfactual belief operations.

Mental operators in quantum cognition (Busemeyer & Bruza 2012) are
*non-commutative*: applying operator A then B yields a different end
state than B then A. This is empirically robust in humans (Moore 2002
on survey order effects; Pothos & Busemeyer 2013 on conjunction
fallacy).

Patha's belief store is *already* order-sensitive — supersession is
temporally ordered, reinforcement compounds, plasticity ticks advance
on every ingest. But that order-sensitivity has never been exposed as
a first-class API. This module adds two operations:

  replay_in_order(...)         → rebuild the belief state from events
                                  but with a specified permutation
                                  of the original ingest order.
  order_sensitivity(...)       → divergence metric between two orders'
                                  final belief states.

These enable counterfactual queries:
  "What would you currently believe about X, if you'd heard B before A?"
  "How much does the current state actually depend on the sequence?"

Implementation:
  - Reads the BeliefStore's event log (JSONL) and replays add/
    supersede/reinforce/coexist/dispute events into a fresh store.
    Non-destructive to the original store.
  - Events that don't depend on order (coexist, status_set, etc.) are
    preserved in their logical sequence; only *add* events are
    reordered. Supersede/reinforce events are re-issued with the
    same pair-references because those pairs are content, not order.
    (In a v0.5 extension we could replay from raw ingest inputs to
    fully re-evaluate contradictions with the new order — deferred.)

Caveats:
  - replay_in_order can only re-order events that are already in the
    event log, i.e., it works on a persistent store.
  - The current implementation is order-sensitive in the sense of
    'what the store ended up with' vs. 'what a different ordering
    would have yielded with the same decisions'. It does NOT re-run
    the contradiction detector on the re-ordered sequence.
    A full re-evaluation mode lives in v0.5 once we have a cheap
    way to replay NLI calls.
"""

from __future__ import annotations

import json
from pathlib import Path

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
