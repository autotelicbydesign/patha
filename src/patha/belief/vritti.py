"""Vṛtti classification — the cognitive mode of a belief at retrieval time.

Per Patañjali's Yoga Sūtras (I.5-11), there are five vṛttis — modifications
of mind. Each one is a *mode of relating to content*, not a label on the
content itself:

  PRAMANA      — valid cognition. The belief is current, well-grounded,
                 non-disputed. Normal retrieval of a true-seeming belief.

  VIPARYAYA    — erroneous cognition. The belief has been contradicted
                 but either we're surfacing it deliberately (for history)
                 or it's in an unresolved dispute. Think 'rope mistaken
                 for snake' — there's a cognition, but it's wrong.

  VIKALPA      — verbal-only / imagined cognition. An assertion with no
                 corresponding reality check. Maps to AMBIGUOUS / low-
                 confidence beliefs the user mentioned but never
                 confirmed through perception. 'I think the meeting
                 is on Thursday' (never verified).

  NIDRA        — latency / dormancy. Beliefs that have been archived
                 or pruned — still in the store, filtered out of
                 default surfaces, but can be reactivated on specific
                 triggers.

  SMRTI        — recall of prior impression. The act of retrieving a
                 belief you previously formed. Distinct from pramāṇa:
                 pramāṇa is 'I know this directly'; smṛti is 'I
                 remember I concluded this'. In Patha terms: a belief
                 surfaced by retrieval but whose original pramāṇa is
                 no longer active (e.g., a superseded belief surfaced
                 via history).

Vṛtti is derived from ResolutionStatus + retrieval context — it's not
a stored field. Every surfaced belief gets a vṛtti classification
computed at query time, which can be surfaced to callers for nuanced
rendering ('you currently believe X [pramāṇa], used to believe Y
[smṛti/viparyaya]').

This module is intentionally small. vṛtti classification exposes a
cognitive dimension; policy consequences (e.g., 'don't surface vikalpa
in direct-answer mode') are implemented by callers that consume the
classification.
"""

from __future__ import annotations

from enum import Enum

from patha.belief.types import Belief, ResolutionStatus


class VrittiClass(str, Enum):
    """Five cognitive modes from Patañjali's taxonomy.

    These are classifications, not stored fields. Derived from a
    belief's status + context at retrieval time.
    """

    PRAMANA = "pramana"
    VIPARYAYA = "viparyaya"
    VIKALPA = "vikalpa"
    NIDRA = "nidra"
    SMRTI = "smrti"


def vritti_of(
    belief: Belief,
    *,
    surfaced_as_history: bool = False,
) -> VrittiClass:
    """Classify a belief's cognitive mode at the moment of retrieval.

    Parameters
    ----------
    belief
        The belief being retrieved.
    surfaced_as_history
        True iff this belief is being surfaced as part of a
        supersession lineage walk (include_history=True on a query),
        not as a current assertion. Changes the vṛtti:
        - SUPERSEDED/BADHITA beliefs surfaced as history → SMRTI
          (you're recalling a past conclusion)
        - SUPERSEDED/BADHITA beliefs surfaced outside history → VIPARYAYA
          (you're entertaining a known-wrong cognition)

    Returns
    -------
    VrittiClass

    Classification logic:
        status=CURRENT, high confidence        → PRAMANA
        status=CURRENT, low confidence         → VIKALPA (asserted but unverified)
        status=COEXISTS                        → PRAMANA (valid co-held beliefs)
        status=SUPERSEDED, as history          → SMRTI (recollection)
        status=SUPERSEDED, not history         → VIPARYAYA (erroneous cognition)
        status=BADHITA, as history             → SMRTI
        status=BADHITA, not history            → VIPARYAYA
        status=DISPUTED                        → VIPARYAYA (one of two is wrong)
        status=AMBIGUOUS                       → VIKALPA (unverified assertion)
        status=ARCHIVED                        → NIDRA (latent, not currently active)
    """
    status = belief.status

    if status == ResolutionStatus.ARCHIVED:
        return VrittiClass.NIDRA

    if status == ResolutionStatus.AMBIGUOUS:
        return VrittiClass.VIKALPA

    if status == ResolutionStatus.DISPUTED:
        return VrittiClass.VIPARYAYA

    if status in (ResolutionStatus.SUPERSEDED, ResolutionStatus.BADHITA):
        return VrittiClass.SMRTI if surfaced_as_history else VrittiClass.VIPARYAYA

    # CURRENT or COEXISTS
    # Low-confidence current beliefs are still vikalpa — they're
    # verbal assertions without robust grounding.
    if belief.confidence < 0.5:
        return VrittiClass.VIKALPA
    return VrittiClass.PRAMANA


def vritti_label(vritti: VrittiClass) -> str:
    """Short human-readable label for a vṛtti.

    Used in direct-answer and summary rendering to surface the
    cognitive mode to callers.
    """
    return {
        VrittiClass.PRAMANA: "valid cognition",
        VrittiClass.VIPARYAYA: "erroneous / contradicted",
        VrittiClass.VIKALPA: "asserted but unverified",
        VrittiClass.NIDRA: "latent / archived",
        VrittiClass.SMRTI: "recollection",
    }[vritti]
