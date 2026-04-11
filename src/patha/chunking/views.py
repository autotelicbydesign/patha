"""Vedic patha-inspired view constructors.

Each proposition is stored in 7 overlapping embedding views that mirror the
*pada / krama / jata / ghana* recitation schemes — the same redundancy
principle the Vedic tradition used to preserve the Rig Veda losslessly for
three millennia. Downstream retrieval embeds each view independently and
fuses hits via RRF, so matching any single view is sufficient for recall.

Views
-----
- **v1 pada** — proposition alone.
- **v2 krama** — proposition + next proposition.
- **v3 rkrama** — previous proposition + proposition.
- **v4 jata** — prev + prop + next (bidirectional triple).
- **v5 ghana** — entities(prop) + jata.
- **v6 reframed** — ``"fact about {dominant entity}: " + prop``.
- **v7 temporal** — ``"{nearest timestamp}: " + prop``.

When entity or timestamp enrichment is not available, the corresponding
view degrades gracefully to its enrichment-free form rather than being
dropped. This keeps every proposition indexable with a constant schema.
"""

from __future__ import annotations

from patha.chunking.propositionizer import Proposition

VIEW_NAMES: tuple[str, ...] = ("v1", "v2", "v3", "v4", "v5", "v6", "v7")


def _neighbor(props: list[Proposition], i: int, offset: int) -> str:
    """Return the text of a neighboring proposition, or an empty string if out of range."""
    j = i + offset
    if 0 <= j < len(props):
        return props[j].text
    return ""


def _join(*parts: str) -> str:
    """Join non-empty parts with a single space."""
    return " ".join(p for p in parts if p).strip()


def build_views(
    props: list[Proposition],
    *,
    entities: list[list[str]] | None = None,
    timestamps: list[str | None] | None = None,
) -> list[dict[str, str]]:
    """Construct the 7 Vedic-inspired views for each proposition.

    Parameters
    ----------
    props
        Ordered list of propositions from a single turn or conversation
        slice. Neighbor views (v2, v3, v4, v5) are computed within this
        list, so the caller decides the window — typically a whole turn.
    entities
        Optional list (parallel to ``props``) of entity strings per
        proposition. Used for v5 (ghana) and v6 (reframed). When ``None``,
        v5 degrades to v4 and v6 degrades to v1.
    timestamps
        Optional list (parallel to ``props``) of nearest-timestamp strings
        per proposition. Used for v7. When ``None`` or an entry is ``None``,
        v7 degrades to v1 for that proposition.

    Returns
    -------
    list[dict[str, str]]
        One dict per proposition, keyed by view name (``v1`` ... ``v7``).
    """
    n = len(props)
    if entities is not None and len(entities) != n:
        raise ValueError(f"entities length {len(entities)} != props length {n}")
    if timestamps is not None and len(timestamps) != n:
        raise ValueError(f"timestamps length {len(timestamps)} != props length {n}")

    out: list[dict[str, str]] = []
    for i, p in enumerate(props):
        prev = _neighbor(props, i, -1)
        nxt = _neighbor(props, i, +1)

        v1 = p.text
        v2 = _join(p.text, nxt) if nxt else p.text
        v3 = _join(prev, p.text) if prev else p.text
        v4 = _join(prev, p.text, nxt) if (prev or nxt) else p.text

        ent_list = entities[i] if entities is not None else []
        if ent_list:
            ent_str = ", ".join(ent_list)
            v5 = _join(f"[{ent_str}]", v4)
            v6 = f"fact about {ent_list[0]}: {p.text}"
        else:
            v5 = v4
            v6 = p.text

        ts = timestamps[i] if timestamps is not None else None
        v7 = f"{ts}: {p.text}" if ts else p.text

        out.append({"v1": v1, "v2": v2, "v3": v3, "v4": v4, "v5": v5, "v6": v6, "v7": v7})

    return out
