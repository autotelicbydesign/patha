"""Streamlit app — Patha belief viewer.

Run:
    uv run patha viewer
    uv run patha-viewer
    streamlit run src/patha/viewer/app.py -- --data-dir ~/.patha

Five panels, each independently useful:

  - Overview    — totals, detector, store path, plasticity gauges
  - Timeline    — chronological ingest events (added/reinforced/superseded)
  - Current     — current beliefs table with confidence and source
  - History     — superseded beliefs with supersession chain
  - Non-commutative replay — pick two orderings, see divergent states
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from patha.belief import (
    BeliefLayer,
    BeliefStore,
    make_detector,
)


DEFAULT_DATA_DIR = Path(
    os.environ.get("PATHA_STORE_PATH", str(Path.home() / ".patha"))
)


# ─── Data loading ──────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading belief store…")
def _load_store(store_path: str):
    """Load the persisted BeliefStore. Cached so Streamlit reruns don't rebuild."""
    return BeliefStore(persistence_path=Path(store_path))


def _beliefs_df(store: BeliefStore) -> pd.DataFrame:
    rows = []
    for b in store.all():
        rows.append({
            "id": b.id[:8],
            "proposition": b.proposition,
            "status": b.status.value,
            "asserted_at": b.asserted_at,
            "session": b.asserted_in_session,
            "confidence": round(b.confidence, 3),
            "pramana": b.pramana.value,
            "reinforced_by": len(b.reinforced_by),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("asserted_at").reset_index(drop=True)
    return df


# ─── Panels ────────────────────────────────────────────────────────

def _panel_overview(store: BeliefStore) -> None:
    st.subheader("Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total beliefs", len(store))
    col2.metric("Current", len(store.current()))
    col3.metric("Superseded", len(store.superseded()))
    col4.metric("Archived", len(store.archived()))

    confidences = [b.confidence for b in store.all()]
    if confidences:
        st.markdown("**Confidence distribution**")
        fig = _confidence_histogram(confidences)
        st.plotly_chart(fig, use_container_width=True)


def _confidence_histogram(confidences: list[float]):
    import plotly.express as px
    df = pd.DataFrame({"confidence": confidences})
    fig = px.histogram(
        df, x="confidence", nbins=20,
        title=f"Confidence across {len(confidences)} beliefs "
              f"(mean={statistics.mean(confidences):.2f}, "
              f"std={statistics.stdev(confidences) if len(confidences) > 1 else 0:.2f})",
    )
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def _panel_timeline(df: pd.DataFrame) -> None:
    st.subheader("Timeline")
    if df.empty:
        st.info("No beliefs ingested yet. Run `patha demo` or ingest some propositions first.")
        return
    import plotly.express as px
    color_map = {"current": "#2ca02c", "superseded": "#d62728",
                 "disputed": "#ff7f0e", "archived": "#7f7f7f"}
    fig = px.scatter(
        df, x="asserted_at", y="status",
        color="status", color_discrete_map=color_map,
        hover_data=["proposition", "session", "confidence", "reinforced_by"],
        title="Ingest events over time",
    )
    fig.update_traces(marker=dict(size=14, line=dict(width=1, color="white")))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)


def _panel_current(df: pd.DataFrame) -> None:
    st.subheader("Current beliefs")
    current = df[df["status"] == "current"].copy()
    if current.empty:
        st.info("No current beliefs.")
        return
    st.dataframe(
        current[["asserted_at", "proposition", "confidence", "session", "pramana", "reinforced_by"]],
        use_container_width=True, hide_index=True,
    )


def _panel_history(df: pd.DataFrame, store: BeliefStore) -> None:
    st.subheader("History (superseded beliefs)")
    history = df[df["status"] == "superseded"].copy()
    if history.empty:
        st.info("No superseded beliefs yet. Ingest contradictory propositions "
                "to see supersession in action.")
        return

    # For each superseded belief, find what superseded it
    # (by looking up beliefs that list this id in their supersedes field)
    successor_lookup = {}
    for b in store.current():
        for superseded_id in getattr(b, "supersedes", []):
            successor_lookup.setdefault(superseded_id, []).append(b)
    full_by_prefix = {b.id[:8]: b for b in store.all()}

    rows = []
    for _, row in history.iterrows():
        belief = full_by_prefix.get(row["id"])
        if belief is None:
            continue
        succs = successor_lookup.get(belief.id, [])
        succ_text = " | ".join(s.proposition[:60] for s in succs) if succs else "(none)"
        rows.append({
            "asserted_at": row["asserted_at"],
            "proposition": row["proposition"],
            "superseded_by": succ_text,
            "confidence": row["confidence"],
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _panel_non_commutative(store: BeliefStore) -> None:
    st.subheader("Non-commutative replay")
    st.markdown(
        "Non-commutativity is Patha's genuinely novel property: "
        "ingesting the same propositions in different orders can produce "
        "different final belief sets. Enter a list of propositions and "
        "Patha will ingest them both forward and reversed, showing which "
        "beliefs end up current under each ordering."
    )

    example = (
        "I love sushi and eat it every week\n"
        "I am avoiding raw fish on my doctor's advice\n"
        "I am now fully vegetarian for ethical reasons"
    )
    raw = st.text_area(
        "Propositions (one per line)", value=example, height=150,
    )
    detector_name = st.selectbox(
        "Detector", ("stub", "full-stack-v7"),
        help="'stub' is instant; 'full-stack-v7' downloads ~1.7 GB on first use.",
    )

    if st.button("Compare orderings"):
        propositions = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        if len(propositions) < 2:
            st.warning("Need at least two propositions to compare orderings.")
            return
        with st.spinner("Running forward + reversed reingestion…"):
            try:
                from datetime import timedelta

                from patha.belief.counterfactual import (
                    CounterfactualInput,
                    reingest_order_sensitivity,
                )
                base = datetime(2024, 1, 1)
                inputs_fwd = [
                    CounterfactualInput(
                        proposition=p,
                        asserted_at=base + timedelta(days=i),
                        asserted_in_session=f"fwd-{i}",
                        source_proposition_id=f"fwd-{i}",
                    )
                    for i, p in enumerate(propositions)
                ]
                inputs_rev = [
                    CounterfactualInput(
                        proposition=p,
                        asserted_at=base + timedelta(days=i),
                        asserted_in_session=f"rev-{i}",
                        source_proposition_id=f"rev-{i}",
                    )
                    for i, p in enumerate(reversed(propositions))
                ]
                det = make_detector(detector_name)
                result = reingest_order_sensitivity(
                    inputs=(inputs_fwd + inputs_rev),
                    orderings=[
                        list(range(len(inputs_fwd))),
                        list(range(len(inputs_fwd), len(inputs_fwd) + len(inputs_rev))),
                    ],
                    detector=det,
                )
            except Exception as e:
                st.error(f"Replay failed: {e}")
                return

        non_c = result["non_commutative"]
        div = result["divergence"]
        col_a, col_b = st.columns(2)
        col_a.metric(
            "Non-commutative?",
            "YES" if non_c else "no",
            help="True if forward and reversed orderings produce different "
                 "current-belief sets.",
        )
        col_b.metric("Divergence (Jaccard)", f"{div:.2f}")

        fwd = result["per_ordering"][0]
        rev = result["per_ordering"][1]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Forward order — current beliefs**")
            for p in fwd["current_props"]:
                st.markdown(f"- {p}")
            if fwd.get("superseded_props"):
                st.caption("Superseded:")
                for p in fwd["superseded_props"]:
                    st.caption(f"  ~ {p}")
        with c2:
            st.markdown("**Reversed order — current beliefs**")
            for p in rev["current_props"]:
                st.markdown(f"- {p}")
            if rev.get("superseded_props"):
                st.caption("Superseded:")
                for p in rev["superseded_props"]:
                    st.caption(f"  ~ {p}")


# ─── Main ──────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    # Streamlit injects its own args; ignore extras
    args, _ = parser.parse_known_args()
    return args


def main() -> int:
    """Entry point for `patha-viewer` script."""
    import subprocess
    app_path = Path(__file__).absolute()
    cmd = ["streamlit", "run", str(app_path)]
    return subprocess.call(cmd + sys.argv[1:])


# ─── Streamlit page ────────────────────────────────────────────────

st.set_page_config(
    page_title="Patha belief viewer",
    page_icon="🧠",
    layout="wide",
)

st.title("Patha belief viewer")
st.caption(
    "Local-first AI memory with supersession, plasticity, and non-commutative "
    "belief evolution."
)

args = _parse_args()
store_path = args.data_dir / "beliefs.jsonl"

if not store_path.exists():
    st.warning(
        f"No belief store found at `{store_path}`.\n\n"
        "Run `patha demo` or ingest a few propositions to populate it, "
        "then reload this page."
    )
    st.stop()

store = _load_store(str(store_path))
df = _beliefs_df(store)

tab_overview, tab_timeline, tab_current, tab_history, tab_nonc = st.tabs(
    ["Overview", "Timeline", "Current", "History", "Non-commutative replay"],
)

with tab_overview:
    _panel_overview(store)
    st.caption(f"Store file: `{store_path}` ({store_path.stat().st_size} bytes)")

with tab_timeline:
    _panel_timeline(df)

with tab_current:
    _panel_current(df)

with tab_history:
    _panel_history(df, store)

with tab_nonc:
    _panel_non_commutative(store)


if __name__ == "__main__":
    # When executed directly by `streamlit run`, main() isn't invoked;
    # the module itself is the script.
    pass
