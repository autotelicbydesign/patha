"""Streamlit eval viewer for Patha.

Visualizes evaluation results: per-stratum breakdown, per-question
drill-down, and ablation comparison.

Usage:
    uv run streamlit run eval/viewer/app.py
"""

import json
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Patha Eval Viewer", layout="wide")
st.title("Patha Eval Viewer")

# ── Load results ─────────────────────────────────────────────────────

RUNS_DIR = Path("runs")


def find_results() -> dict[str, Path]:
    """Find all results.json files in runs/."""
    results = {}
    if RUNS_DIR.exists():
        for p in sorted(RUNS_DIR.rglob("results.json")):
            label = str(p.parent.relative_to(RUNS_DIR))
            results[label] = p
    return results


available = find_results()

if not available:
    st.warning("No results found in `runs/`. Run an eval first: `make eval-quick`")
    st.stop()

# ── Sidebar: select run ──────────────────────────────────────────────

selected = st.sidebar.selectbox("Select run", list(available.keys()))
result_path = available[selected]

with open(result_path) as f:
    data = json.load(f)

summary = data.get("summary", {})
per_question = data.get("per_question", [])

# ── Summary metrics ──────────────────────────────────────────────────

st.header(f"Run: {selected}")

col1, col2, col3, col4 = st.columns(4)
col1.metric("R@5", f"{summary.get('recall_any@5', 0):.3f}")
col2.metric("R@10", f"{summary.get('recall_any@10', 0):.3f}")
col3.metric("R_all@5", f"{summary.get('recall_all@5', 0):.3f}")
col4.metric("Questions", summary.get("total_questions", len(per_question)))

# ── Per-stratum breakdown ────────────────────────────────────────────

st.header("Per-Stratum R@5")

per_stratum = summary.get("per_stratum_recall_any@5", {})
if per_stratum:
    cols = st.columns(len(per_stratum))
    for col, (stratum, value) in zip(cols, sorted(per_stratum.items())):
        col.metric(stratum.replace("_", " ").title(), f"{value:.3f}")

# ── Per-question table ───────────────────────────────────────────────

st.header("Per-Question Results")

# Filter
strata = sorted(set(q.get("stratum", "unknown") for q in per_question))
filter_stratum = st.selectbox("Filter by stratum", ["all"] + strata)
filter_result = st.selectbox("Filter by result", ["all", "hits", "misses"])

filtered = per_question
if filter_stratum != "all":
    filtered = [q for q in filtered if q.get("stratum") == filter_stratum]
if filter_result == "hits":
    filtered = [q for q in filtered if q.get("recall_any@5", 0) == 1.0]
elif filter_result == "misses":
    filtered = [q for q in filtered if q.get("recall_any@5", 0) < 1.0]

st.write(f"Showing {len(filtered)} / {len(per_question)} questions")

for q in filtered:
    r5 = q.get("recall_any@5", 0)
    icon = "+" if r5 == 1.0 else "-"
    with st.expander(f"[{icon}] {q['question_id']} ({q.get('stratum', '?')}) — R@5={r5:.1f}"):
        st.write(f"**Type:** {q.get('question_type', '?')}")
        st.write(f"**Gold sessions:** {q.get('gold_session_ids', [])}")
        st.write(f"**Retrieved chunks:** {q.get('retrieved_chunk_ids', [])}")

        # Check which gold sessions were found
        retrieved_sessions = set()
        for cid in q.get("retrieved_chunk_ids", []):
            sid = cid.split("#")[0]
            retrieved_sessions.add(sid)

        gold = set(q.get("gold_session_ids", []))
        found = gold & retrieved_sessions
        missed = gold - retrieved_sessions

        if found:
            st.success(f"Found: {found}")
        if missed:
            st.error(f"Missed: {missed}")

# ── Ablation comparison ──────────────────────────────────────────────

comparison_path = RUNS_DIR / "ablation_comparison.json"
if comparison_path.exists():
    st.header("Ablation Comparison")

    with open(comparison_path) as f:
        comparisons = json.load(f)

    # Build table
    rows = []
    baseline_r5 = None
    for c in comparisons:
        r5 = c["summary"].get("recall_any@5", 0)
        if baseline_r5 is None:
            baseline_r5 = r5
        delta = r5 - baseline_r5
        rows.append({
            "Ablation": c["name"],
            "R@5": f"{r5:.3f}",
            "Delta": f"{delta:+.3f}" if c["name"] != "baseline" else "—",
            "Time": f"{c['elapsed_seconds']:.0f}s",
            "Description": c["description"],
        })

    st.table(rows)
