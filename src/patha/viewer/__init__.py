"""Patha Streamlit belief viewer.

Run with:
    uv run patha viewer               # via CLI wrapper
    uv run patha-viewer               # direct pypi script
    streamlit run -m patha.viewer.app # low-level

Reads the belief store at ~/.patha/beliefs.jsonl (or PATHA_STORE_PATH
env var) and displays:

  1. Timeline of ingest events (added / reinforced / superseded)
  2. Current beliefs table
  3. Supersession graph (network of which beliefs replace which)
  4. Non-commutativity replay (different orderings → different beliefs)
  5. Plasticity gauges (confidence, Hebbian edges, reinforcement)

Read-only — the viewer never mutates the store.
"""
