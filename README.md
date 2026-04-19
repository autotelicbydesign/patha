# Patha

> *The way. The recitation.*

**Local-first AI memory with contradiction detection, non-destructive supersession, and empirically non-commutative belief evolution.** Runs fully on your machine — zero hosted-LLM API calls anywhere in the pipeline.

Plugs into Claude Desktop / Claude Code / Cursor / Zed / Goose as an MCP server in one config line. Ships with a CLI, a Python library, and a Streamlit viewer for visual inspection.

Built on primitives drawn from two human memory traditions that lasted thousands of years: **Vedic recitation** (redundant multi-view encoding for lossless paraphrase-robust retrieval) and **Aboriginal songlines** (narrative graph traversal for multi-session recall).

---

## Quickstart — 2 minutes

```bash
# 1. Install (Python 3.11+ and uv required)
git clone https://github.com/autotelicbydesign/patha.git
cd patha
uv sync

# 2. Check your environment
uv run patha verify

# 3. Run the end-to-end demo (no downloads, ~10 seconds)
uv run patha demo

# 4. Pick how you want to use it:
uv run patha-mcp     # run as MCP server on stdio (for Claude Desktop)
uv run patha ingest "I am vegetarian"
uv run patha ask "what do I eat?"
uv run patha viewer  # visual inspection in browser
```

That's it. State persists to `~/.patha/beliefs.jsonl` across all three modes.

---

## Three ways to use it

### 1. As an MCP server (Claude Desktop, Claude Code, Cursor)

Add one block to your MCP client's config:

```json
{
  "mcpServers": {
    "patha": {
      "command": "uv",
      "args": ["run", "--project", "/ABSOLUTE/PATH/TO/patha", "patha-mcp"],
      "env": { "PATHA_STORE_PATH": "/Users/YOU/.patha" }
    }
  }
}
```

Restart your client. Four tools become available: `patha_ingest`, `patha_query`, `patha_history`, `patha_stats`. Your AI assistant can now remember things across sessions, detect contradictions, and reason over a personal belief store. See [docs/mcp.md](docs/mcp.md) for the full install guide + Claude Desktop walkthrough.

### 2. As a CLI

```bash
patha ingest "I love sushi"
patha ingest "I am avoiding raw fish on my doctor's advice"
patha ask "what do I currently eat?"          # routes through supersession
patha history "sushi"                         # every mention, current + superseded
patha stats                                   # store counts + plasticity state
```

Use `--detector full-stack-v7` to switch to the production NLI + adhyāsa + numerical + sequential detector (downloads ~1.7 GB on first run). Default is `stub` for instant startup.

### 3. As a Python library

```python
from datetime import datetime
from patha.belief import BeliefLayer, BeliefStore, make_detector
from patha.integrated import IntegratedPatha

layer = BeliefLayer(
    store=BeliefStore(),                           # in-memory, or pass persistence_path
    detector=make_detector("full-stack-v7"),       # or "stub" for fast CI
)
patha = IntegratedPatha(belief_layer=layer)

patha.ingest(
    proposition="I live in Berlin",
    asserted_at=datetime.now(),
    asserted_in_session="s1",
    source_proposition_id="s1-p1",
)
response = patha.query("where do I live?", at_time=datetime.now())
print(response.answer or response.prompt)
```

See [examples/belief_layer_demo.py](examples/belief_layer_demo.py) for a full walkthrough.

### Streamlit viewer

```bash
uv pip install 'patha[viewer]'
uv run patha viewer
```

Opens a browser dashboard over `~/.patha/beliefs.jsonl`:

- **Overview** — totals, confidence histogram, detector status
- **Timeline** — chronological ingest events (added / reinforced / superseded)
- **Current** — live belief table
- **History** — superseded beliefs with their successors
- **Non-commutative replay** — enter propositions, see how forward vs reversed ingest orders produce different final beliefs

---

## Why Patha

Every existing AI memory system I've read treats memory as retrieval — dump candidate chunks into the LLM's context and hope it sorts the contradictions out. That doesn't work when a user's beliefs evolve ("I was vegetarian, then I wasn't, then I started eating fish only"). Patha's belief layer does the sorting before the LLM sees it.

Three architectural choices that distinguish it:

1. **Non-destructive supersession.** When new evidence contradicts an old belief, the old belief moves to history instead of being overwritten. Queries can ask for current-only or current+history; the store retains everything you've ever said.
2. **Order-dependent evolution.** Measured empirically: on 240 supersession scenarios, reversing the ingest order produces a different final belief set 95.8% of the time (mean divergence 0.91). Reinforcement scenarios correctly come out 0% non-commutative. The metric and benchmark are both in this repo (`eval/non_commutative_eval.py`).
3. **Plasticity inspired by neural maintenance.** LTD time decay, Hebbian co-retrieval edges, homeostatic confidence regulation, synaptic pruning. On 10 real LongMemEval conversations the confidence distribution has std=0.106 (LTD is doing real work) and a mean of 150 Hebbian edges per conversation (associative graph emerges from use).

---

## Benchmarks (highlights)

Full numbers with caveats, ablations, and methodology live in [docs/benchmarks.md](docs/benchmarks.md). The headlines:

| Benchmark | Result | Comparison |
|---|---|---|
| **LongMemEval-KU R@5 (Phase 1 retrieval)** | 1.000 (78/78) | Mem0 ECAI 2025: 0.934 |
| LongMemEval S 100q stratified R@5 | 0.989 | — |
| BeliefEval 300-scenario / 347q (Phase 2) | 1.000 with full-stack-v7 | our own benchmark; see caveat |
| **LongMemEval-KU answer-in-summary (Phase 2 alone)** | **0.885 (69/78)** | stub null baseline: 0.795 |
| Non-commutativity on 240 supersession scenarios | 95.8% | 0% on reinforcement |
| Test suite | 602 pass | — |

**Caveat on the 1.000 BeliefEval:** the Phase 2 detector was iteratively tuned on the exact scenarios in our benchmark, so 1.000 on that set should be read as "no known misses" rather than "generalises everywhere." The honest external number is 0.885 on LongMemEval-KU, measured on a benchmark we didn't write.

---

## Architecture

```
INGEST:  conversation turn
           │
           ▼
     Phase 1 — Retrieval (Vedic 7-view + songline graph + BM25 + RRF)
           │
           ▼
     Phase 2 — Belief layer
           ├─ contradiction detection (NLI + adhyāsa + numerical + sequential)
           ├─ non-destructive supersession
           ├─ plasticity (LTD / LTP / Hebbian / homeostasis / pruning)
           └─ validity + pramāṇa + context
           │
           ▼
     BeliefStore  ──────►  ~/.patha/beliefs.jsonl   (append-only event log)
           │
           ▼
QUERY:    current-only  or  current + history
           │
           ▼
     strategy: direct-answer (no LLM) | structured summary | raw
```

Phase 1 is a self-contained retrieval pillar; Phase 2 is a self-contained belief layer. They can be used independently or wired together via `patha.integrated.IntegratedPatha`.

Background reading: [docs/phase_2_spec.md](docs/phase_2_spec.md) (architecture spec), [docs/phase_2_v07_results.md](docs/phase_2_v07_results.md) (latest sprint results with honest caveats), [docs/benchmarks.md](docs/benchmarks.md) (full benchmark tables).

---

## Project structure

```
src/patha/
  chunking/, indexing/, retrieval/, query/, models/   # Phase 1 — retrieval
  belief/                                              # Phase 2 — belief layer
    layer.py, store.py, types.py
    contradiction.py, adhyasa_detector.py, numerical_detector.py,
    sequential_detector.py, llm_judge.py, ollama_judge.py
    plasticity.py, counterfactual.py, validity_extraction.py
    pramana.py, vritti.py, abhava.py, direct_answer.py
    detector_factory.py                               # named-detector registry
  integrated.py                                        # Phase 1 + Phase 2
  cli.py                                               # patha verify/demo/ingest/ask/...
  mcp_server.py                                        # MCP stdio server
  viewer/                                              # Streamlit dashboard
    app.py
  demo.py                                              # patha demo

eval/
  runner.py, ablations.py, metrics.py                 # Phase 1 eval
  belief_eval.py, longmemeval_belief.py               # Phase 2 eval
  non_commutative_eval.py, plasticity_on_real_logs.py # Phase 2 novel metrics
  false_contradiction_eval.py                          # Phase 2 FP rate
  token_economy.py                                     # compression curves

examples/
  belief_layer_demo.py                                 # walkthrough story
  mcp_config_example.json                              # Claude Desktop template

docs/
  mcp.md                                               # MCP install guide
  benchmarks.md                                        # full numbers
  phase_2_spec.md, phase_2_v0{1..7}_results.md        # design + results
```

---

## Reproducing the numbers

```bash
# LongMemEval data (not in this repo — download from upstream)
# https://github.com/xiaowu0162/long-mem-eval → place at data/longmemeval_s_cleaned.json

# Phase 1 retrieval (R@5)
uv run python -m eval.runner --limit 100            # 100q stratified sample
uv run python -m eval.ablations                     # full ablation matrix

# Phase 2 belief layer (external)
uv run python -m eval.longmemeval_belief \
    --detector full-stack-v7 --include-history      # the 0.885 headline

# Phase 2 novelties
uv run python -m eval.non_commutative_eval          # 95.8% on 240 scenarios
uv run python -m eval.plasticity_on_real_logs      # LTD/Hebbian/LTP stats
uv run python -m eval.false_contradiction_eval     # 6% FP rate

# Full test suite
uv run pytest tests/ -q                             # 602 tests, ~10s
```

---

## Roadmap

**Shipped (v0.7, 2026-04):**
- Phase 1 retrieval pillar (R@5 = 1.000 on LongMemEval-KU)
- Phase 2 belief layer with non-destructive supersession, validity, pramāṇa, plasticity, adhyāsa, abhāva, counterfactual replay, contextuality, raw archive
- Sequential-event supersession detector with additive-veto (6% FP rate)
- Non-commutative belief evolution: empirical benchmark + 95.8% measurement
- LongMemEval-KU external eval (0.885 with current+history)
- MCP server, CLI, Streamlit viewer, Python library

**Next (v0.8):**
- Publish to pypi as `patha-memory`
- Phase 1 + Phase 2 integration inside the MCP server (retrieval-filtered supersession)
- Learned sequential-event classifier (target: FPR < 2%)
- LLM-in-the-loop scorer for external benchmarks

**Future:**
- Multi-user belief attribution (whose belief is it?)
- Bayesian confidence propagation
- Adapters for LangChain / LlamaIndex
- Full 500q LongMemEval S eval

---

## Acknowledgments

Built with [Claude Code](https://claude.ai/code) as a pair-programming partner. Architectural decisions, evaluation design, framing, and editorial honesty owned by the author. Code written in collaboration.

## License

MIT.
