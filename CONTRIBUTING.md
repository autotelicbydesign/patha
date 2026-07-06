# Contributing to Patha

Thanks for your interest. Patha is early and moving fast; this page keeps the
loop short.

## Dev setup

```bash
git clone https://github.com/autotelicbydesign/patha
cd patha
uv sync                                   # Python 3.11+; uv manages the env
uv run python -m spacy download en_core_web_sm   # once, for entity extraction
uv run patha verify                       # environment sanity check
uv run pytest tests/ -q                   # the suite must be green before and after
```

The heavy NLI/embedding models lazy-download on first use. `stub` detector
needs nothing and is what most tests use.

## Ground rules

- **Tests accompany changes.** Bug fix → regression test that fails without
  the fix. Feature → tests for the new surface. `uv run pytest tests/ -q`
  green is the bar; CI runs 3.11 + 3.12.
- **Style is `ruff`.** `uv run ruff check src/ tests/` before pushing.
- **Benchmarks are versioned instruments, not decoration.** Detector stacks
  (`full-stack-v7/v8/v9`) are frozen once published — behaviour changes ship
  as a NEW stack version. The EvolutionEval rubric changes only with a
  version bump + re-report. Never run or tune against a sealed held-out set;
  the runner refuses them without `--include-heldout` for a reason. Read
  [docs/benchmarks.md](docs/benchmarks.md) before touching anything in
  `eval/` — the protocol notes are binding.
- **Honest numbers only.** If your change moves a published number, update
  `docs/benchmarks.md` with the new measurement and how to reproduce it. If
  it moves a number *down*, say so plainly — this repo publishes its gaps.

## Pull requests

1. Fork, branch from `main` (`fix/...` or `feat/...`).
2. Keep PRs focused — one behaviour change per PR.
3. Describe **what** changed, **why**, and **how you verified it** (test
   names, benchmark runs).
4. CI must pass.

## Issues

Use the issue templates. For bugs: your OS/Python, the detector in use
(`stub` vs `full-stack-*`), and a minimal reproduction — ideally the exact
`patha` commands or a short Python snippet. For benchmark-number questions,
link the table row you're asking about; every published number has a
reproduce command in [docs/benchmarks.md](docs/benchmarks.md).

## Where help is most valuable right now

- **Karaṇa (tuple extraction)** — the multi-session synthesis gap is the top
  open capability problem (see the roadmap in the README).
- **Supersession precision** — refinement-vs-revision discrimination
  (EvolutionEval rubric v2 measures it; the fix program is open).
- Importers for new sources; MCP client walkthroughs beyond Claude Desktop.
