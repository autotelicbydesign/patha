# How to use Patha

Patha is a tool that stores beliefs locally and makes them available to AI assistants. The tool has no UI, no rooms to walk through, no buttons to click.

You use it directly with your AI assistant. Or, better said, your AI uses it.

It acts as an additional brain for your AI. It can add information into AI's context and pull things back whenever it needs to — and unlike a memory palace, it also notices when your beliefs change and keeps the history.

## At its core, Patha

- **Stores your content exactly as it is**, not as AI-generated summaries (your beliefs live in a plain `~/.patha/beliefs.jsonl` file you can read, edit, grep, and version-control).
- **Retrieves through a redundant 7-view index** inspired by Vedic recitation: every belief is encoded in seven overlapping forms so a paraphrased question still finds the right belief.
- **Walks across sessions through a songline graph** inspired by Aboriginal songlines: shared entities, time, speaker, and topics form edges; retrieval traverses the network.
- **Tracks how your beliefs change over time**: when new evidence contradicts an old belief, the old one is marked *superseded* — not overwritten — so you can ask "what did I used to believe?"
- **Reduces your LLM token bill**: a structured Patha summary is ~4.5× smaller than a naive raw-history dump; aggregation questions can answer with zero LLM tokens.
- **Allows AI assistants to connect and use it** through MCP (Model Context Protocol).

It does not write for you, generate ideas, or think. Synthesis stays with you (or your LLM); Patha hands back faithful sources.

## Open & local-first

Patha is built on open architecture. The code is publicly available at [github.com/autotelicbydesign/patha](https://github.com/autotelicbydesign/patha), your beliefs live on your own computer in standard JSONL format, nothing is sent to the cloud, and you don't need an account or subscription to use it.

ChatGPT Memory, Notion AI, and Claude Projects use a closed architecture: code private, content on the company's servers in their format, accessed through their app.

## One memory, multiple AIs

Patha isn't tied to a single assistant. Through **Model Context Protocol (MCP)**, different AI tools can connect to the same memory:

- Claude Desktop
- Claude Code
- Cursor
- Zed
- Goose
- Any future MCP-compatible client

The same `~/.patha/beliefs.jsonl` file is the source of truth for all of them. No duplication; no per-app silos.

## How your data is structured

Patha doesn't use spatial metaphors (no rooms, halls, drawers). The structure is information-theoretic:

| Element | Role |
|---|---|
| **Belief** | One atomic claim, asserted at a time, by a session, with optional context |
| **Status** | `current` / `superseded` / `disputed` / `archived` |
| **Pramāṇa** | Source-of-knowledge tag (perception, inference, testimony, etc.) — Nyāya epistemology |
| **Validity** | Time-window the belief holds (`permanent`, `dated_range`, `duration`, `decay`) |
| **Plasticity state** | Confidence, reinforcement count, Hebbian edges to co-retrieved beliefs |
| **7 views** (Vedic) | Each belief stored in 7 overlapping forms (pada, krama, jaṭā, ghana, entity-anchored, reframed, temporally-anchored) |
| **Songline edges** (Aboriginal) | Cross-belief edges by shared entity, time, speaker, topic cluster |

Each element maps to a specific retrieval or maintenance operation. This is what keeps retrieval accurate as the archive grows — the redundancy isn't decorative.

## Memory as a design medium

Memory is becoming the most important, and least understood, part of AI. It shapes what AI sees and what it produces.

Yet for most users, memory feels like a black box. Something the AI just has. You may see what's stored, but not what's retrieved, or what actually gets used when the AI responds. For many of us designing AI products, it can feel like a black box too — a technical layer we build around rather than design into.

Patha points to a different approach: **keep the original data local, keep retrieval auditable, ground outputs in real context, and surface the history of what changed**.

You can:
- **`patha viewer`** — open a Streamlit dashboard showing every belief, when it was asserted, what superseded what, the supersession graph, and a non-commutativity replay tool.
- Open `~/.patha/beliefs.jsonl` in any text editor or run `grep` over it.
- `git init` your `~/.patha/` directory and version-control your memory.

## A guide for those who wish to install

The simplest install is via the Patha install helper. From a clone of the repo:

```bash
# Clone & install
git clone https://github.com/autotelicbydesign/patha.git
cd patha
uv sync

# Verify the environment
uv run patha verify

# Run the 10-second demo (no model downloads)
uv run patha demo

# Set up Claude Desktop integration (one command, detects your OS)
uv run patha install-mcp

# (or for Claude Code)
uv run patha install-mcp --client claude-code
```

Quit and reopen Claude Desktop. Open Settings → Developer → MCP Servers (or look for the tools icon). You should see **patha** with four tools:

- `patha_ingest` — remember a fact
- `patha_query` — ask what you currently believe
- `patha_history` — see every mention of a term
- `patha_stats` — counts and storage info

That's it. The entire setup is one command per AI tool.

The first time the production detector is used, an NLI model (~1.7 GB) downloads as part of the system. Pre-download it with `PATHA_DETECTOR=full-stack-v8 uv run patha verify --preload` if you want to avoid the first-call lag.

## Adding data to Patha

There are now four ways to add data to Patha:

1. **Through your AI** — say "remember that I live in Lisbon" or "remember my rent is $1500." Claude Desktop will call `patha_ingest`.
2. **Through the CLI** — `patha ingest "I live in Lisbon"`. Useful for scripts, hotkeys, or shell pipelines.
3. **Through the Python API** — `import patha; m = patha.Memory(); m.remember("I live in Lisbon")`. For developers building chatbots.
4. **Through your existing notes** — `patha import obsidian-vault ~/MyVault` walks an Obsidian vault (or a plain Markdown folder, or a single text file) and turns it into beliefs. Frontmatter `date` becomes `asserted_at`. Wikilinks and tags become entity hints.

Beliefs are time-stamped automatically and persisted to `~/.patha/beliefs.jsonl` immediately.

## How to use it

Patha works with any MCP-compatible AI tool. Two ways to interact:

- **Natural language**: "Remember that I'm vegetarian" / "What do I do for work?" — your AI calls the right tool.
- **Slash-style commands** through your AI tool's plugin model (e.g., Claude Code skills).

## What I tested

To test how it works, I put 4 beliefs into Patha through the Python API:

```
"I bought a $50 saddle for my bike"
"I got a $75 helmet for the bike"
"$30 for new bike lights"
"I spent $30 on bike gloves"
```

Then asked: *"How much total money have I spent on bike-related expenses?"*

Patha's gaṇita layer (Vedic-tradition arithmetic-as-preservation, no LLM) returned:

> **$185.0 USD** — sum of $50 + $75 + $30 + $30, computed deterministically from 4 source belief ids.

The synthesis was procedural. No interpretation, no LLM call, just preserved facts + rule-application — exactly how the Vedic *Sulbasūtras* compute geometric properties from preserved geometric facts.

For paraphrase-robust retrieval, I asked: *"Do I eat raw fish?"*

Patha returns "I love sushi every week" — even though "sushi" and "raw fish" share zero tokens. The 7-view index catches the semantic match.

## How Patha compares to MemPalace

Both are local-first, MCP-served, audit-friendly. Different design choices:

| | MemPalace | Patha |
|---|---|---|
| Metaphor | Greek method of loci (spatial) | Vedic recitation + Aboriginal songlines (information-theoretic) |
| Storage | Wings → halls → rooms → drawers | Beliefs in JSONL, 7-view index, songline graph |
| Retrieval | Spatial-index lookup | 7-view dense + BM25 + RRF + cross-encoder + songline walks + **Hebbian cluster expansion** |
| Belief layer | Stores verbatim, no contradiction handling | Non-destructive supersession + contradiction detection + plasticity (now read at runtime) |
| Synthesis layer | LLM at every recall | **Karaṇa**: LLM at ingest only (optional), zero LLM at recall on aggregation questions |
| Token economy | Not measured (afaik) | Measured — 4.5× reduction on structured, ∞ on direct-answer |
| LongMemEval-S 500q (R@5 / end-to-end) | 0.966 | 1.000 (Phase 1 R@5) / 0.952 (end-to-end) |
| Filesystem-native ingest | Obsidian via separate workflow | `patha import obsidian-vault` built into CLI |
| MCP integration | yes | yes (`make mcp-install`) |
| Cross-tool | yes (MCP) | yes (MCP) |

Patha is more architecturally complex on purpose: it's not just retrieval, it's also belief management (supersession, contradiction, plasticity, non-commutative belief-order tracking). MemPalace's spatial framing is more approachable; Patha's two-traditional framing comes with more depth at the cost of a steeper conceptual landing.

## Three innovations that target Patha's known weaknesses

The honest gaps below have been the focus of recent work. Each gap got a tradition-faithful innovation aimed at it directly:

### 1. Hebbian-cluster-aware retrieval (multi-session gap)

Patha's belief layer already accumulated **Hebbian co-retrieval edges** between beliefs that surfaced in the same query — a neuroplasticity mechanism. That signal was recorded but never read back. Now it is. After Phase 1 retrieves the candidate set, we walk each seed belief's strongest Hebbian neighbors and add them. Beliefs that have surfaced together repeatedly co-surface again, even if cosine similarity to the new query is borderline. The first time the user asks something cross-session, plain Phase 1 has to find both. After that, the Hebbian edges remember they go together.

To bootstrap a fresh store with no query history, every pair of beliefs asserted in the same session gets a small seed weight (`hebbian_session_seed_weight=0.05`).

### 2. Vedic *karaṇa* ingest-time LLM extraction (synthesis-bounded gap)

The Vedic word *karaṇa* means "preparation" — work done in advance so the moment of performance can be deterministic. For Patha, the same idea splits the gaṇita extractor:

- **Ingest** — a small local LLM (default `qwen2.5:7b-instruct` via Ollama) reads each new belief and returns a structured list of `(entity, attribute, value, unit)` tuples. Once per belief.
- **Recall** — pure deterministic arithmetic over the preserved index. **Zero LLM tokens.**

This is the inverse of mainstream RAG, which spends tokens at recall every query. Most users ask the same things many times; spending tokens once at ingest is a strict win, plus recall is reproducible to the cent. Opt-in via `Memory(karana_extractor=OllamaKaranaExtractor())`. Falls back to the regex extractor when Ollama isn't reachable.

### 3. Filesystem-native ingest (first-time setup gap)

`patha import obsidian-vault ~/MyVault` (or `import folder ~/notes/`, or `import file ~/note.md`) walks pre-existing writing — Obsidian, plain Markdown, .txt — and ingests it. Frontmatter `date` becomes `asserted_at`. Wikilinks and `#tags` become entity hints for the gaṇita extractor. Long files split on H1/H2 boundaries. Hidden directories (`.git`, `.obsidian`) are skipped. Read-only — your source files are never touched. The friction of "what do I do after `uv sync`?" drops to "point Patha at the writing you already have."

## Where Patha is still honestly weaker

- **Synthesis-bounded questions on dense conversational text without an LLM** — the regex baseline gaṇita extractor handles clean facts ("I bought a $50 saddle for the bike") but isn't reliable on multi-paragraph LongMemEval haystacks. The karaṇa extractor (innovation #2) closes this when Ollama is available; without Ollama, this gap is real.
- **Multi-session retrieval on the very first query** — Hebbian-cluster-aware retrieval (innovation #1) helps once any query history exists or session-seeding is on. A truly cold-start single query still relies entirely on Phase 1.
- **First-time setup beyond filesystem import** — `uv sync` still required for now. Pip-install via `pip install patha-memory` lands when v0.10 publishes.

These are honest gaps, documented in `docs/benchmarks.md` and on the issue tracker.

## What Patha guarantees

- Your beliefs are yours. Plain JSONL, on your disk. Move them, edit them, delete them, share them.
- Your beliefs are preserved. When you change your mind, the old belief moves to *superseded*, not gone.
- Your beliefs are inspectable. Open the file. Open the viewer. See the supersession graph.
- Your LLM bill goes down. ~4.5× fewer input tokens for the same memory context; aggregation questions answered with zero LLM tokens.
- Your memory works across tools. One file, every MCP-compatible client.
