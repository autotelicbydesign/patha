# Demo video storyboards — three options, ~60–90s each

Each storyboard is a complete shot list + voiceover + screen actions, optimized for a 60–90 second Loom / screen recording. Pick one (or remix). The video runs once on the README, the Show HN / Twitter / LinkedIn announcement, and the MCP registry listing — so it carries weight.

Stefi's earlier note: the original "memory persists across restart" framing is *false-marketing* (Claude Desktop already does that trivially). All three options below tell **true** stories that no other AI memory system can demonstrate.

---

## Option A — Synthesis-intent routing (recommended)

**The clearest architectural-claim demo.** Hooks on a question every developer has thought "that's exactly what RAG can't do."

**Total length**: 80–90 seconds.

### Voiceover script

> (0:00) Most AI memory systems treat memory as retrieval — top-K of N relevant chunks, then let the LLM clean up.
>
> (0:08) That works for *"what did I say about the saddle?"* It breaks for *"how much have I spent on bikes total?"* — because top-100 of 1000 sessions misses 90% of the inputs you'd need to sum.
>
> (0:18) Patha separates retrieval from synthesis.
>
> (0:22) Watch. I'm going to ingest four bike purchases across four sessions:
>
> *(screen: terminal showing four `patha ingest` commands)*
>
> (0:32) Now ask Patha *"how much have I spent on bike-related expenses?"* —
>
> *(screen: `patha ask "..."` shows answer = $185)*
>
> (0:38) — and inspect the result. Strategy: gaṇita. Source: a tuple index built at ingest. **Zero LLM tokens at recall.**
>
> *(screen: `patha-viewer` shows the four belief tuples + the sum, plus the proof-of-derivation chain)*
>
> (0:50) The synthesis answer is independent of retrieval. To prove it: I'll force the retrieval layer to return nothing, and rerun the same question.
>
> *(screen: a Python REPL: `mem._patha._phase1_retrieve = lambda q, k: []` then re-call `recall`)*
>
> (1:05) Same answer: $185. Recovered exhaustively from the tuple index.
>
> (1:12) Two layers, two paths through the system, one belief store. Local-first, Apache 2.0.
>
> (1:18) `pip install patha-memory`.

### Shot list

| Time | Visual |
|---|---|
| 0:00 | Cold open: README hero on github.com/autotelicbydesign/patha — show "Local-first AI memory designed from a different epistemology" |
| 0:08 | Quick cut: a typical RAG diagram (top-K vector search) on screen, narrator notes "this is the default" |
| 0:18 | Title card: **Synthesis vs Retrieval** |
| 0:22 | Terminal — start fresh:<br>`patha ingest "I bought a $50 saddle for my bike"`<br>`patha ingest "I got a $75 helmet for the bike"`<br>`patha ingest "$30 for new bike lights"`<br>`patha ingest "I spent $30 on bike gloves"` |
| 0:32 | `patha ask "how much have I spent on bike-related expenses?"` — answer comes back as `$185.00 USD` (instant, no spinner, no LLM call) |
| 0:42 | Switch to Streamlit viewer (`patha viewer`) — show the four `(entity=bike, amount=$X)` tuples on the gaṇita-index page; show the derivation chain: $50 + $75 + $30 + $30 = $185 |
| 0:55 | Python REPL — show `mem._patha._phase1_retrieve = lambda q, k: []` (force retrieval layer to return nothing); then `mem.recall("how much have I spent on bikes?").ganita.value` → `185.0` |
| 1:08 | Cut back to viewer; highlight `recall.tokens = 0` field |
| 1:15 | Closing card: `pip install patha-memory` + GitHub URL + Apache 2.0 |

### What this demonstrates that no other memory system does

- **Two routing paths** for two question classes
- **Zero LLM tokens at recall** on the synthesis path — verifiable, observable
- **Independence of synthesis from retrieval top-K** — provable by forcing top-K to be empty

### Watchouts

- Use a **fresh `Memory(path=...)`** in the demo so the demo is reproducible. Don't rely on accumulated state in `~/.patha`.
- **Show actual terminal output**, not pre-recorded text. Live runs are what make this credible.
- Ban any "memory persists across restart" framing. That's not what's interesting.

---

## Option B — Non-commutative belief evolution

**The most philosophically distinctive demo.** Hooks on something Mem0/MemPalace/etc. can't show: *order of arrival changes what you currently believe.*

**Total length**: 60–75 seconds.

### Voiceover script

> (0:00) When you change your mind, where does the old belief go?
>
> (0:06) In most AI memory systems: it gets overwritten, or buried in a vector store, or summarised away. Old beliefs disappear.
>
> (0:14) In Patha, old beliefs are *superseded* — moved to history with a full lineage you can walk.
>
> (0:22) But here's the deeper claim: **the order beliefs arrive in matters.** Your final state depends on the path you took to get there.
>
> *(screen: terminal — ingest order A→B)*
>
> (0:32) Forward order: I ingest *"I love sushi every week"*, then *"I'm avoiding raw fish on doctor's advice."*
>
> (0:40) Ask: *"what do I currently eat?"* → "I'm avoiding raw fish."
>
> *(screen: clear store, ingest reversed order B→A)*
>
> (0:48) Reverse order: same two beliefs, ingested in reverse. *"Avoiding raw fish"* first, *"love sushi every week"* second.
>
> (0:56) Same question. Different answer: "I love sushi every week." The fresh assertion supersedes the older one.
>
> (1:04) Most memory systems are commutative — the order doesn't matter, the bag of facts is the bag of facts. **On 240 supersession scenarios, Patha is non-commutative 95.8% of the time.**
>
> (1:15) That's not a bug. That's how human memory actually works.

### Shot list

| Time | Visual |
|---|---|
| 0:00 | Hero: a clip-art / illustration of two arrows pointing to different states |
| 0:14 | Quick architecture sketch: belief → supersession → history (not deletion) |
| 0:32 | Terminal A (forward order):<br>`patha ingest "I love sushi every week"`<br>`patha ingest "I'm avoiding raw fish on doctor's advice"`<br>`patha ask "what do I currently eat?"` |
| 0:42 | Streamlit viewer's "history" panel: show the supersession chain |
| 0:48 | Reset (`rm -rf /tmp/demo.jsonl`); Terminal B (reverse order):<br>`patha ingest "I'm avoiding raw fish on doctor's advice"`<br>`patha ingest "I love sushi every week"`<br>`patha ask "what do I currently eat?"` — answer flips |
| 1:04 | Title card: **95.8% non-commutative on 240 supersession scenarios** with a small graph from `eval/non_commutative_eval.py` output |
| 1:15 | Closing card with install line |

### What this demonstrates that no other memory system does

- Explicit non-destructive supersession with a queryable history
- An empirical *measurement* of non-commutativity (95.8% on a benchmark)
- A correctness-as-architecture stance: the same two facts produce different final states depending on the order they arrived

### Watchouts

- Use a `path=/tmp/demo-A.jsonl` and `/tmp/demo-B.jsonl` so the two timelines don't pollute each other.
- The order-matters framing is intellectually rich but takes longer to explain than option A. If your audience is engineering-skewed (HN), A lands faster. If your audience is design / philosophy / cognitive-science-leaning (your existing followers per `update_06_not_an_engineer`), B lands deeper.

---

## Option C — Cross-tool persistence in one belief store

**The most "I want to install this" demo.** Hooks on the same memory surfacing in multiple AI tools you already use.

**Total length**: 60–75 seconds.

### Voiceover script

> (0:00) Your AI memory shouldn't be locked inside one app's account.
>
> (0:06) Watch the same belief store feed three AI tools at once.
>
> *(screen split into thirds: Claude Desktop, Cursor, Streamlit viewer)*
>
> (0:14) I'm running Patha as an MCP server. Same `~/.patha/beliefs.jsonl` file, three clients.
>
> (0:22) In Claude Desktop I tell it: *"Remember that I'm vegetarian."* It calls `patha_ingest`.
>
> *(screen: Claude Desktop chat)*
>
> (0:30) I switch to Cursor — different app, same belief store. *"What do I eat?"* — Cursor's Claude calls `patha_query`. Answer: vegetarian.
>
> *(screen: Cursor chat)*
>
> (0:40) Now in the Streamlit viewer — read-only inspection of the belief store. I can see the ingest event timestamped, the current belief, the supersession lineage if there were one.
>
> *(screen: viewer panel)*
>
> (0:50) One file. Three tools. Local-first. No cloud. No API keys for the memory itself.
>
> (1:00) `~/.patha/beliefs.jsonl` is plain text. You can `cat` it. You can `git commit` it. You can copy it to another machine and your memory comes with you.
>
> (1:10) `pip install patha-memory`.

### Shot list

| Time | Visual |
|---|---|
| 0:00 | Title card: **One belief store, every AI tool you use** |
| 0:06 | Split screen: Claude Desktop chat (left), Cursor chat (right), terminal/Streamlit viewer (bottom) |
| 0:14 | Bottom panel: terminal showing `cat ~/.patha/beliefs.jsonl` (a few lines of JSONL) |
| 0:22 | Claude Desktop: type "Remember that I'm vegetarian." — show the MCP `patha_ingest` tool call popup |
| 0:30 | Quick switch to Cursor: ask "what do I eat?" — show the MCP `patha_query` tool call → "you're vegetarian" |
| 0:40 | Streamlit viewer panel: show the timeline event for the ingest, the current belief table |
| 0:50 | Bottom: `cat` the JSONL again, show the new line appended |
| 1:00 | Architecture diagram: three AI clients ↔ one MCP server ↔ one JSONL file |
| 1:10 | Closing card |

### What this demonstrates that no other memory system does

- Cross-tool, cross-process: every MCP-compatible AI client reads the *same* belief store
- Local-first: no cloud, no API keys, no SaaS
- Inspectable: the store is a plain text file you can read/edit/grep/version-control

### Watchouts

- The split-screen is technically demanding to record. Consider three sequential cuts instead — Claude Desktop → Cursor → viewer — with the JSONL line shown growing between cuts.
- This demo trades architectural-distinctiveness for "concrete user benefit." If the audience knows MCP already, this lands as "ah, finally." If they don't, the MCP framing is the wrong opener.

---

## My recommendation: **Option A (synthesis-intent routing)**

Reasoning:

1. **Highest architectural distinctiveness.** Synthesis vs retrieval is the central claim of v0.10. The other two options demonstrate features (supersession, MCP); option A demonstrates the *epistemological move* that justifies all of Patha.
2. **Hardest to fake.** "Zero LLM tokens at recall" is an observable, measurable claim. The audience can install Patha and verify. The "force the retrieval layer to return `[]`" sub-demo at 0:55 is the moment that converts skeptics — it proves the synthesis path doesn't depend on retrieval.
3. **Resonates with engineers and designers.** The $185 example is concrete enough for engineers; the "two paths through the system" framing is design-thinking enough for the cognitive-OS audience.
4. **Best single-take recording potential.** It's a coherent narrative: setup ($50 + $75 + $30 + $30) → result ($185) → architectural reveal (zero tokens, retrieval-independent). Options B and C have more shot transitions.

**If you want to do two**, do A as the README hero and B as a follow-up "deeper architecture" video. Don't do C alone — it underclaims.

---

## Production notes (apply to all three)

- **Recording tool**: Loom is fine; OBS gives more control. 1080p, ≤30 fps. No webcam overlay (keeps the focus on the demo).
- **Voiceover**: don't read the script verbatim. Record screen + narrate live. Re-take the whole take if you flub it; editing voice tracks against screen actions reads as edited and weakens the "I just installed this and ran it" feel.
- **Length discipline**: under 90 seconds. People close videos at 90s on Twitter, 60s on LinkedIn, 120s on a README hero. 80s is the sweet spot.
- **Subtitle the voiceover**: 80% of HN / Twitter views are sound-off. Loom auto-captions, but check them — the Sanskrit terms (*pratyakṣa*, *anumāna*, *gaṇita*) get autocorrected to nonsense.
- **Fresh `Memory(path=...)`**: use `/tmp/demo-A.jsonl` (or similar) so the recording is self-contained and doesn't leak any prior state into the demo.
- **No comparison framing**: nothing about "unlike Mem0" or "beats MemPalace" — even in the visual cuts. Stay on what Patha *does*, not what others *don't*.
- **Apache 2.0 + author attribution** in the closing card: "Built by Stefi P. Krishnan with Claude Code as a pair-programming partner. Apache 2.0."

---

## Once you've recorded

1. Upload to Loom (or YouTube unlisted)
2. Embed the link in:
   - The README hero (just under "At a glance")
   - The Show HN post (top of the comment thread, not the title)
   - The Twitter / X thread (tweet 2)
   - The LinkedIn post (header image position)
   - The MCP registry listing (in the description, if the schema allows)
3. Add a `<video>` reference to `docs/launch/launch-checklist.md` (next doc to draft) so the public assets all point at the same URL.

If Loom: the public-share link looks like `https://www.loom.com/share/<id>`. Test it in incognito to confirm it doesn't require a Loom account to view.
