# Running the Frontier-LLM Articulation Bridge measurement

The current published Articulation Bridge number is **0.308** on KU 78q with `qwen2.5:14b-instruct` (local Ollama, token-overlap ≥0.6). That's a *floor* for "real LLMs in the loop" — a 14B local quantized model. A frontier-class model (Claude Sonnet 4 / Sonnet 4.5, GPT-4o, Gemini 2.5 Pro) would likely lift this to the 0.7–0.9 range and substantially strengthen the launch story.

This doc is the step-by-step for running it once you have an API key.

## Prerequisites

- Anthropic API key. Get one at https://console.anthropic.com/settings/keys.
- ~$0.50–$2 in API spend. KU 78q × ~500 tokens context × ~50 tokens output × Claude Sonnet 4 pricing.
- ~10 minutes of wall time.

## Step 1 — paste the key into `.env`

In the repo root:

```bash
cp .env.example .env   # if you haven't already
# Edit .env and put your real key in ANTHROPIC_API_KEY=sk-ant-...
```

`.env` is in `.gitignore` and won't be committed.

## Step 2 — load the key into your shell session

```bash
set -a; source .env; set +a
echo "key length: ${#ANTHROPIC_API_KEY}"   # should be ~108
```

(`uv run` and `python -m` inherit the environment, so this is enough.)

## Step 3 — run the eval

```bash
uv run python -m eval.run_answer_eval \
    --data data/longmemeval_ku_78.json \
    --llm claude --claude-model claude-sonnet-4-20250514 \
    --scorer overlap \
    --output runs/answer_eval/ku-claude-sonnet4-overlap.json \
    --verbose
```

Default behavior:
- Reads each KU question
- Builds a fresh `patha.Memory` per question, ingests the chronological haystack
- Calls `memory.recall(question)` → renders into the default Articulation Bridge prompt template
- POSTs the prompt to the Anthropic Messages API with `claude-sonnet-4-20250514`
- Scores the response with `token_overlap_match(threshold=0.6)` (LongMemEval-S official scorer)
- Aggregates and writes per-question JSON + summary

Expected output (last few lines):

```
results: NN/78 = 0.XXX  (~600s)
by strategy:
  ganita              X/41 = 0.YYY
  structured          Y/37 = 0.ZZZ
by question_type:
  knowledge-update    NN/78 = 0.XXX

wrote: runs/answer_eval/ku-claude-sonnet4-overlap.json
```

## Step 4 — re-score under multiple scorers (free; no extra API spend)

The output JSON has the per-question Claude answer text. Re-score under any scorer with no further API calls:

```bash
uv run python3 << 'PYEOF'
import json, sys, os
sys.path.insert(0, os.path.abspath("."))
from eval.answer_eval import (
    numeric_match, normalised_match, token_overlap_match, embedding_cosine_match
)

d = json.load(open("runs/answer_eval/ku-claude-sonnet4-overlap.json"))
outcomes = d['outcomes']

scorers = {
    "numeric (5% tol)": numeric_match(tol=0.05),
    "token_overlap ≥0.6 (LongMemEval-S official)": token_overlap_match(threshold=0.6),
    "token_overlap ≥0.4": token_overlap_match(threshold=0.4),
    "embedding_cosine ≥0.85": embedding_cosine_match(threshold=0.85),
    "embedding_cosine ≥0.55": embedding_cosine_match(threshold=0.55),
}

for name, scorer in scorers.items():
    correct = sum(1 for o in outcomes if scorer(o['answer'], o['gold']))
    print(f"  {name:42s} {correct:>3}/{len(outcomes)} = {correct/len(outcomes):.3f}")
PYEOF
```

## Step 5 — update README + benchmarks.md, bump v0.10.8

Once you have the number you want to publish (likely the token-overlap ≥0.6 result), the diff is:

1. **README.md** — replace the "Frontier-LLM measurement pending" line with the actual number:
   - *"Real measurement on KU 78q (Claude Sonnet 4, token-overlap ≥0.6 — the LongMemEval-S official scorer): **0.XXX (NN/78)**"*
2. **README.md headlines table** — same change in the Articulation Bridge row
3. **docs/benchmarks.md** Phase 3 section — add Claude row to the table next to qwen2.5:14b
4. **CHANGELOG.md + docs/releases/v0.10.8.md** — document the new measurement
5. **Bump version**: `pyproject.toml`, `__init__.py`, `scripts/verify_install.sh`
6. **Build + smoke + tag + upload to PyPI** following the same flow as v0.10.6 / 0.10.7

Total ~30 minutes after the run finishes.

## Failure modes — what to watch for

- **`ClaudeLLM: no API key.`** — the env var didn't propagate to the subprocess. Re-run `set -a; source .env; set +a` and check `echo $ANTHROPIC_API_KEY`.
- **Anthropic 429 rate-limit** — the runner makes one API call per question, sequentially. If you hit a rate limit, the run will fail mid-way; the existing per-question outcomes aren't preserved (the runner doesn't checkpoint). Re-running is fine; the only cost is the API spend.
- **Low score (<0.6 even with frontier model)** — would suggest either the retrieval is missing too many gold answers (check `o['summary_tokens']` and `o['strategy']`) or the prompt template is suboptimal. Investigate before publishing.

## Cost / time table

| LLM | Approx wall time | Approx cost | Expected accuracy on KU |
|---|---|---|---|
| Claude Sonnet 4 (`claude-sonnet-4-20250514`) | ~10 min | $0.50–$2 | 0.75–0.92 (estimated) |
| Claude Sonnet 4.5 | ~10 min | $0.50–$2 | likely slightly higher |
| GPT-4o | similar | similar | similar range |
| qwen2.5:14b local (already published) | 12 min | $0 | 0.308 (measured) |
| NullTemplateLLM (already published, deterministic floor) | 12 s | $0 | 0.064 (measured) |

The frontier-LLM run is the highest-leverage missing item for the launch — it likely lifts the headline Articulation Bridge number by 2–3×.
