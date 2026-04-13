.PHONY: test eval eval-quick eval-32 lint typecheck clean

# ── Testing ──────────────────────────────────────────────────────────

test:
	uv run python -m pytest tests/ -q

test-verbose:
	uv run python -m pytest tests/ -v

# ── Evaluation ───────────────────────────────────────────────────────

# Smoke test: 10 questions, fast, no reranker
eval-quick:
	uv run python -m eval.runner \
		--data data/longmemeval_s_cleaned.json \
		--embedder minilm \
		--limit 10 \
		--output runs/quick/results.json \
		--verbose

# 32-question diverse sample with cross-encoder
eval-32:
	uv run python -m eval.runner \
		--data data/longmemeval_s_cleaned.json \
		--embedder minilm \
		--reranker ce-mini \
		--limit 32 \
		--output runs/eval_32/results.json \
		--verbose

# 100-question stratified sample with cross-encoder
eval-100:
	uv run python -m eval.runner \
		--data /tmp/patha_stratified_100.json \
		--embedder minilm \
		--reranker ce-mini \
		--output runs/eval_100/results.json \
		--verbose

# Full 500-question eval with pre-warming + cross-encoder
eval:
	uv run python -m eval.runner \
		--data data/longmemeval_s_cleaned.json \
		--embedder minilm \
		--reranker ce-mini \
		--pre-warm \
		--output runs/full/results.json \
		--verbose

# ── Ablations ────────────────────────────────────────────────────────

ablation-no-reranker:
	uv run python -m eval.runner \
		--data /tmp/patha_stratified_100.json \
		--embedder minilm \
		--reranker none \
		--output runs/ablation_no_reranker/results.json \
		--verbose

ablation-no-songline:
	uv run python -m eval.runner \
		--data /tmp/patha_stratified_100.json \
		--embedder minilm \
		--reranker ce-mini \
		--no-songline \
		--output runs/ablation_no_songline/results.json \
		--verbose

# ── Code Quality ─────────────────────────────────────────────────────

lint:
	uv run ruff check src/ eval/ tests/

lint-fix:
	uv run ruff check --fix src/ eval/ tests/

typecheck:
	uv run mypy src/patha/ --ignore-missing-imports

# ── Cleanup ──────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf .ruff_cache .mypy_cache
