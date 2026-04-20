"""Patha — developer quickstart.

For developers building LLM apps: add local memory in 5 lines. This
example shows how Patha compresses user memory into a ~20-token
summary you can paste into any LLM prompt, replacing the ~280-token
naive-history dump.

Run:

    uv run python examples/developer_quickstart.py

Expected output: a trace of (ingest → recall → prompt) with the
compact summary that would go to the LLM.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import patha


def main() -> None:
    # 1. Open a memory store (defaults to ~/.patha/beliefs.jsonl;
    #    pass path=... for a project-specific store).
    memory = patha.Memory(
        path="/tmp/patha-demo.jsonl",
        detector="full-stack-v8",  # NLI + adhyasa + numerical + sequential + learned
    )

    # 2. Ingest things the user told you about themselves. Patha's
    #    belief layer handles contradictions and supersession
    #    automatically.
    today = datetime.now()
    last_week = today - timedelta(days=7)

    print("── Ingesting ──")
    for claim, when in [
        ("I live in Sofia",                             last_week),
        ("I am vegetarian",                             last_week),
        ("I work as an AI engineer at Anthropic",       last_week),
        ("I just moved to Lisbon and no longer live in Sofia",  today),
    ]:
        result = memory.remember(claim, asserted_at=when)
        print(f"  [{result['action']:>11}] {claim}")

    # 3. When the user asks a question, recall() returns a compact
    #    summary. Pass `.summary` directly into your LLM system prompt.
    print()
    print("── Recall ──")
    user_msg = "What do you know about me?"
    rec = memory.recall(user_msg)
    print(f"  strategy: {rec.strategy}")
    print(f"  tokens:   {rec.tokens}")
    print(f"  answer:   {rec.answer}")
    print()
    print("  summary (this goes into the LLM system prompt):")
    for line in rec.summary.split("\n"):
        print(f"  │ {line}")

    # 4. History shows contradictions / supersessions over time.
    print()
    print("── History for 'Sofia' (what did I used to think?) ──")
    for m in memory.history("Sofia"):
        status = m["status"]
        print(f"  [{status}] {m['proposition']} @ {m['asserted_at'][:10]}")

    # 5. Stats for budgeting.
    print()
    print("── Stats ──")
    for k, v in memory.stats().items():
        print(f"  {k:10}: {v}")

    # 6. The belief store is a plain JSONL file at memory.path.
    #    Copy it, git commit it, edit it in a text editor.
    print()
    print(f"── All stored at: {memory.path}")
    print("   (plain JSONL — portable, inspectable, yours)")


if __name__ == "__main__":
    main()
