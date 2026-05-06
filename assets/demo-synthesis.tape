# Patha synthesis-intent routing demo
#
# Runs the canonical $185-across-4-sessions demo end-to-end through
# the public `patha` CLI, showing strategy=ganita and tokens=0 at recall.
#
# Reproduce: vhs assets/demo-synthesis.tape   (writes assets/demo-synthesis.gif)

Output assets/demo-synthesis.gif

Set FontSize 18
Set Width 1200
Set Height 720
Set Theme "Catppuccin Mocha"
Set TypingSpeed 40ms
Set Padding 24

# ─── Setup (hidden) ─────────────────────────────────────────────────
Hide
Type "cd /Users/stefi/Sandbox/Memory"
Enter
Type "rm -rf /tmp/patha-demo && export PATHA_STORE_PATH=/tmp/patha-demo"
Enter
Type "alias patha='uv run --quiet patha'"
Enter
Type "clear"
Enter
Sleep 200ms
Show

# ─── Title ──────────────────────────────────────────────────────────
Type "# Patha: synthesis-intent routing"
Sleep 800ms
Enter
Type "# 'How much have I spent on bikes total?' is not a top-K question."
Sleep 1500ms
Enter
Enter
Sleep 400ms

# ─── Step 1: Four bike purchases, four sessions ─────────────────────
Type "# 1. Ingest four bike purchases, each in its own session"
Enter
Sleep 400ms

Type "patha ingest 'I bought a $50 saddle for my bike'"
Enter
Sleep 2000ms

Type "patha ingest 'I got a $75 helmet for the bike'"
Enter
Sleep 2000ms

Type "patha ingest '$30 for new bike lights'"
Enter
Sleep 2000ms

Type "patha ingest 'I spent $30 on bike gloves'"
Enter
Sleep 2000ms

Enter
Sleep 400ms

# ─── Step 2: The synthesis question ─────────────────────────────────
Type "# 2. Ask the synthesis question: sum across all 4 sessions"
Enter
Sleep 600ms

Type 'patha ask "how much have I spent on bike-related expenses?"'
Enter
Sleep 4500ms

Enter
Sleep 200ms

# ─── Step 3: The reveal ─────────────────────────────────────────────
Type '# strategy=ganita    tokens=0    answer=$185'
Sleep 1500ms
Enter
Type "# Sum is deterministic, derived exhaustively from a tuple index."
Sleep 1200ms
Enter
Type "# No LLM was called at recall."
Sleep 2000ms
Enter
Enter

Type "# pip install patha-memory     github.com/autotelicbydesign/patha"
Sleep 4000ms

# Hold the closing frame
Sleep 1500ms
