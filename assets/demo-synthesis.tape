# Patha demo 1: AI memory that does math
#
# Uses `patha shell` so the typing reads as natural conversation,
# not "patha ingest" boilerplate.
# Reproduce: vhs assets/demo-synthesis.tape

Output assets/demo-synthesis.gif

Set FontSize 20
Set Width 1200
Set Height 900
Set Theme "Catppuccin Mocha"
Set TypingSpeed 40ms
Set Padding 28

# ─── Setup (hidden) ─────────────────────────────────────────────────
Hide
Type "cd /Users/stefi/Sandbox/Memory"
Enter
Type "rm -rf /tmp/patha-demo && export PATHA_STORE_PATH=/tmp/patha-demo"
Enter
Type "clear"
Enter
Sleep 200ms
Show

# ─── Title ──────────────────────────────────────────────────────────
Type "# Tell your AI things over time. Ask it for totals."
Sleep 1500ms
Enter
Type "# Most AI memory cannot do this. Watch."
Sleep 2200ms
Enter
Enter

# ─── Drop into the REPL ─────────────────────────────────────────────
Type "uv run --quiet patha shell"
Enter
Sleep 2500ms

# ─── Four bike purchases, conversational ────────────────────────────
Type "I bought a $50 saddle for my bike"
Sleep 600ms
Enter
Sleep 1800ms

Type "I got a $75 helmet for the bike"
Sleep 600ms
Enter
Sleep 1800ms

Type "$30 for new bike lights"
Sleep 600ms
Enter
Sleep 1800ms

Type "I spent $30 on bike gloves"
Sleep 600ms
Enter
Sleep 1800ms

# ─── The synthesis question ─────────────────────────────────────────
Type "? how much have I spent on bike-related expenses?"
Sleep 800ms
Enter
Sleep 5000ms

# ─── Exit + reveal ──────────────────────────────────────────────────
Type "exit"
Enter
Sleep 1500ms

Type "# Patha added them up. $50 + $75 + $30 + $30 = $185."
Sleep 2200ms
Enter
Type "# It did NOT call another AI to do the math."
Sleep 2200ms
Enter
Type "# It computed the answer directly. Zero LLM tokens at recall."
Sleep 2500ms
Enter
Type "# Faster, cheaper, always correct."
Sleep 2500ms
Enter
Enter
Type "# Try it: pip install patha-memory"
Sleep 4000ms

Sleep 1500ms
