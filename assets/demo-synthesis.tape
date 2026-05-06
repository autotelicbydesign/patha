# Patha demo 1: AI memory that does math
#
# Reframed for a non-technical reader.
# Reproduce: vhs assets/demo-synthesis.tape   (writes assets/demo-synthesis.gif)

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
Type "alias patha='uv run --quiet patha'"
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

# ─── Setup framing ──────────────────────────────────────────────────
Type "# Over the last few months, you mentioned to your AI:"
Enter
Sleep 700ms

Type "patha ingest 'I bought a $50 saddle for my bike'"
Enter
Sleep 1800ms

Type "patha ingest 'I got a $75 helmet for the bike'"
Enter
Sleep 1800ms

Type "patha ingest '$30 for new bike lights'"
Enter
Sleep 1800ms

Type "patha ingest 'I spent $30 on bike gloves'"
Enter
Sleep 2200ms

Enter

# ─── The question ──────────────────────────────────────────────────
Type "# Today you ask:"
Enter
Sleep 700ms

Type 'patha ask "how much have I spent on bike-related expenses?"'
Enter
Sleep 5000ms

Enter

# ─── Plain-language reveal ─────────────────────────────────────────
Type "# Patha added them up. $50 + $75 + $30 + $30 = $185."
Sleep 2200ms
Enter
Type "# It did NOT ask another AI to do the math."
Sleep 2200ms
Enter
Type "# It computed the answer directly. Zero LLM tokens at recall."
Sleep 2500ms
Enter
Type "# That is faster, cheaper, and always correct."
Sleep 2500ms
Enter
Enter
Type "# Try it: pip install patha-memory"
Sleep 4000ms

Sleep 1500ms
