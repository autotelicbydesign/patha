# Patha demo 2: When you change your mind, AI memory should change too
#
# Uses `patha shell` for natural conversational typing.
# Reproduce: vhs assets/demo-supersession.tape

Output assets/demo-supersession.gif

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
Type "rm -rf /tmp/patha-demo2 && export PATHA_STORE_PATH=/tmp/patha-demo2"
Enter
Type "clear"
Enter
Sleep 200ms
Show

# ─── Title ──────────────────────────────────────────────────────────
Type "# When you change your mind, where does the old belief go?"
Sleep 2000ms
Enter
Type "# Most AI memory just overwrites it. Patha doesn't."
Sleep 2500ms
Enter
Enter

# ─── Drop into the REPL ─────────────────────────────────────────────
Type "uv run --quiet patha shell --detector full-stack-v7"
Enter
Sleep 2500ms

# ─── June: love sushi ───────────────────────────────────────────────
Type "I love sushi and eat it every week"
Sleep 700ms
Enter
Sleep 2500ms

# ─── December: avoiding raw fish ───────────────────────────────────
Type "I am avoiding raw fish on my doctor advice"
Sleep 700ms
Enter
Sleep 4000ms

# ─── Ask: what now? ────────────────────────────────────────────────
Type "? what do I currently eat?"
Sleep 800ms
Enter
Sleep 4500ms

# ─── Exit + reveal ──────────────────────────────────────────────────
Type "exit"
Enter
Sleep 1500ms

Type "# Patha caught the contradiction."
Sleep 2000ms
Enter
Type "# The new belief supersedes the old one."
Sleep 2000ms
Enter
Type "# But the old belief is not deleted. Walk the history:"
Sleep 2500ms
Enter
Enter

Type "uv run --quiet patha history sushi"
Enter
Sleep 4500ms

Type "# Old belief: superseded. New belief: current."
Sleep 2500ms
Enter
Type "# This is how human memory actually works."
Sleep 2500ms
Enter
Enter
Type "# Try it: pip install patha-memory"
Sleep 4000ms

Sleep 1500ms
