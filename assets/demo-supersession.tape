# Patha demo 2: When you change your mind, AI memory should change too
#
# Shows non-destructive supersession + history walking.
# Reproduce: vhs assets/demo-supersession.tape   (writes assets/demo-supersession.gif)

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
Type "alias patha='uv run --quiet patha --detector full-stack-v7'"
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

# ─── Setup story: June ──────────────────────────────────────────────
Type "# In June, you told your AI:"
Enter
Sleep 700ms

Type "patha ingest 'I love sushi and eat it every week'"
Enter
Sleep 2500ms

Enter

# ─── Setup story: December ──────────────────────────────────────────
Type "# Six months later, things changed:"
Enter
Sleep 700ms

Type "patha ingest 'I am avoiding raw fish on my doctor advice'"
Enter
Sleep 3500ms

Enter

# ─── The question (current) ────────────────────────────────────────
Type "# Now you ask: 'what do I currently eat?'"
Enter
Sleep 700ms

Type 'patha ask "what do I currently eat?"'
Enter
Sleep 4500ms

Enter
Type "# Patha caught the contradiction. The new belief supersedes the old."
Sleep 3000ms
Enter
Enter

# ─── The history ───────────────────────────────────────────────────
Type "# But the old belief is not deleted. It is filed under history:"
Enter
Sleep 800ms

Type "patha history sushi"
Enter
Sleep 4000ms

Enter

# ─── Plain-language reveal ─────────────────────────────────────────
Type "# You can ask: 'what did I used to think about sushi?' "
Sleep 2200ms
Enter
Type "# Patha will show you the timeline."
Sleep 2200ms
Enter
Enter
Type "# This is how human memory actually works. AI memory should match."
Sleep 3000ms
Enter
Enter
Type "# Try it: pip install patha-memory"
Sleep 4000ms

Sleep 1500ms
