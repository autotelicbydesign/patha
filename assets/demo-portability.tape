# Patha demo 3: Your AI memory belongs to you
#
# Shows the local-first, plain-text, portable-file architecture.
# Reproduce: vhs assets/demo-portability.tape   (writes assets/demo-portability.gif)

Output assets/demo-portability.gif

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
Type "rm -rf /tmp/patha-demo3 && export PATHA_STORE_PATH=/tmp/patha-demo3"
Enter
Type "alias patha='uv run --quiet patha'"
Enter
Type "clear"
Enter
Sleep 200ms
Show

# ─── Title ──────────────────────────────────────────────────────────
Type "# Where does your AI memory actually live?"
Sleep 2000ms
Enter
Type "# In a SaaS company database? On your computer? You should know."
Sleep 2800ms
Enter
Enter

# ─── Tell the AI things ─────────────────────────────────────────────
Type "# You tell Patha some things about yourself:"
Enter
Sleep 700ms

Type "patha ingest 'I am vegetarian'"
Enter
Sleep 2000ms

Type "patha ingest 'I live in Lisbon'"
Enter
Sleep 2000ms

Type "patha ingest 'I am learning Portuguese'"
Enter
Sleep 2200ms

Enter

# ─── The reveal: it's just a file ──────────────────────────────────
Type "# Where did Patha put them? In a text file on your laptop:"
Enter
Sleep 800ms

Type "ls /tmp/patha-demo3/"
Enter
Sleep 2200ms

Enter
Type "# You can open it with any editor:"
Enter
Sleep 700ms

Type "head -3 /tmp/patha-demo3/beliefs.jsonl"
Enter
Sleep 4000ms

Enter

# ─── Same file, different tool ─────────────────────────────────────
Type "# Same file is read by every AI tool that supports MCP."
Enter
Sleep 800ms
Type "# Claude Desktop, Cursor, Zed, your terminal: same memory."
Enter
Sleep 2500ms

Type "patha ask 'what do I eat?'"
Enter
Sleep 4500ms

Enter

# ─── Plain-language reveal ─────────────────────────────────────────
Type "# No cloud. No login. No SaaS account."
Sleep 2200ms
Enter
Type "# You can grep it. Git-commit it. Copy it to another machine."
Sleep 2500ms
Enter
Type "# Your AI memory belongs to you, not to a company."
Sleep 3000ms
Enter
Enter
Type "# Try it: pip install patha-memory"
Sleep 4000ms

Sleep 1500ms
