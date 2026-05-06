# Patha demo 3: Your AI memory belongs to you
#
# Shows the local-first plain-text architecture, with `patha shell`
# for the conversational ingest part.
# Reproduce: vhs assets/demo-portability.tape

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
Type "clear"
Enter
Sleep 200ms
Show

# ─── Title ──────────────────────────────────────────────────────────
Type "# Where does your AI memory actually live?"
Sleep 2000ms
Enter
Type "# In a SaaS database? On your computer? You should know."
Sleep 2800ms
Enter
Enter

# ─── Tell Patha some things ─────────────────────────────────────────
Type "uv run --quiet patha shell"
Enter
Sleep 2500ms

Type "I am vegetarian"
Sleep 600ms
Enter
Sleep 1800ms

Type "I live in Lisbon"
Sleep 600ms
Enter
Sleep 1800ms

Type "I am learning Portuguese"
Sleep 600ms
Enter
Sleep 2200ms

Type "exit"
Enter
Sleep 1500ms

# ─── The reveal: it's just a file ──────────────────────────────────
Type "# Where did Patha put them?"
Enter
Sleep 800ms

Type "ls /tmp/patha-demo3/"
Enter
Sleep 2200ms

Type "# A plain text file. Open it with any editor:"
Enter
Sleep 700ms

Type "head -3 /tmp/patha-demo3/beliefs.jsonl"
Enter
Sleep 4000ms

Type "# No cloud. No login. No SaaS account."
Sleep 2200ms
Enter
Type "# You can grep it. Git-commit it. Copy it to another machine."
Sleep 2500ms
Enter
Type "# Same file is read by every AI tool that supports MCP."
Sleep 2200ms
Enter
Type "# Claude Desktop, Cursor, Zed, your terminal: same memory."
Sleep 2500ms
Enter
Enter
Type "# Your AI memory belongs to you, not to a company."
Sleep 3000ms
Enter
Enter
Type "# Try it: pip install patha-memory"
Sleep 4000ms

Sleep 1500ms
