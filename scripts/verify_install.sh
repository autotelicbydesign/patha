#!/usr/bin/env bash
# Pre-flight verification for a Patha wheel.
#
# Creates a throwaway Python 3.11 venv, installs a wheel into it, and
# runs a representative smoke test (Memory.remember / Memory.recall +
# the synthesis-intent path with the regex false-positive filters).
# Exits non-zero on any failure so this is safe to wire into CI or
# a release Makefile target.
#
# Usage:
#   scripts/verify_install.sh dist/patha_memory-0.10.8-py3-none-any.whl
#   scripts/verify_install.sh patha-memory==0.10.8                       # from PyPI
#   scripts/verify_install.sh -i https://test.pypi.org/simple/ patha-memory==0.10.8
#
# Requirements: python3.11 on PATH (or settable via PYTHON env var).

set -euo pipefail

PYTHON="${PYTHON:-python3.11}"
TMPDIR="$(mktemp -d -t patha-verify-XXXXXXXX)"
PATHA_PATH="$TMPDIR/beliefs.jsonl"

cleanup() {
  deactivate 2>/dev/null || true
  rm -rf "$TMPDIR"
}
trap cleanup EXIT

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <wheel-or-pip-spec> [extra pip args...]" >&2
  exit 2
fi

echo "== Patha install verification =="
echo "Python:    $($PYTHON --version)"
echo "Tmp dir:   $TMPDIR"
echo "Spec:      $*"
echo

echo "== Creating venv =="
"$PYTHON" -m venv "$TMPDIR/venv"
# shellcheck disable=SC1091
source "$TMPDIR/venv/bin/activate"
pip install --quiet --upgrade pip

echo "== Installing =="
pip install --quiet "$@"

echo "== Smoke test =="
PATHA_PATH="$PATHA_PATH" python <<'PYEOF'
import os
from patha import Memory, __version__

print(f"version:           {__version__}")

path = os.environ["PATHA_PATH"]

# 1. Basic remember + recall
m = Memory(path=path, enable_phase1=False)
m.remember("I love sushi every week")
m.remember("I am avoiding raw fish on doctor advice")
rec = m.recall("what do I eat?")
assert rec.summary, "recall.summary should be non-empty"
print(f"basic recall:      strategy={rec.strategy} current={len(rec.current)} tokens={rec.tokens}")

# 2. Synthesis-intent path with the regex false-positive filters
m2 = Memory(path=path + ".2", enable_phase1=False)
m2.remember("Bike racks range from $100 to $500.")          # range filter
m2.remember("I bought a $50 saddle for my bike.")           # real
m2.remember("I was thinking about a $300 helmet.")          # hypothetical
m2.remember("I didn't buy the $400 frame in the end.")      # negated
rec2 = m2.recall("how much have I spent on bike-related expenses?")
assert rec2.ganita is not None, "synthesis path should produce a ganita result"
assert rec2.strategy == "ganita", f"expected ganita strategy, got {rec2.strategy}"
assert rec2.ganita.value == 50.0, (
    f"expected $50 (filters drop range/hypothetical/negated), got {rec2.ganita.value}"
)
print(f"synthesis path:    strategy={rec2.strategy} answer={rec2.answer} value={rec2.ganita.value}")

print("\nAll checks passed.")
PYEOF

echo
echo "== verify_install.sh: success =="
