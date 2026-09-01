#!/usr/bin/env bash
# Cross-venv ABBA driver for turbo_probe.py.
#
# The stock `mlx-lm` and the `mlx-lm-turboquant` fork live in separate venvs, so
# a fork-vs-stock comparison cannot run inside one process. This alternates them
# in an ABBA schedule at the process level -- stock, fork, fork, stock -- so a
# monotonic drift in host state (thermals, background load) cancels between the
# two arms instead of landing entirely on whichever ran first.
#
# Within one arm, use turbo_probe.py's own --seq palindrome instead: that gives
# a single-process ABBA with one model load and is strictly the better
# instrument. Use this script only for the cross-venv leg.
#
# Usage:
#   turbo_abba.sh <model_dir> <prompt_tokens> <gen> <reps_per_process> <seq> <out.jsonl>
#
# Exit codes:
#   0  both arms ran
#   2  an interpreter is missing, or an arm's mlx_lm tree could not be read
#   6  the two arms resolve the SAME mlx_lm source tree -- see the arm-identity
#      block below. Distinct from 2 because "the comparison is not a comparison"
#      and "the environment is not set up" want different fixes.
#
# <seq> must be runnable under BOTH venvs -- i.e. fp16 / mlxq8 / mlxq4 only.
# Turbo modes exist in the fork alone and belong in a fork-only single-process
# run.

set -euo pipefail

MODEL="${1:?model_dir required}"
PROMPT_TOKENS="${2:?prompt_tokens required}"
GEN="${3:?gen required}"
REPS="${4:?reps required}"
SEQ="${5:?seq required}"
OUT="${6:?out jsonl required}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROBE="$ROOT/scripts/baseline/turbo_probe.py"

STOCK_PY="${MLX_LM_ROOT:-$ROOT/../mlx-lm}/.venv/bin/python"
FORK_PY="${MLX_LM_TURBOQUANT_ROOT:-$ROOT/../mlx-lm-turboquant}/.venv/bin/python"

for py in "$STOCK_PY" "$FORK_PY"; do
    [ -x "$py" ] || { echo "missing interpreter: $py" >&2; exit 2; }
done

# The two arms have to be two different trees, and this is where that is
# decided rather than assumed. A venv that resolves mlx_lm from site-packages
# instead of its checkout makes both arms the same code, and the failure is
# silent in the worst way: all four runs succeed and the summarizer prints a
# fork-vs-stock ratio of 1.000x, which reads as a measured null.
#
# Emits `<sha256>\t<tree>\t<mlx_version>` for one interpreter.
arm_identity() {  # arm_identity <python>
    "$1" - <<'PYEOF'
import hashlib, os
import mlx.core as mx
import mlx_lm
d = os.path.dirname(mlx_lm.__file__)
h = hashlib.sha256()
for root, dirs, files in os.walk(d):
    dirs[:] = sorted(x for x in dirs if x != "__pycache__")
    for f in sorted(files):
        if f.endswith(".py"):
            h.update(f.encode())
            with open(os.path.join(root, f), "rb") as fh:
                h.update(fh.read())
print(f"{h.hexdigest()}\t{d}\t{mx.__version__}")
PYEOF
}

echo "--- arm identity ---"
STOCK_ID="$(arm_identity "$STOCK_PY")" ||
    { echo "could not read the stock arm's mlx_lm tree" >&2; exit 2; }
FORK_ID="$(arm_identity "$FORK_PY")" ||
    { echo "could not read the fork arm's mlx_lm tree" >&2; exit 2; }

STOCK_SHA="${STOCK_ID%%$'\t'*}"
FORK_SHA="${FORK_ID%%$'\t'*}"
STOCK_TREE="$(printf '%s' "$STOCK_ID" | cut -f2)"
FORK_TREE="$(printf '%s' "$FORK_ID" | cut -f2)"
STOCK_MLX="$(printf '%s' "$STOCK_ID" | cut -f3)"
FORK_MLX="$(printf '%s' "$FORK_ID" | cut -f3)"

printf 'stock  mlx=%s tree=%s sha256=%s\n' "$STOCK_MLX" "$STOCK_TREE" "${STOCK_SHA:0:16}"
printf 'fork   mlx=%s tree=%s sha256=%s\n' "$FORK_MLX" "$FORK_TREE" "${FORK_SHA:0:16}"

if [ -z "$STOCK_SHA" ] || [ -z "$FORK_SHA" ]; then
    echo "REFUSING: an arm reported no source digest" >&2
    exit 2
fi
if [ "$STOCK_SHA" = "$FORK_SHA" ]; then
    cat >&2 <<MSG
REFUSING: both arms resolve the same mlx_lm source tree.
  stock $STOCK_TREE
  fork  $FORK_TREE
  sha256 $STOCK_SHA
Every ratio this run could produce would be a tree compared against itself, and
it would come back at 1.000x looking like a measured null. Point
MLX_LM_TURBOQUANT_ROOT at the fork checkout, or fix a venv that is resolving
mlx_lm from site-packages instead of its own tree.
MSG
    exit 6
fi
echo "--------------------"

# The artifact carries the proof, not just the terminal: a jsonl read later has
# to be able to answer "were these two arms actually different code?".
printf '{"record":"arm_identity","stock":{"sha256":"%s","tree":"%s","mlx":"%s"},"fork":{"sha256":"%s","tree":"%s","mlx":"%s"}}\n' \
    "$STOCK_SHA" "$STOCK_TREE" "$STOCK_MLX" \
    "$FORK_SHA" "$FORK_TREE" "$FORK_MLX" >> "$OUT"

run() {  # run <arm_label> <python>
    "$2" "$PROBE" --model "$MODEL" --prompt-tokens "$PROMPT_TOKENS" \
        --seq "$SEQ" --reps "$REPS" --gen "$GEN" --arm "$1" --out "$OUT"
}

# ABBA at the process level.
run stock "$STOCK_PY"
run fork  "$FORK_PY"
run fork  "$FORK_PY"
run stock "$STOCK_PY"

echo "wrote $OUT"
