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

# Verify the two arms really are two different trees before trusting any ratio
# between them. A venv that silently resolves mlx_lm from site-packages instead
# of the checkout would make both arms the same code.
echo "--- arm identity ---"
for pair in "stock:$STOCK_PY" "fork:$FORK_PY"; do
    name="${pair%%:*}"; py="${pair#*:}"
    "$py" - "$name" <<'PYEOF'
import hashlib, os, sys
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
print(f"{sys.argv[1]:<6} mlx={mx.__version__} tree={d} sha256={h.hexdigest()[:16]}")
PYEOF
done
echo "--------------------"

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
