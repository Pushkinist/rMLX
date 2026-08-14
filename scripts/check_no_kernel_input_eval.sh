#!/usr/bin/env bash
# scripts/check_no_kernel_input_eval.sh — CI gate: flag blocking Array::eval()
# on a flash-decode kernel's inputs.
#
# PROBLEM
# -------
# `Array::eval()` is a synchronous graph evaluation: it blocks the calling
# thread until the GPU has produced the array. A flash-decode dispatcher runs
# once per attention layer per decode step, so an `eval()` on its inputs
# serialises the host against the GPU that many times per token — the forward
# pass then advances one layer at a time with nothing queued ahead, and the
# codec measures far slower than the bf16 path it was written to beat. This is
# invisible at review: the call looks like a correctness precaution and the
# output is bit-identical either way.
#
# WHY THE PRECAUTION IS NOT NEEDED
# --------------------------------
# These kernels read their buffers by raw linear offset, so they need
# row-contiguous inputs. That guarantee already comes from the custom-kernel
# builder: `MetalKernel::new` passes `ensure_row_contiguous`, and MLX copies any
# non-row-contiguous input before the dispatch. A blocking `eval()` adds nothing
# to it.
#
# WHAT THIS GATE CHECKS
# ---------------------
# Every non-test `*flash_decode*_msl.rs` under crates/rmlx-kv-quant/src/ is
# scanned for `.eval()` calls. A hit is a violation unless it carries an
# `// eval-ok: <reason>` marker, either on the same line or in the contiguous
# block of comments / `eval()` calls / block delimiters directly above it.
#
# Keyed on shape (a flash-decode dispatcher forcing evaluation), never on a
# codec or architecture name.
#
# Exit 0 = clean. Exit 1 = violation found.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$ROOT/crates/rmlx-kv-quant/src"

shopt -s nullglob
FILES=()
for f in "$SRC_DIR"/*flash_decode*_msl.rs; do
    case "$f" in
        *_tests.rs) continue ;;
    esac
    FILES+=("$f")
done
shopt -u nullglob

if [ ${#FILES[@]} -eq 0 ]; then
    echo "check-no-kernel-input-eval: no flash-decode dispatcher found under $SRC_DIR" >&2
    echo "the gate would pass vacuously — fix the scan path" >&2
    exit 1
fi

VIOLATIONS=0
for f in "${FILES[@]}"; do
    out=$(awk '
        # Track the contiguous run of lines that may carry the marker for the
        # eval below: comments, other eval() calls, and bare block delimiters.
        {
            line = $0
            stripped = line
            gsub(/^[ \t]+|[ \t]+$/, "", stripped)
        }
        stripped ~ /eval-ok:/ { marked = 1 }
        {
            is_comment = (stripped ~ /^\/\//)
            is_eval     = (index(line, ".eval()") > 0)
            is_delim    = (stripped == "}" || stripped ~ /^(if|for)[ (].*\{$/ || stripped == "")
            has_eval    = is_eval && !is_comment
        }
        has_eval && !marked { printf "%s:%d: %s\n", FILENAME, FNR, stripped; bad = 1 }
        # A line that is none of the above ends the marker run.
        !(is_comment || is_eval || is_delim) { marked = 0 }
        END { exit bad ? 1 : 0 }
    ' "$f") || VIOLATIONS=1
    if [ -n "$out" ]; then
        printf '%s\n' "$out"
    fi
done

if [ "$VIOLATIONS" -ne 0 ]; then
    cat >&2 <<'EOF'

check-no-kernel-input-eval: FAIL

A flash-decode dispatcher force-evaluates an array. `Array::eval()` blocks the
calling thread on the GPU; called once per layer per decode step it serialises
the whole forward pass and costs multiples of the decode rate, with no change
to the produced tokens.

The row-contiguous guarantee these raw-linear kernels need already comes from
`MetalKernel::new`, which passes `ensure_row_contiguous` — drop the eval and
leave the graph lazy.

If a specific eval really is load-bearing, say why at the call site:

    // eval-ok: <reason this dispatch cannot be lazy>
    some_array.eval()?;

EOF
    exit 1
fi

echo "check-no-kernel-input-eval: ok (${#FILES[@]} flash-decode dispatchers scanned)"
