#!/usr/bin/env bash
# scripts/check_no_kernel_input_eval.sh — CI gate: flag blocking Array::eval()
# in a custom-Metal-kernel dispatcher.
#
# usage: check_no_kernel_input_eval.sh [SRC_DIR] [MIN_FILES]
#
# PROBLEM
# -------
# `Array::eval()` is a synchronous graph evaluation: it blocks the calling
# thread until the GPU has produced the array. A decode-path dispatcher runs
# once per attention layer per decode step, so an `eval()` on its inputs
# serialises the host against the GPU that many times per token — the forward
# pass then advances one layer at a time with nothing queued ahead, and the
# codec measures far slower than the bf16 path it was written to beat. This is
# invisible at review: the call looks like a correctness precaution and the
# output is bit-identical either way.
#
# WHY THE PRECAUTION IS NOT NEEDED
# --------------------------------
# `MetalKernel::apply` enqueues an MLX `fast::CustomKernel` graph node; it does
# not dispatch. MLX runs that node's `eval_gpu` only once every input edge is
# materialised, and the row-contiguous copy (`ensure_row_contiguous`, passed by
# `MetalKernel::new`) happens inside that same `eval_gpu`. A kernel therefore
# cannot read an uncomputed or strided buffer, and a caller-side `eval()` adds
# no ordering the graph does not already give. The long-form version of this
# argument lives in one place — the `flash_decode_common` module docs.
#
# WHAT THIS GATE SCANS
# --------------------
# Recursively under SRC_DIR (default `crates/rmlx-kv-quant/src`), every
# non-test `.rs` file that is either
#
#   * a custom-Metal-kernel dispatcher — it constructs a `MetalKernelInvoke`, or
#   * a shared dispatcher scaffold — its name ends in `_common.rs`.
#
# Keyed on shape, never on a codec or architecture name, so a dispatcher that
# is renamed, or moved into a sub-directory, stays in scope. The `_common.rs`
# arm is what keeps an eval from being hidden one call deep in a helper that
# every dispatcher already imports.
#
# A `.eval(` / `::eval(` call is a violation unless the *same* call carries an
# `// eval-ok: <reason>` marker — on its own line, or on the contiguous comment
# lines directly above it. One marker exempts exactly one call.
#
# Exit 0 = clean. Exit 1 = violation, or fewer dispatchers found than expected.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="${1:-$ROOT/crates/rmlx-kv-quant/src}"

# Vacuity floor. A rename or a relocation that drops a dispatcher out of the
# scan must fail the gate, not silently shrink its coverage — so the expected
# file count is pinned, not merely required to be non-zero. Raise it when a
# dispatcher is added; a drop below it means coverage was lost.
#
# Lower it only against a dispatcher that is *gone from the tree*, never to
# quiet a failure, and record which one here. An unexplained decrement is
# indistinguishable from the broken glob this floor exists to catch, so the
# reason is the load-bearing part of the change and the number is not.
#
# 27 -> 26: the sparse-V weighted-sum dispatcher was deleted, not relocated —
# its kernel dequantized affine V data with a symmetric formula and was removed
# outright rather than repaired. No other file entered or left the scanned set.
MIN_FILES="${2:-${CHECK_EVAL_MIN_FILES:-26}}"

if [ ! -d "$SRC_DIR" ]; then
    echo "check-no-kernel-input-eval: scan root does not exist: $SRC_DIR" >&2
    exit 1
fi

FILES=()
while IFS= read -r f; do
    FILES+=("$f")
done < <(
    find "$SRC_DIR" -type f -name '*.rs' \
        ! -name '*_tests.rs' ! -name 'tests.rs' \
        \( -name '*_common.rs' -o -exec grep -lq 'MetalKernelInvoke' {} \; \) \
        -print | sort
)

if [ "${#FILES[@]}" -lt "$MIN_FILES" ]; then
    echo "check-no-kernel-input-eval: found ${#FILES[@]} dispatcher/scaffold files under" >&2
    echo "  $SRC_DIR" >&2
    echo "but expected at least $MIN_FILES." >&2
    echo "A dispatcher was renamed, moved or deleted and the scan lost coverage." >&2
    echo "Fix the scan (or lower the pinned floor deliberately) — do not ignore this." >&2
    exit 1
fi

VIOLATIONS=0
for f in "${FILES[@]}"; do
    out=$(awk '
        {
            line = $0
            stripped = line
            gsub(/^[ \t]+|[ \t]+$/, "", stripped)

            # Strip a trailing line comment before looking for a call, so a
            # comment that merely mentions eval() is not a violation.
            code = line
            sub(/\/\/.*$/, "", code)

            is_full_comment = (stripped ~ /^\/\//)
            has_marker      = (stripped ~ /\/\/.*eval-ok:/)
            is_eval         = (code ~ /(\.|::)eval[[:space:]]*\(/)
        }

        # Same-line marker: exempts this call and nothing else.
        has_marker && is_eval { armed = 0; next }

        # Comment-line marker: arms exactly one following call.
        has_marker { armed = 1; next }

        is_eval {
            if (armed) { armed = 0; next }
            printf "%s:%d: %s\n", FILENAME, FNR, stripped
            bad = 1
            next
        }

        # Further comment lines keep the marker armed; anything else disarms it,
        # so a marker cannot leak across an intervening statement or block.
        is_full_comment { next }
        { armed = 0 }

        END { exit bad ? 1 : 0 }
    ' "$f") || VIOLATIONS=1
    if [ -n "$out" ]; then
        printf '%s\n' "$out"
    fi
done

if [ "$VIOLATIONS" -ne 0 ]; then
    cat >&2 <<'EOF'

check-no-kernel-input-eval: FAIL

A custom-Metal-kernel dispatcher force-evaluates an array. `Array::eval()`
blocks the calling thread on the GPU; called once per layer per decode step it
serialises the whole forward pass and costs multiples of the decode rate, with
no change to the produced tokens.

`MetalKernel::apply` enqueues a graph node, not a dispatch: MLX materialises
every input — and applies the `ensure_row_contiguous` copy — inside the
kernel's own `eval_gpu`. Drop the eval and leave the graph lazy.

If a specific eval really is load-bearing (a host readback, or an ordering
constraint the graph genuinely cannot express), say why at the call site:

    // eval-ok: <reason this one call cannot be lazy>
    some_array.eval()?;

One marker exempts one call.

EOF
    exit 1
fi

echo "check-no-kernel-input-eval: ok (${#FILES[@]} dispatcher/scaffold files scanned)"
