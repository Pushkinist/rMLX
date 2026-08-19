#!/usr/bin/env bash
# scripts/check_kernel_dtype_contract.sh — CI gate: a custom-Metal-kernel
# dispatcher that declares an f32 kernel output must not hand that f32 back to
# its caller unexamined.
#
# usage: check_kernel_dtype_contract.sh [SRC_DIR] [MIN_FNS]
#
# PROBLEM
# -------
# An MSL dispatcher declares its output dtype explicitly:
#
#     invoke.add_output_shape(&[dst_len], Dtype::F32)?;
#
# f32 is usually the right choice *inside* the kernel — online softmax, FWHT
# butterflies and log-sum-exp merges all want it. The bug is returning that
# array as-is to a caller that handed in bf16. MLX then does exactly what it is
# designed to do: a binary op between the f32 result and a bf16 tensor promotes
# to f32, and the promotion propagates through the residual add into the next
# layer's norm, its weight GEMV, its elementwise ops and the sampler. The
# activation stream and the quantized weights' scales all re-instantiate at
# twice the width. Nothing errors and nothing warns; the symptoms are
# throughput and non-bit-exactness, which read as "kernel imprecision".
#
# The same shape applies to quantization *parameters*: `mx.quantized_matmul`
# and `mx.dequantize` take their operand width from the scales/biases they are
# given, so a fused quantize kernel that returns f32 scales where `mx.quantize`
# would have returned bf16 ones promotes the decode graph just as effectively —
# without any array being obviously "the output".
#
# WHAT THIS GATE SCANS
# --------------------
# Recursively under every scan root — by default the three crates that own
# `.metal` kernels (`scripts/metal_dirs.sh`: rmlx-kv-quant, rmlx-models,
# rmlx-mlx), or a single root passed as `$1` — every non-test `.rs` file that
# constructs a `MetalKernelInvoke`, i.e. every custom Metal kernel dispatcher.
# Keyed on shape and never on a codec or an architecture name, so a rename or a
# move keeps the file in scope. Scanning all three roots is deliberate: the
# `scalar_f32` gate this one sits beside is pinned to one crate, and that is
# half of why it could not see the defect that prompted this file.
#
# Inside those files, every function that declares at least one
# `add_output_shape(..., Dtype::F32)` must either
#
#   * cast to a *derived* dtype somewhere in its body — `.astype(x.dtype(), …)`,
#     `.astype(out_dtype, …)`, `.astype(in_dtype, …)`. A cast to a literal
#     (`.astype(Dtype::F32, …)`, `.astype(Dtype::Bf16, …)`) does NOT count: the
#     first is the upcast that creates the problem, and the second pins a width
#     the caller never asked for, or
#
#   * carry a `// f32-out-ok: <reason>` marker in the contiguous comment block
#     directly above its signature (or on the signature line). The reason must
#     say who consumes the f32 and why that consumer cannot promote — typically
#     "read by an MSL kernel that declares the buffer type", which is true of
#     our own kernels and false of every MLX op.
#
# WHAT IT CANNOT SEE
# ------------------
# It is a source-shape gate: it cannot tell which of a tuple's elements is the
# f32 one, and it cannot follow the array to its consumer. The runtime
# counterpart is `crates/rmlx-kv-quant/tests/kv_decode_dtype_contract.rs`,
# which sweeps every `ALL_KV_QUANTS` codec through a real decode step under
# both policy arms and asserts the attention output comes back in the query's
# dtype. That test catches the promotion wherever it happens; this gate catches
# the source pattern before a GPU is involved. Neither subsumes the other.
#
# Exit 0 = clean. Exit 1 = violation, or fewer dispatcher functions found than
# expected.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ $# -ge 1 ]; then
    SRC_DIRS=("$1")
else
    SRC_DIRS=(
        "$ROOT/crates/rmlx-kv-quant/src"
        "$ROOT/crates/rmlx-models/src"
        "$ROOT/crates/rmlx-mlx/src"
    )
fi

# Vacuity floor: the number of f32-declaring dispatcher functions the scan is
# expected to find. A refactor that hides them behind a helper the scan cannot
# see must fail here rather than silently shrink coverage to nothing.
#
# Lower it only against dispatchers that are *gone from the tree*, and say
# which ones here. An unexplained decrement is indistinguishable from the
# broken scan this floor exists to catch.
MIN_FNS="${2:-33}"

for d in "${SRC_DIRS[@]}"; do
    if [ ! -d "$d" ]; then
        echo "check-kernel-dtype-contract: scan root does not exist: $d" >&2
        exit 1
    fi
done

FILES=()
while IFS= read -r f; do
    FILES+=("$f")
done < <(
    find "${SRC_DIRS[@]}" -type f -name '*.rs' \
        ! -name '*_tests.rs' ! -name 'tests.rs' \
        -exec grep -lq 'MetalKernelInvoke' {} \; \
        -print | sort
)

TOTAL_FNS=0
VIOLATIONS=0
# `${FILES[@]+...}` keeps `set -u` from aborting on an empty scan root — an
# empty root must reach the vacuity floor below and fail there, with a reason.
for f in ${FILES[@]+"${FILES[@]}"}; do
    out=$(awk '
        function is_literal_astype(line,   rest) {
            # `.astype(Dtype::Bf16` / `.astype(F32` / `.astype(rmlx_mlx::Dtype::F32`
            # — a fixed width, not the one the caller passed. Not a guard.
            rest = line
            sub(/^.*\.astype\(/, "", rest)
            return (rest ~ /^(rmlx_mlx::)?Dtype::/ || rest ~ /^(F32|Bf16|F16)[,)]/)
        }

        function is_derived_astype(line) {
            if (index(line, ".astype(") == 0) return 0
            return !is_literal_astype(line)
        }

        # Is this line a function signature?
        function is_fn(s) {
            return (s ~ /^[[:space:]]*(pub([[:space:]]*\([^)]*\))?[[:space:]]+)?(async[[:space:]]+)?fn[[:space:]]+/)
        }

        # Does the doc/attribute block directly above line `i` carry the
        # marker? Walk back over comments and attributes, stopping at the
        # previous item — the closing brace of the function before it, or its
        # signature. Walking back is what keeps a multi-line `#[allow(...)]`
        # between the comment and the signature from breaking the lookup.
        function has_marker_above(i,   j) {
            for (j = i - 1; j >= 1; j--) {
                if (line[j] ~ /^[[:space:]]*\}/) return 0
                if (is_fn(line[j])) return 0
                if (index(line[j], "f32-out-ok:") > 0) return 1
            }
            return 0
        }

        { line[NR] = $0 }

        END {
            bad = 0
            total = 0
            for (i = 1; i <= NR; i++) {
                if (!is_fn(line[i])) continue
                # Body runs to the line before the next signature (or EOF).
                end = NR
                for (k = i + 1; k <= NR; k++) {
                    if (is_fn(line[k])) { end = k - 1; break }
                }
                f32_out = 0
                guard   = 0
                for (k = i; k <= end; k++) {
                    if (index(line[k], "add_output_shape(") > 0 && index(line[k], "Dtype::F32") > 0) f32_out = 1
                    if (is_derived_astype(line[k])) guard = 1
                }
                if (!f32_out) continue
                total++
                if (guard) continue
                if (index(line[i], "f32-out-ok:") > 0 || has_marker_above(i)) continue
                sig = line[i]
                sub(/^[[:space:]]*/, "", sig)
                printf "%s:%d: %s\n", FILENAME, i, sig
                bad = 1
            }
            printf "TOTAL %d\n", total > "/dev/stderr"
            exit bad ? 1 : 0
        }
    ' "$f" 2>>"${TMPDIR:-/tmp}/kernel_dtype_totals.$$") || VIOLATIONS=1
    if [ -n "$out" ]; then
        printf '%s\n' "$out"
    fi
done

if [ -f "${TMPDIR:-/tmp}/kernel_dtype_totals.$$" ]; then
    TOTAL_FNS=$(awk '{ s += $2 } END { print s + 0 }' "${TMPDIR:-/tmp}/kernel_dtype_totals.$$")
    rm -f "${TMPDIR:-/tmp}/kernel_dtype_totals.$$"
fi

if [ "$VIOLATIONS" -ne 0 ]; then
    cat >&2 <<'EOF'

check-kernel-dtype-contract: FAIL

The dispatcher(s) above declare an f32 kernel output and return it without
restoring a dtype the caller supplied. An f32 array leaving a kernel promotes
every bf16 op it then meets — the residual stream, the next layer's norm and
weight GEMV, the softmax, the sampler — and MLX reports nothing.

Restore the caller's dtype on the way out:

    let dst = dst_flat.reshape(&[b, n_q_heads, 1, head_dim], device)?;
    dst.astype(queries.dtype(), device)

Keep the f32 accumulation inside the kernel; only the returned array changes.

If the f32 is genuinely safe — read back by another MSL kernel that declares
its own buffer type, never handed to an MLX op — say so above the signature:

    // f32-out-ok: <who consumes it, and why that consumer cannot promote>

EOF
    exit 1
fi

if [ "$TOTAL_FNS" -lt "$MIN_FNS" ]; then
    echo "check-kernel-dtype-contract: found $TOTAL_FNS f32-declaring dispatcher functions under" >&2
    printf '  %s\n' "${SRC_DIRS[@]}" >&2
    echo "but expected at least $MIN_FNS." >&2
    echo "A dispatcher was renamed, moved or hidden behind a helper and the scan lost" >&2
    echo "coverage. Fix the scan (or lower the pinned floor deliberately)." >&2
    exit 1
fi

echo "check-kernel-dtype-contract: ok (${#FILES[@]} dispatcher files, $TOTAL_FNS f32-declaring functions)"
