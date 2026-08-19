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
# Recursively under every scan root — by default the crate `src/` directories
# that own `.metal` kernels, derived from `scripts/metal_dirs.sh` (the single
# source; nothing here restates the list), or a single root passed as `$1` —
# every non-test `.rs` file that constructs a `MetalKernelInvoke`, i.e. every
# custom Metal kernel dispatcher. Keyed on shape and never on a codec or an
# architecture name, so a rename or a move keeps the file in scope. Scanning
# every metal-owning crate is deliberate: the `scalar_f32` gate beside this one
# was pinned to a single crate, and that was one of three reasons it could not
# see the defect that prompted this file.
#
# THE RULE
# --------
# Source is read as *statements*, not lines, so a rustfmt-split call is seen
# exactly as a one-line one. For each function:
#
#   escaping   = (# `add_output_shape(..., Dtype::F32)`)
#              - (# of those outputs fed straight back into another kernel via
#                 `add_input(&x)`, which never leave the function)
#   guards     = # of casts to a *derived* dtype (`.astype(x.dtype(), …)`,
#                `.astype(out_dtype, …)`) occurring AFTER the last `.apply(` —
#                a kernel output does not exist before that point, so a cast
#                earlier in the body is a cast on an INPUT and guards nothing.
#
# A function passes when `guards >= escaping`, or when it carries a
# `// f32-out-ok: <reason>` marker in the comment/attribute block directly
# above its signature (the walk-back stops at the first line of anything else,
# so a marker cannot leak across a `const`, a `use` or a type alias).
#
# Counting instead of "at least one cast" is what keeps a two-buffer quantizer
# honest: narrowing `scales` and forgetting `biases` leaves the promotion in
# place, and one cast would otherwise exempt both.
#
# A cast to a *literal* dtype (`.astype(Dtype::F32, …)`, `.astype(Dtype::Bf16, …)`)
# is not a guard: the first is the upcast that creates the problem, the second
# pins a width the caller never asked for.
#
# WHAT IT CANNOT SEE
# ------------------
# It is a source-shape gate: it counts, it does not follow dataflow, so it
# cannot say *which* buffer a given cast landed on. The runtime counterpart is
# `crates/rmlx-kv-quant/tests/kv_decode_dtype_contract.rs`, which sweeps every
# `ALL_KV_QUANTS` codec through a real decode step under both policy arms and
# asserts the attention output comes back in the query's dtype. That test
# catches the promotion wherever it happens; this gate catches the source
# pattern before a GPU is involved. Neither subsumes the other.
#
# Exit 0 = clean. Exit 1 = violation, or fewer dispatcher functions found than
# expected.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ $# -ge 1 ]; then
    SRC_DIRS=("$1")
else
    # Single-sourced from the same list the MSL compile / format gates use.
    # `METAL_DIRS` names each crate's `metal/` directory; the Rust dispatchers
    # sit one level up, so strip that suffix.
    # shellcheck source=scripts/metal_dirs.sh
    . "$(dirname "${BASH_SOURCE[0]}")/metal_dirs.sh"
    SRC_DIRS=()
    for d in "${METAL_DIRS[@]}"; do
        SRC_DIRS+=("${d%/metal}")
    done
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

TOTALS_FILE="$(mktemp "${TMPDIR:-/tmp}/kernel_dtype_totals.XXXXXX")"
trap 'rm -f "$TOTALS_FILE"' EXIT

TOTAL_FNS=0
VIOLATIONS=0
# `${FILES[@]+...}` keeps `set -u` from aborting on an empty scan root — an
# empty root must reach the vacuity floor below and fail there, with a reason.
for f in ${FILES[@]+"${FILES[@]}"}; do
    out=$(awk '
        function trim(s) { gsub(/^[[:space:]]+|[[:space:]]+$/, "", s); return s }

        function is_fn(s) {
            return (s ~ /^[[:space:]]*(pub([[:space:]]*\([^)]*\))?[[:space:]]+)?(async[[:space:]]+)?fn[[:space:]]+/)
        }

        # A cast to a fixed width is not a guard; a cast to a dtype read off a
        # value is. `rest` is the text following `.astype(`.
        function is_literal_target(rest) {
            rest = trim(rest)
            if (rest ~ /^(rmlx_mlx::)?Dtype::/) return 1
            if (rest ~ /^(F32|Bf16|F16)[,)]/) return 1
            return 0
        }

        # Count derived-dtype casts in one statement.
        function derived_astype_count(s,   n, p, rest) {
            n = 0
            while ((p = index(s, ".astype(")) > 0) {
                rest = substr(s, p + 8)
                if (!is_literal_target(rest)) n++
                s = rest
            }
            return n
        }

        # Does the doc/attribute block directly above line `i` carry the
        # marker? The walk stops at the first line that is neither a comment
        # nor part of an attribute nor blank — so a marker written above a
        # `const`, a `use` or another item cannot leak onto this function.
        function has_marker_above(i,   j, t, in_attr) {
            in_attr = 0
            for (j = i - 1; j >= 1; j--) {
                t = trim(line[j])
                if (in_attr) {                      # inside a multi-line #[...]
                    if (t ~ /^#\[/) in_attr = 0
                    continue
                }
                if (t == "") continue
                if (t ~ /^\/\//) {
                    if (index(t, "f32-out-ok:") > 0) return 1
                    continue
                }
                if (t ~ /^#\[/) continue            # single-line attribute
                if (t ~ /\)\]$/ || t == "]") { in_attr = 1; continue }
                return 0                            # any other item: stop
            }
            return 0
        }

        { line[NR] = $0 }

        END {
            bad = 0
            total = 0
            for (i = 1; i <= NR; i++) {
                if (!is_fn(line[i])) continue
                end = NR
                for (k = i + 1; k <= NR; k++) {
                    if (is_fn(line[k])) { end = k - 1; break }
                }

                # ---- read the body as statements, not lines ----------------
                ns = 0
                cur = ""
                for (k = i; k <= end; k++) {
                    t = trim(line[k])
                    if (t == "") continue
                    cur = (cur == "") ? t : cur " " t
                    if (t ~ /[;{}\]]$/) { ns++; st[ns] = cur; cur = "" }
                }
                if (cur != "") { ns++; st[ns] = cur }

                f32_out = 0; last_apply = 0; removed = ""
                for (k = 1; k <= ns; k++) {
                    if (index(st[k], "add_output_shape(") > 0 && index(st[k], "Dtype::F32") > 0) f32_out++
                    if (index(st[k], ".apply(") > 0) last_apply = k
                    if (match(st[k], /^let[[:space:]]+(mut[[:space:]]+)?[A-Za-z_][A-Za-z0-9_]*[[:space:]]*=/) &&
                        st[k] ~ /\.remove\(/) {
                        v = st[k]
                        sub(/^let[[:space:]]+(mut[[:space:]]+)?/, "", v)
                        sub(/[[:space:]]*=.*$/, "", v)
                        removed = removed " " v " "
                    }
                }

                internal = 0
                guards = 0
                for (k = 1; k <= ns; k++) {
                    if (index(st[k], "add_input(") > 0 && removed != "") {
                        a = st[k]
                        sub(/^.*add_input\(&?/, "", a)
                        sub(/[^A-Za-z0-9_].*$/, "", a)
                        if (a != "" && index(removed, " " a " ") > 0) internal++
                    }
                    if (k > last_apply) guards += derived_astype_count(st[k])
                }

                if (f32_out == 0) continue
                total++
                escaping = f32_out - internal
                if (escaping < 0) escaping = 0
                if (escaping == 0) continue
                if (guards >= escaping) continue
                if (index(line[i], "f32-out-ok:") > 0 || has_marker_above(i)) continue
                sig = trim(line[i])
                printf "%s:%d: %s  [%d f32 output(s) escape, %d derived cast(s) after dispatch]\n", \
                    FILENAME, i, sig, escaping, guards
                bad = 1
            }
            printf "TOTAL %d\n", total > "/dev/stderr"
            exit bad ? 1 : 0
        }
    ' "$f" 2>>"$TOTALS_FILE") || VIOLATIONS=1
    if [ -n "$out" ]; then
        printf '%s\n' "$out"
    fi
done

TOTAL_FNS=$(awk '{ s += $2 } END { print s + 0 }' "$TOTALS_FILE")

if [ "$VIOLATIONS" -ne 0 ]; then
    cat >&2 <<'EOF'

check-kernel-dtype-contract: FAIL

The dispatcher(s) above declare more f32 kernel outputs that escape the
function than they cast back to a caller-supplied dtype. An f32 array leaving a
kernel promotes every bf16 op it then meets — the residual stream, the next
layer's norm and weight GEMV, the softmax, the sampler — and MLX reports
nothing.

Restore the caller's dtype on the way out, once per escaping buffer:

    let dst = dst_flat.reshape(&[b, n_q_heads, 1, head_dim], device)?;
    dst.astype(queries.dtype(), device)

Keep the f32 accumulation inside the kernel; only the returned arrays change.
A cast BEFORE the dispatch lands on an input and is not counted.

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
