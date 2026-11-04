#!/usr/bin/env bash
# CI gate: Metal-kernel dispatchers keep inputs lazy, return the caller's dtype, leak no f32 scalar.
#
# One population, three rules: the Rust that dispatches the custom Metal kernels of the crates
# that own `.metal` files. Every mode takes the crates from `scripts/lib/metal_dirs.sh` (the one
# list; nothing here restates it): each directory listed there names a crate's `metal/`
# directory, and the Rust sits one level up. `--root DIR` scans exactly the DIRs given, in every
# mode, and may be repeated.
#
# input-eval: no blocking `Array::eval()` on a kernel input.
#
#   `Array::eval()` blocks the calling thread until the GPU has produced the array. A
#   decode-path dispatcher runs once per attention layer per decode step, so an `eval()` on
#   its inputs serialises the host against the GPU that many times per token and the codec
#   measures far slower than the bf16 path it was written to beat. The output is
#   bit-identical either way, so review cannot see it.
#
#   The precaution is not needed: `MetalKernel::apply` enqueues an MLX `fast::CustomKernel`
#   graph node and does not dispatch. MLX runs that node's `eval_gpu` only once every input
#   edge is materialised, and the row-contiguous copy (`ensure_row_contiguous`, passed by
#   `MetalKernel::new`) happens inside that same `eval_gpu`. The long form of this argument
#   lives in the `flash_decode_common` module docs.
#
#   Scanned: every non-test `.rs` file that constructs a `MetalKernelInvoke` (a dispatcher), and
#   every `*_common.rs` file (a shared scaffold) in a directory that holds a dispatcher. Keyed on
#   shape and never on a codec or architecture name, so a rename or a move keeps a file in scope.
#   The scaffold arm keeps an eval from hiding one call deep in a helper the dispatchers beside it
#   import; a `_common.rs` with no dispatcher beside it is no scaffold of one and is not scanned.
#
#   A `.eval(` or `::eval(` call is a violation unless the same call carries an
#   `// eval-ok: <reason>` marker, on its own line or on the contiguous comment lines
#   directly above it. One marker exempts exactly one call.
#
# dtype-contract: no declared-f32 kernel output handed back without restoring the caller dtype.
#
#   A dispatcher declares its output dtype (`invoke.add_output_shape(&[n], Dtype::F32)?`). f32
#   is right inside the kernel (online softmax, FWHT butterflies, log-sum-exp merges). The bug
#   is returning that array as is to a caller that handed in bf16: a binary op between the f32
#   result and a bf16 tensor promotes to f32, and the promotion carries through the residual
#   add into the next layer's norm, its weight GEMV, its elementwise ops and the sampler.
#   Nothing errors; the symptoms are throughput and non-bit-exactness. Quantization
#   parameters are the same shape: `quantized_matmul` and `dequantize` take their operand
#   width from the scales and biases they are given.
#
#   Scanned: every non-test `.rs` file that constructs a `MetalKernelInvoke`. Source is read
#   as statements, not lines, so a rustfmt-split call reads like a one-line one. Per function:
#     escaping = (# `add_output_shape(..., Dtype::F32)`)
#                - (# of those outputs fed straight back into another kernel via
#                   `add_input(&x)`, which never leave the function)
#     guards   = # of casts to a derived dtype (`.astype(x.dtype(), ..)`, `.astype(out_dtype, ..)`)
#                after the call of the last `.apply(`, in its statement or a later one; a cast
#                before it, or inside its arguments, lands on an input and guards nothing.
#   A function passes when `guards >= escaping`, or when a `// f32-out-ok: <reason>` marker
#   sits in the comment and attribute block directly above its signature (the walk back stops
#   at the first line of anything else, so a marker cannot leak across a `const`, a `use` or a
#   type alias). Counting is what keeps a two-buffer quantizer honest: narrowing `scales` and
#   forgetting `biases` leaves the promotion in place. A cast to a literal dtype
#   (`Dtype::F32`, `Dtype::Bf16`, bare `F32`) is not a guard.
#   It counts and does not follow dataflow. The runtime counterpart is
#   `crates/rmlx-kv-quant/tests/kv_decode_dtype_contract.rs`; neither subsumes the other.
#
# scalar-f32: no unguarded `scalar_f32(` in the crates that own `.metal` kernels.
#
#   `scalar_f32(x)` creates a strong-F32 scalar Array. Combined (multiply, add, divide, ...)
#   with a BF16 activation, MLX promotes the result to F32 and contaminates the residual
#   stream, Q/K/V and the KV cache. The call site looks harmless at review.
#
#   Scanned: every non-test `.rs` file, `laguna/` and `dr_venus/` excluded (out of scope). A
#   line with `scalar_f32(` as code (not a pure comment line) is a violation unless
#     (a) a non-F32 `.astype(` follows on the same line, or on the continuation lines of the
#         same chain (lines that do not end the statement with `;` or `}`); or
#     (b) a `// f32-ok: <reason>` marker is on the same line or in the contiguous comment
#         block directly above.
#   An `.astype(` targeting `Dtype::F32`, `F32` or `rmlx_mlx::Dtype::F32` keeps the f32 and is
#   no guard. The fix is `scalar_f32(x).astype(operand.dtype(), device)?`. The marker is for a
#   scalar that is genuinely f32-only (a tower that runs in f32, an `Vec<f32>` output) and
#   names the specific reason.
#
# Floor. A rename or a move that drops a dispatcher out of the scan must fail the gate, not
# shrink its coverage: input-eval counts scanned files and dtype-contract counts f32-declaring
# functions, and either fails below `--min N` (1 to 9 digits, no leading zero). The default
# floor is the figure the real tree gives for the derived crates, files for input-eval and
# functions for dtype-contract; lower it only against a file or function gone from the tree,
# in the same change. With `--root` and no `--min` the floor is 1. No recall table can see a
# default raised or lowered, because every row names its own `--root`: the real-tree targets
# run with the default and are what catches it. scalar-f32 has no floor (every non-test file
# is in its population, which has no stable count) and fails on an empty scan instead.
#
# Usage:  bash scripts/check/kernel_dispatch.sh <input-eval|dtype-contract|scalar-f32> [--root DIR]... [--min N]
# Exit:   0 clean; 1 a violation, or fewer files or functions than the floor; 2 usage error
#         (no mode, an unknown mode, a bad option, a repeated --min, an invalid --min, --min on
#         scalar-f32), a scan root that is not a directory, nothing to scan (scalar-f32), or an
#         awk scan that failed
# Not a mode of: check/eval_lock.sh — that gate reads every MLX eval FFI call of the workspace
#                for the lock held around it; this one reads only the dispatcher files of the
#                crates that own `.metal` kernels, for three rules that are not about a lock

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/lib/awk_text.sh
. "$REPO_ROOT/scripts/lib/awk_text.sh"

usage() {
    echo "usage: bash scripts/check/kernel_dispatch.sh <input-eval|dtype-contract|scalar-f32> [--root DIR]... [--min N]" >&2
    exit 2
}

MODE="${1:-}"
case "$MODE" in
input-eval | dtype-contract | scalar-f32) shift ;;
*) usage ;;
esac

SCAN_ROOTS=()
MIN=""
while [ $# -gt 0 ]; do
    case "$1" in
    --root)
        [ $# -ge 2 ] && [ -n "$2" ] || usage
        SCAN_ROOTS+=("$2")
        shift 2
        ;;
    --min)
        [ $# -ge 2 ] && [ -z "$MIN" ] || usage
        case "$2" in
        '' | *[!0-9]* | 0* | ??????????*) usage ;;
        esac
        MIN="$2"
        shift 2
        ;;
    *) usage ;;
    esac
done

case "$MODE" in
input-eval)
    LABEL=check-no-kernel-input-eval
    UNIT="dispatcher/scaffold files"
    DEFAULT_MIN=29
    ;;
dtype-contract)
    LABEL=check-kernel-dtype-contract
    UNIT="f32-declaring dispatcher functions"
    DEFAULT_MIN=30
    ;;
scalar-f32)
    LABEL=check-no-scalar-f32-leak
    if [ -n "$MIN" ]; then
        echo "$LABEL: scalar-f32 has no floor; --min is for input-eval and dtype-contract" >&2
        exit 2
    fi
    ;;
esac

if [ "${#SCAN_ROOTS[@]}" -gt 0 ]; then
    SRC_DIRS=(${SCAN_ROOTS[@]+"${SCAN_ROOTS[@]}"})
else
    # shellcheck source=scripts/lib/metal_dirs.sh
    . "$REPO_ROOT/scripts/lib/metal_dirs.sh"
    SRC_DIRS=()
    for d in ${METAL_DIRS[@]+"${METAL_DIRS[@]}"}; do
        SRC_DIRS+=("${d%/metal}")
    done
fi

for d in ${SRC_DIRS[@]+"${SRC_DIRS[@]}"}; do
    if [ ! -e "$d" ]; then
        echo "$LABEL: scan root does not exist: $d" >&2
        exit 2
    fi
    if [ ! -d "$d" ]; then
        echo "$LABEL: scan root is not a directory: $d" >&2
        exit 2
    fi
done

dispatcher_files() {
    find ${SRC_DIRS[@]+"${SRC_DIRS[@]}"} -type f -name '*.rs' \
        ! -name '*_tests.rs' ! -name 'tests.rs' \
        -exec grep -lq 'MetalKernelInvoke' {} \; \
        -print
}

# The dispatchers, and each `*_common.rs` in a directory that holds one.
input_eval_files() {
    local dispatchers dirs nl scaffold
    nl=$'\n'
    dispatchers="$(dispatcher_files)"
    [ -z "$dispatchers" ] || printf '%s\n' "$dispatchers"
    dirs="$nl"
    while IFS= read -r scaffold; do
        [ -z "$scaffold" ] || dirs="$dirs${scaffold%/*}$nl"
    done <<<"$dispatchers"
    find ${SRC_DIRS[@]+"${SRC_DIRS[@]}"} -type f -name '*_common.rs' | while IFS= read -r scaffold; do
        case "$dirs" in
        *"$nl${scaffold%/*}$nl"*) printf '%s\n' "$scaffold" ;;
        esac
    done
}

list_files() {
    case "$MODE" in
    input-eval) input_eval_files ;;
    dtype-contract) dispatcher_files ;;
    scalar-f32)
        find ${SRC_DIRS[@]+"${SRC_DIRS[@]}"} -type f -name '*.rs' \
            -not -path '*/target/*' \
            -not -path '*/tests/*' \
            -not -name '*_tests.rs' \
            -not -name 'tests.rs' \
            -not -path '*/laguna/*' \
            -not -path '*/dr_venus/*' \
            -print
        ;;
    esac | sort -u
}

# Each scan prints "file:line: ..." for a violation and exits 1; exit 0 is clean.
scan_input_eval() {
    awk "$AWK_TEXT_FNS"'
        {
            line = $0
            stripped = line
            gsub(/^[ \t]+|[ \t]+$/, "", stripped)

            code = decomment(line)
            comment = substr(line, length(code) + 1)

            is_full_comment = (stripped ~ /^\/\//)
            has_marker      = (comment ~ /eval-ok:/)
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
    ' "$1"
}

scan_dtype_contract() {
    awk -v totals="$TOTALS_FILE" "$AWK_TEXT_FNS"'
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

        # What follows the call that opens at the last `.apply(` of `s`: the
        # arguments of the dispatch are inputs, the chain after it is the output.
        function after_last_apply(s,   p, q, i, depth, ch) {
            p = 0
            while ((q = index(substr(s, p + 1), ".apply(")) > 0) p += q
            depth = 1
            for (i = p + 7; i <= length(s); i++) {
                ch = substr(s, i, 1)
                if (ch == "(") depth++
                else if (ch == ")" && --depth == 0) return substr(s, i + 1)
            }
            return ""
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
                    t = trim(decomment(line[k]))
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
                    else if (k == last_apply) guards += derived_astype_count(after_last_apply(st[k]))
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
            printf "TOTAL %d\n", total >> totals
            exit bad ? 1 : 0
        }
    ' "$1"
}

scan_scalar_f32() {
    awk -v file="$1" "$AWK_TEXT_FNS"'
        function is_f32_astype(line,    p) {
            # Returns 1 if the line contains .astype( targeting F32 — not a guard.
            p = index(line, ".astype(Dtype::F32")
            if (p > 0) return 1
            p = index(line, ".astype(F32")
            if (p > 0) return 1
            p = index(line, ".astype(rmlx_mlx::Dtype::F32")
            if (p > 0) return 1
            return 0
        }

        function has_good_astype(line) {
            # Returns 1 if .astype( is present AND it is NOT an F32 target.
            if (index(line, ".astype(") == 0) return 0
            return !is_f32_astype(line)
        }

        function is_terminated(line,    s, c) {
            # Returns 1 if the line ends a statement (trailing ; or }).
            s = line
            gsub(/^[[:space:]]+/, "", s)
            gsub(/[[:space:]]+$/, "", s)
            c = substr(s, length(s), 1)
            return (c == ";" || c == "}")
        }

        # State: in_comment_block / block_has_f32ok track the contiguous run of
        # // lines being read; prev_f32ok holds the verdict of that run once the
        # first code line after it is reached; pending is a scalar_f32( line whose
        # statement continues, awaiting a non-F32 .astype( on a continuation line.
        BEGIN {
            in_comment_block = 0
            block_has_f32ok  = 0
            prev_f32ok       = 0
            pending          = 0
            pending_line     = 0
            pending_text     = ""
            found            = 0
        }

        # ---- blank line -------------------------------------------------------
        /^[[:space:]]*$/ {
            if (!pending) {
                in_comment_block = 0
                block_has_f32ok  = 0
                prev_f32ok       = 0
            }
            next
        }

        # ---- pure comment line ------------------------------------------------
        /^[[:space:]]*\/\// {
            in_comment_block = 1
            if (index($0, "// f32-ok:") > 0) block_has_f32ok = 1
            next
        }

        # ---- code line --------------------------------------------------------
        {
            code = decomment($0)
            comment = substr($0, length(code) + 1)

            # Capture preceding comment block f32-ok status on first code line.
            if (in_comment_block) {
                prev_f32ok = block_has_f32ok
            }
            in_comment_block = 0
            block_has_f32ok  = 0

            # Handle pending multi-line chain from a previous scalar_f32( line.
            if (pending) {
                if (has_good_astype(code)) {
                    # Chain resolved with a non-F32 .astype( — OK.
                    pending = 0
                } else if (is_terminated(code)) {
                    # Statement terminated without a non-F32 .astype( → violation.
                    print file ":" pending_line ": unguarded scalar_f32(  →  " pending_text
                    found = 1
                    pending = 0
                }
                # else: non-terminated continuation — keep pending open, scan next line.
                # Fall through to check this line for its own scalar_f32(.
            }

            # Check this code line for scalar_f32(.
            if (index(code, "scalar_f32(") > 0) {
                # (a) Same-line f32-ok allowlist.
                if (index(comment, "// f32-ok:") > 0) { prev_f32ok = 0; next }
                # (b) Preceding comment block had f32-ok.
                if (prev_f32ok) { prev_f32ok = 0; next }
                # (c) Same-line non-F32 .astype( chain.
                if (has_good_astype(code)) { prev_f32ok = 0; next }
                # (d) Same-line .astype( that targets F32 — counts as NO guard.
                # (falls through to the termination check below)

                # (e) Is the statement terminated on this line?
                if (is_terminated(code)) {
                    # Terminated with no guard → violation.
                    print file ":" NR ": unguarded scalar_f32(  →  " $0
                    found = 1
                    prev_f32ok = 0
                    next
                }
                # Non-terminated: enter pending state to scan continuation lines.
                pending      = 1
                pending_line = NR
                pending_text = $0
                prev_f32ok   = 0
                next
            }

            # Not a scalar_f32( line.
            prev_f32ok = (index(comment, "// f32-ok:") > 0)
        }

        END {
            if (pending) {
                print file ":" pending_line ": unguarded scalar_f32(  →  " pending_text
                found = 1
            }
            exit found ? 1 : 0
        }
    ' "$1"
}

scan_file() {
    case "$MODE" in
    input-eval) scan_input_eval "$1" ;;
    dtype-contract) scan_dtype_contract "$1" ;;
    scalar-f32) scan_scalar_f32 "$1" ;;
    esac
}

print_failure() {
    case "$MODE" in
    input-eval)
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
        ;;
    dtype-contract)
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
        ;;
    scalar-f32)
        cat >&2 <<'EOF'

check-no-scalar-f32-leak: FAIL

CANONICAL FIX: scalar_f32(x).astype(operand.dtype(), device)?
  NOTE: .astype(Dtype::F32, ...) does NOT count as a guard.

To allowlist a genuine f32-only scalar, add an inline marker:
  // f32-ok: <reason why f32 is safe here>
  (on the same line as scalar_f32, or in the comment block immediately above)

EOF
        ;;
    esac
}

FILES=()
while IFS= read -r f; do
    FILES+=("$f")
done < <(list_files)

if [ "$MODE" = scalar-f32 ] && [ "${#FILES[@]}" -eq 0 ]; then
    echo "$LABEL: no .rs file to scan under" >&2
    printf '  %s\n' ${SRC_DIRS[@]+"${SRC_DIRS[@]}"} >&2
    exit 2
fi

TOTALS_FILE="$(mktemp "${TMPDIR:-/tmp}/kernel_dispatch_totals.XXXXXX")"
trap 'rm -f "$TOTALS_FILE"' EXIT

VIOLATIONS=0
for f in ${FILES[@]+"${FILES[@]}"}; do
    rc=0
    out="$(scan_file "$f")" || rc=$?
    if [ "$rc" -gt 1 ]; then
        echo "$LABEL: the scan of $f failed (status $rc)" >&2
        exit 2
    fi
    if [ -n "$out" ]; then
        printf '%s\n' "$out"
    fi
    if [ "$rc" -eq 1 ]; then
        VIOLATIONS=1
    fi
done

if [ "$VIOLATIONS" -ne 0 ]; then
    print_failure
    exit 1
fi

if [ "$MODE" = scalar-f32 ]; then
    echo "$LABEL: ok (${#FILES[@]} files scanned)"
    exit 0
fi

FOUND="${#FILES[@]}"
if [ "$MODE" = dtype-contract ]; then
    FOUND="$(awk '{ s += $2 } END { print s + 0 }' "$TOTALS_FILE")"
fi

if [ "${#SCAN_ROOTS[@]}" -gt 0 ]; then
    FLOOR="${MIN:-1}"
else
    FLOOR="${MIN:-$DEFAULT_MIN}"
fi

if [ "$FOUND" -lt "$FLOOR" ]; then
    echo "$LABEL: found $FOUND $UNIT under" >&2
    printf '  %s\n' ${SRC_DIRS[@]+"${SRC_DIRS[@]}"} >&2
    echo "but expected at least $FLOOR." >&2
    echo "A dispatcher was renamed, moved or hidden behind a helper and the scan lost" >&2
    echo "coverage. Fix the scan (or lower the floor deliberately, in the same change)." >&2
    exit 1
fi

case "$MODE" in
dtype-contract) echo "$LABEL: ok (${#FILES[@]} dispatcher files, $FOUND f32-declaring functions)" ;;
input-eval) echo "$LABEL: ok ($FOUND $UNIT scanned)" ;;
esac
