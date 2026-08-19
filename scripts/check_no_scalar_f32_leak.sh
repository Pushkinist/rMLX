#!/usr/bin/env bash
# scripts/check_no_scalar_f32_leak.sh — CI gate: flag unguarded scalar_f32( in arch-layer code.
#
# PROBLEM
# -------
# Calling scalar_f32(x) creates a strong-F32 scalar Array. When that Array is
# then combined (multiply / add / divide / etc.) with a BF16 activation, MLX
# promotes the result to F32, silently contaminating the residual stream, Q/K/V,
# and the KV cache. This class of bug has shipped at least three times in rMLX
# and is invisible at review because the call site looks harmless.
#
# WHAT THIS GATE CHECKS
# ---------------------
# Every non-test .rs file under the crates that own `.metal` kernels — the
# roots derived from `scripts/metal_dirs.sh`, so this gate and
# `check_kernel_dtype_contract.sh` cannot drift apart — minus `rmlx-mlx` (see
# the exclusion beside the list, which is a recorded finding, not a preference).
# Scanned (excluding out-of-scope arch directories: laguna/, dr_venus/) for
# lines that:
#
# The scope was `crates/rmlx-models/src` alone until 2026-08. That was one of
# three reasons this gate could not see the TurboFlash promotion: the leak sat
# in `rmlx-kv-quant`, outside the scan root entirely. Widening it does not make
# this gate sufficient for that class — it keys on `scalar_f32(`, and that leak
# had none — but the KV codec layer runs on the same promotion path and carries
# live `scalar_f32(` call sites in the decode hot path, so it belongs in scope.
#   (a) contain scalar_f32( (as actual code, not a pure line comment), AND
#   (b) are NOT followed by a non-F32 .astype( on the SAME LINE, AND
#   (c) are NOT followed by a non-F32 .astype( within the immediately following
#       continuation lines of the same multi-line chain expression (lines that
#       do not terminate the statement — i.e. no trailing ';' or '}' after
#       stripping whitespace), AND
#   (d) do NOT have a  // f32-ok: <reason>  marker on the SAME LINE, OR on
#       ANY preceding comment line that is part of the contiguous comment block
#       immediately above the scalar_f32( line.
#
# WHAT COUNTS AS A GUARD
# ----------------------
# A guarding .astype( call MUST NOT target Dtype::F32 (which preserves the
# scalar f32 rather than casting to the activation dtype). The following forms
# are rejected as guards (false guards — they keep the f32):
#   .astype(Dtype::F32,        → rejected (explicit F32 target)
#   .astype(F32,               → rejected
#   .astype(rmlx_mlx::Dtype::F32,  → rejected
# Any other .astype( target (e.g. Dtype::Bf16, operand.dtype(), x.dtype(),
# Dtype::I32, …) is accepted as a legitimate guard.
#
# SCOPE EXCLUSIONS
# ----------------
# Out-of-scope arch directories are excluded from the scan:
#   laguna/      — excluded (not bench-provable, out of scope per project rules)
#   dr_venus/    — excluded (same reason, defensive)
#
# ALLOWLISTING
# ------------
# To allow a genuine f32-only scalar (e.g. inside a vision tower or audio
# encoder that runs entirely in f32, or a scalar passed to an f32-only API):
#   - Same-line:   scalar_f32(x) // f32-ok: <reason>
#   - Preceding:   // f32-ok: <reason>
#                  let s = scalar_f32(x);
#   The marker may appear in a multi-line comment block directly above the
#   scalar_f32( line. The reason must be specific (e.g. "tower is f32",
#   "output is Vec<f32>", "passed to compile_shapeless", "terminal logits").
#
# THE CANONICAL FIX
# -----------------
# scalar_f32(x).astype(operand.dtype(), device)?
#
# Exit 0 = clean. Exit 1 = violation found.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Scope: the crate `src/` directories that own `.metal` kernels, single-sourced
# from `scripts/metal_dirs.sh` (which names each crate's `metal/` directory;
# the Rust sits one level up). Excludes test files and out-of-scope arch
# directories.
# shellcheck source=scripts/metal_dirs.sh
. "$(dirname "${BASH_SOURCE[0]}")/metal_dirs.sh"
SCAN_DIRS=()
for d in "${METAL_DIRS[@]}"; do
    root="${d%/metal}"
    # `rmlx-mlx` is deliberately excluded, and the reason is a finding rather
    # than a preference: it DEFINES `scalar_f32`, and its cached GELU constants
    # (`ops/activation.rs`) are f32 arrays multiplied straight into a bf16 `x`
    # with no cast back — `gelu`/`gelu_tanh` return f32 for a bf16 input today.
    # That is a live instance of this gate's own class, on the MLP path of every
    # arch that calls them. Bringing the crate in scope now would force either a
    # marker on that suspect (laundering a real finding into an allowlist entry)
    # or a numerics change across every gelu arch inside an unrelated fix.
    # Widen this list when that is settled on its own branch.
    case "${root}" in
    *"/crates/rmlx-mlx/src") continue ;;
    esac
    SCAN_DIRS+=("${root}")
done

violations=()

while IFS= read -r -d '' f; do
    # Use awk to scan each file.
    #
    # State machine:
    #   in_comment_block  — currently inside a contiguous run of // comment lines
    #   block_has_f32ok   — the current/recent comment block contained // f32-ok:
    #   prev_f32ok        — the preceding comment block had f32-ok (set when the
    #                       first code line after the block is reached)
    #   pending           — scalar_f32( seen; awaiting continuation lines for a
    #                       multi-line chain check
    #   pending_line/text — saved location of the pending scalar_f32( line
    #   found             — 1 if any violation detected (drives exit code)
    #
    # Key invariant: a guarding .astype( MUST NOT target F32.
    # is_f32_astype(line): returns 1 when the line contains .astype( targeting F32.
    # has_good_astype(line): .astype( present AND is NOT an F32 target.
    #
    # Pending (multi-line chain) logic:
    #   Once pending is set, keep scanning continuation lines (lines that do not
    #   terminate the statement, i.e. do not end with ';' or '}').  Accept if any
    #   continuation carries a non-F32 .astype(.  Emit a violation only when a
    #   terminating line is reached without having seen a non-F32 .astype(.

    awk -v file="$f" '
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

        function is_terminated(line,    s) {
            # Returns 1 if the line ends a statement (trailing ; or }).
            s = line
            gsub(/^[[:space:]]+/, "", s)
            gsub(/[[:space:]]+$/, "", s)
            c = substr(s, length(s), 1)
            return (c == ";" || c == "}")
        }

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
            # Capture preceding comment block f32-ok status on first code line.
            if (in_comment_block) {
                prev_f32ok = block_has_f32ok
            }
            in_comment_block = 0
            block_has_f32ok  = 0

            # Handle pending multi-line chain from a previous scalar_f32( line.
            if (pending) {
                if (has_good_astype($0)) {
                    # Chain resolved with a non-F32 .astype( — OK.
                    pending = 0
                } else if (is_terminated($0)) {
                    # Statement terminated without a non-F32 .astype( → violation.
                    print file ":" pending_line ": unguarded scalar_f32(  →  " pending_text
                    found = 1
                    pending = 0
                }
                # else: non-terminated continuation — keep pending open, scan next line.
                # Fall through to check this line for its own scalar_f32(.
            }

            # Check this code line for scalar_f32(.
            if (index($0, "scalar_f32(") > 0) {
                # (a) Same-line f32-ok allowlist.
                if (index($0, "// f32-ok:") > 0) { prev_f32ok = 0; next }
                # (b) Preceding comment block had f32-ok.
                if (prev_f32ok) { prev_f32ok = 0; next }
                # (c) Same-line non-F32 .astype( chain.
                if (has_good_astype($0)) { prev_f32ok = 0; next }
                # (d) Same-line .astype( that targets F32 — counts as NO guard.
                # (falls through to the termination check below)

                # (e) Is the statement terminated on this line?
                if (is_terminated($0)) {
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
            prev_f32ok = (index($0, "// f32-ok:") > 0)
        }

        END {
            if (pending) {
                print file ":" pending_line ": unguarded scalar_f32(  →  " pending_text
                found = 1
            }
            exit !found
        }
    ' "$f" 2>/dev/null && violations+=("$f")
done < <(
    find "${SCAN_DIRS[@]}" -name "*.rs" \
        -not -path "*/target/*" \
        -not -path "*/tests/*" \
        -not -name "*_tests.rs" \
        -not -path "*/laguna/*" \
        -not -path "*/dr_venus/*" \
        -print0
)

if [ ${#violations[@]} -gt 0 ]; then
    echo "" >&2
    echo "CANONICAL FIX: scalar_f32(x).astype(operand.dtype(), device)?" >&2
    echo "  NOTE: .astype(Dtype::F32, ...) does NOT count as a guard." >&2
    echo "" >&2
    echo "To allowlist a genuine f32-only scalar, add an inline marker:" >&2
    echo "  // f32-ok: <reason why f32 is safe here>" >&2
    echo "  (on the same line as scalar_f32, or in the comment block immediately above)" >&2
    echo "" >&2
    exit 1
fi

echo "OK: no unguarded scalar_f32( in arch-layer code."
