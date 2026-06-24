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
# Every non-test .rs file under crates/rmlx-models/src/ is scanned for lines
# that:
#   (a) contain scalar_f32( (as actual code, not a pure line comment), AND
#   (b) are NOT followed by .astype( on the SAME LINE, AND
#   (c) are NOT followed by .astype( on the NEXT NON-BLANK LINE
#       (handles multi-line method chains where .astype is wrapped to the
#       next line — but only when the scalar_f32 line does not end the
#       statement, i.e. does not end with ';' or '{' after stripping
#       trailing whitespace), AND
#   (d) do NOT have an  // f32-ok: <reason>  marker on the SAME LINE, OR on
#       ANY preceding comment line that is part of the contiguous comment block
#       immediately above the scalar_f32( line.
#
# ALLOWLISTING
# ------------
# To allow a genuine f32-only scalar (e.g. one consumed by a raw-f32 param or
# inside a module where every tensor is already f32):
#   - Same-line:   scalar_f32(x) // f32-ok: <reason>
#   - Preceding:   // f32-ok: <reason>
#                  let s = scalar_f32(x);
#     (the marker may appear in a multi-line comment block directly above)
#
# THE CANONICAL FIX
# -----------------
# scalar_f32(x).astype(operand.dtype(), device)?
#
# Exit 0 = clean. Exit 1 = violation found.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Arch-layer scope: all .rs files under crates/rmlx-models/src/.
# Excludes test files (*_tests.rs, files under tests/ directories).
SCAN_DIR="${REPO_ROOT}/crates/rmlx-models/src"

violations=()

while IFS= read -r -d '' f; do
    # Use awk to scan each file.
    #
    # State machine:
    #   in_comment_block  — currently inside a contiguous run of // comment lines
    #   block_has_f32ok   — the current/recent comment block contained // f32-ok:
    #   pending           — scalar_f32( seen; awaiting next non-blank line for
    #                       multi-line chain check
    #   pending_line/text — saved location of the pending scalar_f32( line
    #   found             — 1 if any violation detected (drives exit code)
    #
    # Transition logic:
    #   On a pure comment line (^[[:space:]]*//) :
    #     - Still in (or entering) a comment block.
    #     - If it contains "// f32-ok:", set block_has_f32ok = 1.
    #
    #   On a blank line:
    #     - Ends the comment block (reset in_comment_block and block_has_f32ok).
    #     - If pending, skip blank lines.
    #
    #   On any other (code) line:
    #     - The comment block ended with the preceding comments; capture
    #       block_has_f32ok as prev_f32ok, then reset in_comment_block.
    #     - Check pending from previous iteration.
    #     - If line contains scalar_f32( (not in a pure comment):
    #         * same-line // f32-ok:   → OK
    #         * prev_f32ok             → OK
    #         * same-line .astype(     → OK
    #         * terminated statement (ends with ; or {) → violation
    #         * else                   → set pending
    #     - Reset prev_f32ok / block state for next line.

    awk -v file="$f" '
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
                # Blank line ends any comment block.
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

        # ---- code line (anything that is not blank and not a pure comment) ----
        {
            # Capture whether the preceding comment block had f32-ok.
            # (In the previous iteration a code line already cleared these, so
            # prev_f32ok here reflects the most recent comment block directly
            # above this code line.)
            if (in_comment_block) {
                # We were in a comment block; its f32-ok status is now available.
                prev_f32ok = block_has_f32ok
            }
            # Reset the comment-block state now that we have a code line.
            in_comment_block = 0
            block_has_f32ok  = 0

            # Handle any pending check from the previous code line.
            if (pending) {
                if (index($0, ".astype(") > 0) {
                    # Multi-line chain resolved — OK.
                    pending = 0
                } else {
                    # Next code line has no .astype( → violation.
                    print file ":" pending_line ": unguarded scalar_f32(  →  " pending_text
                    found = 1
                    pending = 0
                }
                # After resolving pending, continue to check this line for
                # its own scalar_f32( (fall through).
            }

            # Check this code line for scalar_f32(.
            if (index($0, "scalar_f32(") > 0) {
                # (a) Same-line f32-ok allowlist.
                if (index($0, "// f32-ok:") > 0) { prev_f32ok = 0; next }
                # (b) Preceding comment block had f32-ok.
                if (prev_f32ok) { prev_f32ok = 0; next }
                # (c) Same-line .astype( chain.
                if (index($0, ".astype(") > 0) { prev_f32ok = 0; next }
                # (d) Is the statement terminated on this line?
                stripped = $0
                gsub(/^[[:space:]]+/, "", stripped)
                gsub(/[[:space:]]+$/, "", stripped)
                last_ch = substr(stripped, length(stripped), 1)
                if (last_ch == ";" || last_ch == "{") {
                    # Statement ends here — violation.
                    print file ":" NR ": unguarded scalar_f32(  →  " $0
                    found = 1
                    prev_f32ok = 0
                    next
                }
                # Non-terminated: defer to next non-blank code line.
                pending      = 1
                pending_line = NR
                pending_text = $0
                prev_f32ok   = 0
                next
            }

            # Not a scalar_f32( line — just update prev_f32ok for next line.
            prev_f32ok = (index($0, "// f32-ok:") > 0)
        }

        END {
            # Unresolved pending at EOF.
            if (pending) {
                print file ":" pending_line ": unguarded scalar_f32(  →  " pending_text
                found = 1
            }
            exit !found
        }
    ' "$f" 2>/dev/null && violations+=("$f")
done < <(
    find "${SCAN_DIR}" -name "*.rs" \
        -not -path "*/target/*" \
        -not -path "*/tests/*" \
        -not -name "*_tests.rs" \
        -print0
)

if [ ${#violations[@]} -gt 0 ]; then
    echo "" >&2
    echo "CANONICAL FIX: scalar_f32(x).astype(operand.dtype(), device)?" >&2
    echo "" >&2
    echo "To allowlist a genuine f32-only scalar, add an inline marker:" >&2
    echo "  // f32-ok: <reason why f32 is safe here>" >&2
    echo "  (on the same line as scalar_f32, or in the comment block immediately above)" >&2
    echo "" >&2
    exit 1
fi

echo "OK: no unguarded scalar_f32( in arch-layer code."
