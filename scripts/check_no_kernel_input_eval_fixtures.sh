#!/usr/bin/env bash
# scripts/check_no_kernel_input_eval_fixtures.sh — recall test for the
# `check_no_kernel_input_eval.sh` gate.
#
# A source gate is only worth its runtime if it still fires when the thing it
# looks for is spelled differently, moved, or hidden one call deep. Each
# fixture under `scripts/fixtures/no_kernel_input_eval/` is a synthetic scan
# root paired with the exit code the gate must produce for it. A gate change
# that loses recall fails here instead of passing silently on the real tree.
#
# Exit 0 = every fixture produced its expected exit code.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GATE="$ROOT/scripts/check_no_kernel_input_eval.sh"
FIX="$ROOT/scripts/fixtures/no_kernel_input_eval"

# fixture | min-files | expected-exit | what it proves
CASES=(
    "clean|1|0|a dispatcher with no eval passes"
    "plain_eval|1|1|the plain x.eval() spelling is caught"
    "ufcs_eval|1|1|the UFCS Array::eval(&x) spelling is caught"
    "common_helper|2|1|an eval hidden in a shared _common.rs scaffold is caught"
    "nested_dir|1|1|a dispatcher in a sub-directory is still scanned"
    "marker_same_line|1|0|a same-line eval-ok marker exempts its own call"
    "marker_above|1|0|an eval-ok marker on the lines above exempts one call"
    "marker_leaks_to_loop|1|1|one marker exempts one call, not a following loop"
    "trailing_comment|1|0|a comment naming .eval() is not itself a call"
    "empty_tree|1|1|an empty scan root fails instead of passing vacuously"
    "clean|2|1|fewer dispatchers than the pinned floor fails"
)

FAILED=0
PASSED=0

for case in "${CASES[@]}"; do
    IFS='|' read -r name min want what <<<"$case"
    out=$("$GATE" "$FIX/$name" "$min" 2>&1)
    got=$?
    if [ "$got" -eq "$want" ]; then
        PASSED=$((PASSED + 1))
        printf '  ok   %-22s min=%s exit=%s — %s\n' "$name" "$min" "$got" "$what"
    else
        FAILED=$((FAILED + 1))
        printf '  FAIL %-22s min=%s exit=%s (want %s) — %s\n' \
            "$name" "$min" "$got" "$want" "$what"
        printf '%s\n' "$out" | sed 's/^/       | /'
    fi
done

if [ "$FAILED" -ne 0 ]; then
    echo "check-no-kernel-input-eval fixtures: FAIL ($FAILED of $((PASSED + FAILED)))" >&2
    exit 1
fi

echo "check-no-kernel-input-eval fixtures: ok ($PASSED cases)"
