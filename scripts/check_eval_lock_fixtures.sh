#!/usr/bin/env bash
# scripts/check_eval_lock_fixtures.sh — recall test for `check_eval_lock.sh`.
#
# A source gate is only worth its runtime if it still fires when the thing it
# looks for is spelled differently, nested deeper, or hidden behind a comment.
# Both of that gate's positive rules are one regex edit away from matching
# nothing and going permanently green, and its anti-vacuity check only covers
# "no call sites at all" — not "the pattern stopped matching the ones that
# exist". Each fixture below is a synthetic scan root paired with the exit code
# the gate must produce for it, so a gate change that loses recall fails here
# instead of passing silently on the real tree.
#
# Several cases pin bugs that were actually present during review:
#   long_safety_comment  the first version scanned a fixed 3-line window, so a
#                        4-line SAFETY comment inside the guarded closure — the
#                        crate's own convention — reported the call unguarded.
#   guarded_sibling      an enclosing-block scan must not let a guarded call
#                        earlier in the same function launder a later bare one.
#   ffi_path_spelling    `sys::ffi::mlx_*` named the same functions and matched
#                        neither rule.
#   closure_evals        RULE 3; and an early awk version mis-read every clean
#                        closure as dirty because `exit` still runs `END`.
#
# Exit 0 = every fixture produced its expected exit code.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GATE="$ROOT/scripts/check_eval_lock.sh"
FIX="$ROOT/scripts/fixtures/eval_lock"

# fixture | expected-exit | what it proves
CASES=(
    "clean|0|guarded call sites in both one-line and block form pass"
    "long_safety_comment|0|a long comment inside the guarded closure is not a violation"
    "nested_block|0|the guard counts from two block levels up"
    "closure_clean|0|a closure body that does not evaluate passes, braces in strings and all"
    "prose_only|0|comments naming the banned symbols are not call sites"
    "lock_removed|1|dropping the guard from a call site is caught"
    "guarded_sibling|1|a guarded call does not launder a bare one later in the same fn"
    "banned_item|1|sys::mlx_array_item_* is caught"
    "banned_tostring|1|sys::mlx_array_tostring is caught"
    "banned_save|1|sys::mlx_save_safetensors — the rmlx convert write side — is caught"
    "ffi_path_spelling|1|the sys::ffi::mlx_* module path is caught"
    "closure_evals|1|a Closure::from_fn body that evaluates is caught (self-deadlock)"
    "empty_tree|1|a scan root with no guarded call sites fails instead of passing vacuously"
)

FAILED=0
PASSED=0

for case in "${CASES[@]}"; do
    IFS='|' read -r name want what <<<"$case"
    out=$("$GATE" "$FIX/$name" 2>&1)
    got=$?
    if [ "$got" -eq "$want" ]; then
        PASSED=$((PASSED + 1))
        printf '  ok   %-20s exit=%s — %s\n' "$name" "$got" "$what"
    else
        FAILED=$((FAILED + 1))
        printf '  FAIL %-20s exit=%s (want %s) — %s\n' "$name" "$got" "$want" "$what"
        printf '%s\n' "$out" | sed 's/^/       | /'
    fi
done

if [ "$FAILED" -ne 0 ]; then
    echo "check-eval-lock fixtures: FAIL ($FAILED of $((PASSED + FAILED)))" >&2
    exit 1
fi

echo "check-eval-lock fixtures: ok ($PASSED cases)"
