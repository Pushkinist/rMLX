#!/usr/bin/env bash
# scripts/check_eval_lock_fixtures.sh — recall test for `check_eval_lock.sh`.
#
# A source gate is only worth its runtime if it still fires when the thing it
# looks for is spelled differently, nested deeper, or hidden behind a comment.
# All three of that gate's rules are one regex edit away from matching nothing
# and going permanently green, and RULE 2's anti-vacuity check only covers
# "no call sites at all" — not "the pattern stopped matching the ones that
# exist". Each fixture below is a synthetic scan root paired with the exit code
# AND THE RULE the gate must produce for it.
#
# WHY THE EXPECTED RULE IS CHECKED, NOT JUST THE EXIT CODE
#   The first version of this corpus asserted exit codes only, and every
#   must-fail fixture happened to contain no guarded call site — so each one
#   tripped RULE 2's anti-vacuity branch and exited 1 no matter what the rule
#   under test did. Measured on that corpus: deleting RULE 1 outright left it
#   GREEN; deleting RULE 3 outright left it GREEN. It had power over exactly
#   one rule. Every RULE 1 / RULE 3 fixture now ends with a correctly-guarded
#   call site (`guarded_anchor`) so RULE 2 is satisfied and the failure is
#   attributable, and the runner asserts which rule fired.
#
# Several cases pin bugs that were live during review:
#   long_safety_comment   a fixed 3-line window read a 4-line SAFETY comment
#                         inside the guarded closure as unguarded.
#   guarded_sibling       an enclosing-block scan must not let a guarded call
#                         earlier in the same fn launder a later bare one.
#   comment_launder       a trailing `// with_eval_lock: ...` satisfied RULE 2
#                         — and the old failure message handed the developer
#                         that exact string to paste.
#   column0_comment       a column-0 comment inside the closure was read as a
#                         block opener and severed the guard chain.
#   ffi_path_spelling     `sys::ffi::mlx_*` named the same functions and
#                         matched neither rule.
#   closure_applies       RULE 3's real invariant is "must not take the lock",
#                         not "must not evaluate": `Closure::apply` takes it,
#                         so a body applying another compiled closure
#                         deadlocks without calling .eval() at all.
#   closure_ufcs_eval     the UFCS `Array::eval(&y)` spelling.
#   closure_unbalanced_brace_string
#                         an unbalanced brace in a string literal ended the
#                         body scan early and hid the evaluation after it.
#   closure_evals         and an early awk version mis-read every clean closure
#                         as dirty because `exit` still runs `END`.
#
# Exit 0 = every fixture produced its expected exit code and rule.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GATE="$ROOT/scripts/check_eval_lock.sh"
FIX="$ROOT/scripts/fixtures/eval_lock"

# fixture | expected-exit | expected-rule ("-" when it must pass) | what it proves
CASES=(
    "clean|0|-|guarded call sites for all three guarded entry points pass"
    "long_safety_comment|0|-|a long comment inside the guarded closure is not a violation"
    "nested_block|0|-|the guard counts from two block levels up"
    "column0_comment|0|-|a column-0 comment inside the closure does not sever the guard chain"
    "closure_clean|0|-|a body that neither evaluates nor applies passes, MetalKernel::apply included"
    "prose_only|0|-|parenthesised mentions of the banned symbols in prose are not call sites"
    "local_fn_lookalike|0|-|a crate-local fn sharing an mlx-c name is not an FFI call (pins the sys:: anchor)"

    "lock_removed|1|RULE 2|dropping the guard from a call site is caught"
    "lock_removed_async|1|RULE 2|an unguarded sys::mlx_async_eval is caught on its own merits"
    "lock_removed_closure_apply|1|RULE 2|an unguarded sys::mlx_closure_apply is caught on its own merits"
    "guarded_sibling|1|RULE 2|a guarded call does not launder a bare one later in the same fn"
    "comment_launder|1|RULE 2|a trailing comment naming with_eval_lock does not satisfy the guard"
    "empty_tree|1|RULE 2|a scan root with no guarded call sites fails instead of passing vacuously"

    "banned_eval|1|RULE 1|sys::mlx_eval is caught"
    "banned_item|1|RULE 1|sys::mlx_array_item_* is caught"
    "banned_tostring|1|RULE 1|sys::mlx_array_tostring is caught"
    "banned_save|1|RULE 1|sys::mlx_save_safetensors — the rmlx convert write side — is caught"
    "banned_save_writer|1|RULE 1|sys::mlx_save_writer is caught"
    "banned_load_gguf|1|RULE 1|sys::mlx_load_gguf is caught"
    "ffi_path_spelling|1|RULE 1|the sys::ffi::mlx_* module path is caught"

    "closure_evals|1|RULE 3|a body calling .eval() is caught"
    "closure_evals_async|1|RULE 3|a body calling .async_eval() is caught"
    "closure_to_bytes|1|RULE 3|a body calling .to_bytes() is caught"
    "closure_ufcs_eval|1|RULE 3|the UFCS Array::eval(&y) spelling is caught"
    "closure_applies|1|RULE 3|a body applying another compiled closure is caught (takes the lock, deadlocks)"
    "closure_unbalanced_brace_string|1|RULE 3|an unbalanced brace in a string does not hide the evaluation after it"
)

FAILED=0
PASSED=0

for case in "${CASES[@]}"; do
    IFS='|' read -r name want_exit want_rule what <<<"$case"
    out=$("$GATE" "$FIX/$name" 2>&1)
    got=$?

    ok=1
    [ "$got" -eq "$want_exit" ] || ok=0
    if [ "$want_rule" != "-" ]; then
        printf '%s\n' "$out" | grep -q "$want_rule" || ok=0
    fi

    if [ "$ok" -eq 1 ]; then
        PASSED=$((PASSED + 1))
        printf '  ok   %-32s exit=%s %-7s — %s\n' "$name" "$got" "$want_rule" "$what"
    else
        FAILED=$((FAILED + 1))
        printf '  FAIL %-32s exit=%s (want %s / %s) — %s\n' \
            "$name" "$got" "$want_exit" "$want_rule" "$what"
        printf '%s\n' "$out" | sed 's/^/       | /'
    fi
done

if [ "$FAILED" -ne 0 ]; then
    echo "check-eval-lock fixtures: FAIL ($FAILED of $((PASSED + FAILED)))" >&2
    exit 1
fi

echo "check-eval-lock fixtures: ok ($PASSED cases)"
