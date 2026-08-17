#!/usr/bin/env bash
# scripts/check_gpu_tests_ignored_fixtures.sh — recall test for the
# `check_gpu_tests_ignored.sh` gate.
#
# The gate it guards enforces a rule ("a test reaching Device::Gpu carries
# #[ignore]") that nothing else in the tree can check, so a silent loss of
# recall there is indistinguishable from compliance. Each fixture under
# `scripts/fixtures/gpu_tests_ignored/` is a synthetic workspace paired with the
# exit code the gate must produce for it, driven through the gate's `--root`
# option. Half the cases are violations the gate must catch and half are
# legitimate shapes it must leave alone — a gate that fails everything is as
# useless as one that fails nothing, and only the pair pins it.
#
# The macro cases are the reason this file exists. A `macro_rules!` body
# declaring `#[test] fn $name()` names no readable fn at its definition site and
# emits no `fn` line at its invocation sites, so a source scanner can miss it in
# both directions at once: never flagged however much Metal it dispatches, and
# never listed for the runner either.
#
# Exit 0 = every fixture produced its expected exit code.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GATE="$ROOT/scripts/check_gpu_tests_ignored.sh"
FIX="$ROOT/scripts/fixtures/gpu_tests_ignored"

# fixture | expected-exit | what it proves
CASES=(
    "macro_gpu_no_ignore|1|a macro body binding Device::Gpu without #[ignore] is caught"
    "macro_gpu_via_helper|1|a macro body reaching Metal through a helper is caught"
    "macro_unreadable|1|a #[test] the parser cannot read fails closed, not silently"
    "plain_gpu_no_ignore|1|the original non-macro detection still fires"
    "macro_gpu_ignored|0|a compliant macro body does not fire"
    "macro_cpu_no_ignore|0|a macro body that never reaches the GPU stays un-ignored"
    "macro_gpu_exempt|0|the per-test exemption marker works inside a macro body"
    "plain_gpu_ignored|0|the compliant non-macro shape stays green"
)

FAILED=0
PASSED=0

for case in "${CASES[@]}"; do
    IFS='|' read -r name want what <<<"$case"
    out=$(bash "$GATE" --root "$FIX/$name" 2>&1)
    got=$?
    if [ "$got" -eq "$want" ]; then
        PASSED=$((PASSED + 1))
        printf '  ok   %-22s exit=%s — %s\n' "$name" "$got" "$what"
    else
        FAILED=$((FAILED + 1))
        printf '  FAIL %-22s exit=%s (want %s) — %s\n' "$name" "$got" "$want" "$what"
        printf '%s\n' "$out" | sed 's/^/       | /'
    fi
done

# The list/enforce split, pinned on a fixture that holds one of each. The macro
# cell is GPU-touching and compliant, so it is enforced; only the plain test may
# reach `--list`, because the runner turns a listed name into a libtest filter
# and a `$metavar` matches no test. Asserting the exact output is what stops the
# split from silently becoming "macro cells are listed" (which under-matches in
# the runner) or "plain tests are dropped" (which stops running them).
want_list=$'fx\tplain_gpu_test'
got_list=$(bash "$GATE" --list --root "$FIX/macro_gpu_ignored" 2>/dev/null)
if [ "$got_list" = "$want_list" ]; then
    PASSED=$((PASSED + 1))
    printf '  ok   %-22s --list — macro cell enforced but not listed; plain test listed\n' "macro_gpu_ignored"
else
    FAILED=$((FAILED + 1))
    printf '  FAIL %-22s --list — got %q, want %q\n' "macro_gpu_ignored" "$got_list" "$want_list"
fi

if [ "$FAILED" -ne 0 ]; then
    echo "check-gpu-tests-ignored fixtures: FAIL ($FAILED of $((PASSED + FAILED)))" >&2
    exit 1
fi

echo "check-gpu-tests-ignored fixtures: ok ($PASSED cases)"
