#!/usr/bin/env bash
# scripts/check_gpu_tests_ignored_fixtures.sh — recall test for the
# `check_gpu_tests_ignored.sh` gate.
#
# The gate it guards enforces a rule ("a test reaching Device::Gpu carries
# #[ignore]") that nothing else in the tree can check, so a silent loss of
# recall there is indistinguishable from compliance. Each fixture under
# `scripts/fixtures/gpu_tests_ignored/` is a synthetic workspace paired with the
# outcome the gate must produce for it, driven through the gate's `--root`
# option. Half the cases are violations the gate must catch and half are
# legitimate shapes it must leave alone — a gate that fails everything is as
# useless as one that fails nothing, and only the pair pins it.
#
# WHY THE EXIT CODE ALONE IS NOT AN ASSERTION
#   The gate has several fail-closed paths that also exit 1 — a missing --root
#   directory, zero parsed workspace members, fewer members than crate dirs on
#   disk, a member whose src/ is gone, an unreadable package name, zero matched
#   test files. A case that only checks `exit == 1` is satisfied by every one of
#   them, so DELETING a fixture makes its case pass. (Measured: removing one
#   fixture tree and gutting another still reported "ok (9 cases)".) Each case
#   therefore pins three things: the exit code, the violation-class MARKER the
#   gate must print, and the specific LABEL it must name — plus an optional
#   string that must NOT appear. A wrong reason is a failure.
#
# The macro cases are the reason this file exists. A `macro_rules!` body
# declaring `#[test] fn $name()` names no readable fn at its definition site and
# emits no `fn` line at its invocation sites, so a source scanner can miss it in
# both directions at once: never flagged however much Metal it dispatches, and
# never listed for the runner either.
#
# Exit 0 = every fixture produced its expected outcome.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GATE="$ROOT/scripts/check_gpu_tests_ignored.sh"
FIX="$ROOT/scripts/fixtures/gpu_tests_ignored"

VIOLATION="ERROR: GPU-touching tests missing the #[ignore]"
UNREADABLE="ERROR: a macro_rules! body declares #[test] items this gate cannot read"
# Pins the file count too, so a fixture that lost its source file cannot pass by
# scanning nothing.
CLEAN="OK: every GPU-touching test carries #[ignore] (1 files"

# fixture | exit | marker that must appear | label that must appear | must NOT appear | what it proves
CASES=(
    "macro_gpu_no_ignore|1|${VIOLATION}|gpu_cell!{\$name}||a macro body binding Device::Gpu without #[ignore] is caught"
    "macro_gpu_via_helper|1|${VIOLATION}|cell!{\$name}||a macro body reaching Metal through a helper is caught"
    "macro_inline_body|1|${VIOLATION}|later_plain_gpu_no_ignore||a one-line generated fn does not swallow the rest of the file"
    "macro_close_comment|1|${VIOLATION}|second_cell!{\$name}||a commented close brace still ends the body, so blame lands on the right macro"
    "macro_unreadable|1|${UNREADABLE}|gpu_cell!||an assembled fn name fails closed, not silently"
    "macro_one_line|1|${UNREADABLE}|m!||a whole-macro-on-one-line #[test] fails closed"
    "plain_gpu_no_ignore|1|${VIOLATION}|plain_gpu_test||the original non-macro detection still fires"
    "macro_gpu_ignored|0|${CLEAN}|gpu_cell!{\$name}||a compliant macro body does not fire, and was read"
    "macro_cpu_no_ignore|0|${CLEAN}||NOTE:|a macro body that never reaches the GPU stays un-ignored"
    "macro_gpu_exempt|0|${CLEAN}||NOTE:|the per-test exemption marker works inside a macro body"
    "plain_gpu_ignored|0|${CLEAN}||NOTE:|the compliant non-macro shape stays green"
)

FAILED=0
PASSED=0

fail() { # fail <name> <detail>
    FAILED=$((FAILED + 1))
    printf '  FAIL %-22s %s\n' "$1" "$2"
}

for case in "${CASES[@]}"; do
    IFS='|' read -r name want marker label forbid what <<<"$case"

    # A deleted fixture must be a hard error, never a case that "passes"
    # because the gate refused to scan a directory that is not there.
    if [ ! -d "$FIX/$name" ]; then
        fail "$name" "fixture directory is missing: $FIX/$name"
        continue
    fi

    out=$(bash "$GATE" --root "$FIX/$name" 2>&1)
    got=$?

    if [ "$got" -ne "$want" ]; then
        fail "$name" "exit=$got (want $want) — $what"
        printf '%s\n' "$out" | sed 's/^/       | /'
        continue
    fi
    case "$out" in
        *"$marker"*) ;;
        *)
            fail "$name" "exit matched but the reason did not: no '$marker'"
            printf '%s\n' "$out" | sed 's/^/       | /'
            continue
            ;;
    esac
    if [ -n "$label" ]; then
        case "$out" in
            *"$label"*) ;;
            *)
                fail "$name" "right class, wrong subject: no '$label'"
                printf '%s\n' "$out" | sed 's/^/       | /'
                continue
                ;;
        esac
    fi
    if [ -n "$forbid" ]; then
        case "$out" in
            *"$forbid"*)
                fail "$name" "output contains '$forbid', which it must not"
                printf '%s\n' "$out" | sed 's/^/       | /'
                continue
                ;;
        esac
    fi

    PASSED=$((PASSED + 1))
    printf '  ok   %-22s exit=%s — %s\n' "$name" "$got" "$what"
done

# The list/enforce split, pinned on a fixture that holds one of each. The macro
# cell is GPU-touching and compliant, so it is enforced; only the plain test may
# reach `--list`, because the runner turns a listed name into a libtest filter
# and a `$metavar` matches no test. Asserting the exact stdout is what stops the
# split from silently becoming "macro cells are listed" (which under-matches in
# the runner) or "plain tests are dropped" (which stops running them).
want_list=$'fx\tplain_gpu_test'
got_list=$(bash "$GATE" --list --root "$FIX/macro_gpu_ignored" 2>/dev/null)
if [ "$got_list" = "$want_list" ]; then
    PASSED=$((PASSED + 1))
    printf '  ok   %-22s --list — macro cell enforced but not listed; plain test listed\n' "macro_gpu_ignored"
else
    fail "macro_gpu_ignored" "$(printf -- '--list got %q, want %q' "$got_list" "$want_list")"
fi

# The same run must still ANNOUNCE the excluded macro cells. `make gpu-test`
# calls only `--list`, so a note printed exclusively by the enforcing path is
# invisible to the one operator who needs it.
list_err=$(bash "$GATE" --list --root "$FIX/macro_gpu_ignored" 2>&1 >/dev/null)
case "$list_err" in
    *"NOTE: macro-generated GPU tests"*"gpu_cell!{\$name}"*)
        PASSED=$((PASSED + 1))
        printf '  ok   %-22s --list — the excluded macro cells are announced on stderr\n' "macro_gpu_ignored"
        ;;
    *)
        fail "macro_gpu_ignored" "--list did not announce the excluded macro cells on stderr"
        printf '%s\n' "$list_err" | sed 's/^/       | /'
        ;;
esac

if [ "$FAILED" -ne 0 ]; then
    echo "check-gpu-tests-ignored fixtures: FAIL ($FAILED of $((PASSED + FAILED)))" >&2
    exit 1
fi

echo "check-gpu-tests-ignored fixtures: ok ($PASSED cases)"
