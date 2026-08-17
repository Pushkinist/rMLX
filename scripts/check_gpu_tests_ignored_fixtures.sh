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
# EVERY CASE RUNS ONCE PER AWK ON THE MACHINE
#   The gate is an awk program, and awk implementations genuinely disagree. A
#   bracket range of octal escapes (`[\300-\337]`, the obvious way to match a
#   UTF-8 continuation byte) is accepted by BSD awk and is a hard syntax error
#   in gawk — so a gate green on a developer's Mac can be hard-DOWN on the Linux
#   CI runner, which is strictly worse than the blind spot it exists to close.
#   Byte-vs-character indexing and the POSIX classes differ too: `[[:print:]]`
#   calls a UTF-8 continuation byte printable in BSD awk and mawk under a UTF-8
#   locale but not under C, and gawk indexes characters where the others index
#   bytes. A suite that checks one implementation cannot see any of that, so it
#   runs under each — and when only one is installed it SAYS so rather than
#   reporting a clean run that proved less than it looks like it did.
#
# Exit 0 = every fixture produced its expected outcome, under every awk present.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GATE="$ROOT/scripts/check_gpu_tests_ignored.sh"
FIX="$ROOT/scripts/fixtures/gpu_tests_ignored"

# Outer pass: discover the awks and re-run this suite under each, shimming
# `awk` onto PATH (the gate invokes `awk`, not an absolute path). The inner
# runs set GATE_AWK_SHIM and fall straight through to the cases below.
if [ -z "${GATE_AWK_SHIM:-}" ]; then
    awks=()
    for cand in awk gawk mawk; do
        found="$(command -v "$cand" 2>/dev/null)" || continue
        [ -n "$found" ] || continue
        dup=0
        for seen in ${awks[@]+"${awks[@]}"}; do
            # Same file under two names (Debian's `awk` -> `mawk`) is one
            # implementation, not two.
            if [ "$found" -ef "$seen" ]; then dup=1; break; fi
        done
        [ "$dup" -eq 0 ] && awks+=("$found")
    done

    if [ ${#awks[@]} -eq 0 ]; then
        echo "ERROR: no awk found — the gate cannot run, so this suite proves nothing." >&2
        exit 1
    fi
    if [ ${#awks[@]} -eq 1 ]; then
        echo "NOTE: only one awk implementation here (${awks[0]}). This run does NOT"
        echo "      cover the gate's awk portability: a construct BSD awk accepts and"
        echo "      gawk rejects outright leaves this suite green and CI hard-down."
        echo "      \`brew install gawk mawk\` (or apt) to check it here, not in CI."
    fi

    shim_root="$(mktemp -d)"
    trap 'rm -rf "$shim_root"' EXIT
    rc=0
    for impl in "${awks[@]}"; do
        shim="$shim_root/$(basename "$impl")"
        mkdir -p "$shim"
        ln -sf "$impl" "$shim/awk"
        echo "== awk: $impl =="
        GATE_AWK_SHIM=1 PATH="$shim:$PATH" bash "${BASH_SOURCE[0]}" || rc=1
    done
    exit "$rc"
fi

VIOLATION="ERROR: GPU-touching tests missing the #[ignore]"
UNREADABLE="ERROR: this gate could not classify part of a scanned file"
# Pins the file count too, so a fixture that lost its source file cannot pass by
# scanning nothing.
CLEAN="OK: every GPU-touching test carries #[ignore] (1 files"

# fixture | exit | marker that must appear | label that must appear | must NOT appear | what it proves
CASES=(
    "macro_gpu_no_ignore|1|${VIOLATION}|gpu_cell!{\$name}||a macro body binding Device::Gpu without #[ignore] is caught"
    "macro_gpu_via_helper|1|${VIOLATION}|cell!{\$name}||a macro body reaching Metal through a helper is caught"
    "macro_inline_body|1|${VIOLATION}|later_plain_gpu_no_ignore||a one-line generated fn does not swallow the rest of the file"
    "macro_inline_string_brace|1|${VIOLATION}|second_cell!{\$name}||a } inside a string literal does not make the line look unbalanced"
    "fn_url_in_string|1|${VIOLATION}|later_plain_gpu_no_ignore||a // inside a string literal is not a comment"
    "fn_char_literal_quote|1|${VIOLATION}|later_plain_gpu_no_ignore||a \" inside a char literal does not desynchronise the string scan"
    "fn_comment_brace|1|${VIOLATION}|plain_gpu_no_ignore||a } inside a trailing comment does not make the line look self-contained"
    "trait_signature_fns|1|${VIOLATION}|after_trait_gpu_no_ignore||one-line signature-only fns neither latch nor hard-fail the run"
    "trait_where_signature|1|${UNREADABLE}|fn never closed||a where-split signature latches, and is loud ONLY when nothing closes the latch"
    "trait_where_signature_open_hole|0|${CLEAN}||ERROR|KNOWN OPEN HOLE: one nested block closes the latch and the swallowed GPU test is reported clean"
    "fn_never_closes|1|${UNREADABLE}|fn never closed||a capture that cannot terminate is reported, not skipped"
    "attr_multiline_ignore|1|${VIOLATION}|later_plain_gpu_no_ignore|probe|a wrapped #[ignore = \"..\" ] closes, and is read rather than merely un-latched"
    "attr_trailing_comment|1|${VIOLATION}|gpu_after_comment_attr||a comment after an attribute's closing ] does not latch the capture"
    "attr_never_closes|1|${UNREADABLE}|attribute never closed||an attribute capture that cannot terminate is reported, not swallowed"
    "exempt_in_body|1|${VIOLATION}|gpu_marker_inside_body||a marker among a fn's statements exempts nothing"
    "exempt_in_body|1|${VIOLATION}|gpu_after_marker_in_body||and does not carry to the next fn"
    "macro_one_line_no_test|1|${VIOLATION}|gpu_cell!{\$name}||a one-line macro declaring no test is stepped over, not latched"
    "tokio_test_gpu|1|${VIOLATION}|tokio_gpu_no_ignore||#[tokio::test] classifies like #[test]"
    "tokio_test_gpu|1|${VIOLATION}|tokio_flavored_gpu_no_ignore||the parameterised #[tokio::test(..)] spelling classifies too"
    "macro_close_comment|1|${VIOLATION}|second_cell!{\$name}||a commented close brace still ends the body, so blame lands on the right macro"
    "macro_unreadable|1|${UNREADABLE}|gpu_cell!||an assembled fn name fails closed, not silently"
    "macro_one_line|1|${UNREADABLE}|one_line_cell!||a whole-macro-on-one-line #[test] fails closed"
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

# The inverse of the missing-directory check above: a fixture tree that no case
# names is never executed and never noticed. Lining up N directories against N
# case entries by hand-count is exactly the sort of bookkeeping that silently
# drifts, and an unreferenced fixture is a test someone wrote and nothing runs.
for dir in "$FIX"/*/; do
    name="$(basename "$dir")"
    referenced=0
    for case in "${CASES[@]}"; do
        [ "${case%%|*}" = "$name" ] && referenced=1 && break
    done
    if [ "$referenced" -eq 0 ]; then
        fail "$name" "fixture directory is not referenced by any CASES entry"
    fi
done

if [ "$FAILED" -ne 0 ]; then
    echo "check-gpu-tests-ignored fixtures: FAIL ($FAILED of $((PASSED + FAILED)))" >&2
    exit 1
fi

echo "check-gpu-tests-ignored fixtures: ok ($PASSED cases)"
