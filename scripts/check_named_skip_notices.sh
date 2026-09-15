#!/usr/bin/env bash
# scripts/check_named_skip_notices.sh — every classified GPU test that announces
# its own stand-down names itself while doing it.
#
# WHY
#   `scripts/run_gpu_tests.sh` reads the tests' own words to tell a cell that
#   held from one that was never asked. The notice it reads is
#   `SKIP <test>: <why>`, and the NAME is the whole of the attribution: it is
#   what puts the cell on the stand-down list an operator reads, and what lets
#   the shader-validation census drop that cell's pinned hit count instead of
#   expecting a count the run could not produce.
#
#   A notice spelled `SKIP: <why>` passes the runner's stand-down pattern and
#   names nothing. The runner counts it — it will not attribute it to whichever
#   test was nearby — so the run is marked INCOMPLETE with a number and no list,
#   and the reader is told that something stood down without being told what.
#   That is the state `make ci-perf` ends in today on a host holding every
#   snapshot, and no number of snapshots fixes it: the notice cannot be
#   attributed at any point after it is printed.
#
#   So the rule is enforced where it can be fixed — at the source line. This
#   scan is the recall half of the runner's report: the runner says how many
#   notices could not be attributed, this says which lines they come from.
#
# WHAT IS SCANNED
#   The classified GPU tests, from `check_gpu_tests_ignored.sh --list` — the
#   same population the runner selects, derived from the same classifier rather
#   than listed again here. A notice in a helper, or in a test the classifier
#   does not name, is out of scope: the runner never selects it, so it never
#   reaches a report.
#
# WHAT IS ACCEPTED
#   A notice whose name is the enclosing test fn, either written out or built
#   from a format placeholder (`SKIP {test}: …`, the spelling a test that passes
#   its own name uses). A notice naming a DIFFERENT test is refused: the runner
#   would list it under that other name and send the reader after a cell that
#   ran.
#
# Exit 0 = every notice in a classified GPU test names its own test.
# Exit 1 = at least one does not; each is named with its file, line and reason.
# Exit 2 = the scan could not be trusted (no classification, no crates).

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
while [ $# -gt 0 ]; do
    case "$1" in
        --root) ROOT="${2:?--root needs a value}"; shift 2 ;;
        *) echo "ERROR: unknown argument '$1' (expected --root <dir>)." >&2; exit 2 ;;
    esac
done

listing="$(bash "${ROOT}/scripts/check_gpu_tests_ignored.sh" --list --root "${ROOT}" 2>/dev/null)"
if [ -z "${listing}" ]; then
    echo "ERROR: check_gpu_tests_ignored.sh --list produced no GPU tests — the scan would" >&2
    echo "pass by having nothing to look at." >&2
    exit 2
fi

# The test-bearing files, in the classifier's own population: sibling test files
# under src/ and the integration tests. A notice anywhere else is in a fn the
# classifier cannot have named.
files="$(find "${ROOT}/crates" \
    \( -path '*/src/*_tests.rs' -o -path '*/src/*/tests.rs' -o -path '*/src/tests.rs' \
       -o -path '*/tests/*.rs' \) -type f 2>/dev/null | sort)"
if [ -z "${files}" ]; then
    echo "ERROR: no test-bearing sources found under ${ROOT}/crates." >&2
    exit 2
fi

# One `<file>\t<line>\t<enclosing fn>\t<verdict>\t<text>` record per stand-down
# notice, where the verdict is what the notice names:
#
#   self   — the enclosing fn, written out or as a format placeholder
#   none   — nothing; `SKIP: <why>`
#   other  — a different identifier
#
# The enclosing fn is the nearest preceding `fn`, which is what a notice inside
# a closure or a nested block belongs to as well. A notice in a helper resolves
# to the helper, which the attribution step below then drops.
notices="$(printf '%s\n' "${files}" | while IFS= read -r f; do
    [ -n "${f}" ] || continue
    awk -v FILE="${f}" '
        match($0, /^[[:space:]]*(pub[[:space:]]+(\([^)]*\)[[:space:]]*)?)?(async[[:space:]]+)?fn[[:space:]]+[A-Za-z_][A-Za-z0-9_]*/) {
            head = substr($0, RSTART, RLENGTH)
            sub(/^.*fn[[:space:]]+/, "", head)
            cur = head
        }
        # A stand-down notice is a quoted literal carrying SKIP as a whole word.
        # Both halves are required: the bare word appears in prose comments and
        # in variable names, and a quote alone is every other string in the file.
        /"/ && /(^|[^A-Za-z0-9_])SKIP([^A-Za-z0-9_]|$)/ {
            verdict = "none"
            if (match($0, /SKIP[[:space:]]+\{[^}]*\}[[:space:]]*:/)) {
                verdict = "self"
            } else if (match($0, /SKIP[[:space:]]+[A-Za-z_][A-Za-z0-9_]*[[:space:]]*:/)) {
                named = substr($0, RSTART, RLENGTH)
                sub(/^SKIP[[:space:]]+/, "", named)
                sub(/[[:space:]]*:$/, "", named)
                verdict = (named == cur) ? "self" : "other:" named
            }
            text = $0
            sub(/^[[:space:]]+/, "", text)
            printf "%s\t%d\t%s\t%s\t%s\n", FILE, NR, cur, verdict, text
        }
    ' "${f}"
done)"

violations=""
n=0
while IFS=$'\t' read -r v_file v_line v_fn v_verdict v_text; do
    [ -n "${v_file:-}" ] || continue
    case "${v_verdict}" in self) continue ;; esac
    # Attribute to a crate by path, then keep only what the classifier named.
    rel="${v_file#"${ROOT}"/crates/}"
    crate="${rel%%/*}"
    case $'\n'"${listing}"$'\n' in
        *$'\n'"${crate}"$'\t'"${v_fn}"$'\n'*) ;;
        *) continue ;;
    esac
    case "${v_verdict}" in
        none) why="names no test" ;;
        other:*) why="names ${v_verdict#other:}, not itself" ;;
        *) why="unclassified notice" ;;
    esac
    violations="${violations}    ${rel}:${v_line} — ${crate} ${v_fn} ${why}"$'\n'
    violations="${violations}        ${v_text}"$'\n'
    n=$((n + 1))
done <<< "${notices}"

if [ "${n}" -gt 0 ]; then
    echo "ERROR: ${n} stand-down notice(s) in classified GPU tests do not name their test:" >&2
    printf '%s' "${violations}" >&2
    echo >&2
    echo "scripts/run_gpu_tests.sh counts such a notice and lists it nowhere, so the run" >&2
    echo "is INCOMPLETE with a number and no name. Write it as" >&2
    echo "  SKIP <this test fn>: <why>" >&2
    echo "See docs/TESTING.md." >&2
    exit 1
fi

echo "OK: every stand-down notice in a classified GPU test names its own test."
