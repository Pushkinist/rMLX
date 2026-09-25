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
#   from the `{test}` placeholder — the spelling a test that passes its own name
#   uses. That placeholder is accepted by its exact text and no other: `{other}`
#   reads identically here and names a cell that ran.
#
#   A notice naming a DIFFERENT test is refused for the same reason: the runner
#   would list it under a name no libtest filter reaches.
#
# WHAT IS ALSO REFUSED: A GUARD THAT SAYS NOTHING AT ALL
#   The rule above reads notices. The second rule reads their ABSENCE, which is
#   the shape the first one cannot see: a line carrying no SKIP token is not a
#   notice to it, libtest reports the cell as `ok`, and the runner's harvest
#   never finds it — so a cell that stood down is counted as a pass by every
#   gate in the tree, and no number of snapshots changes that.
#
#   The shape is a block opened by a line reading an environment variable and
#   closed by a `return` that CARRIES NO VALUE, with no notice inside it.
#   `return;`, a bare `return`, `return None;` and `return Ok(());` stand a cell
#   down; `return Some(p);` or `return v.parse().unwrap_or(4096);` is a result,
#   and a guard whose every exit carries one has answered rather than skipped —
#   the two-env-key resolvers in this tree are that shape. The needle is not
#   anchored at line start either, or a one-line guard body
#   (`let Ok(p) = std::env::var("X") else { return; };`) reads as a block with
#   no return in it at all.
#
#   `RMLX_SKIP_GPU` is out of it: `scripts/run_gpu_tests.sh` refuses to start
#   with that variable set, so a guard on it cannot stand a cell down in this
#   suite.
#
#   Its population is the DECLARING FILES of the classified GPU tests, not the
#   test fns: such a guard is routinely in a file-local helper, and a rule
#   scoped to test bodies reads a helper's silent return as a clean scan.
#
#   The notice's SHAPE is not defined here. `scripts/lib/skip_notice_patterns.sh`
#   holds it, and `scripts/run_gpu_tests.sh` reads the same file — a source gate
#   that accepted `SKIP  foo:` while the runner counted it as nameless would pass
#   CI and leave every run INCOMPLETE with a number and no name.
#
# Exit 0 = every notice in a classified GPU test names its own test, and every
# environment guard in those tests' files announces the stand-down it takes.
# Exit 1 = at least one does not; each is named with its file, line and reason.
# Exit 2 = the scan could not be trusted (no classification, no crates).

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=lib/skip_notice_patterns.sh
. "${ROOT}/scripts/lib/skip_notice_patterns.sh"
# shellcheck source=lib/awk_text.sh
. "${ROOT}/scripts/lib/awk_text.sh"
while [ $# -gt 0 ]; do
    case "$1" in
        --root) ROOT="${2:?--root needs a value}"; shift 2 ;;
        *) echo "ERROR: unknown argument '$1' (expected --root <dir>)." >&2; exit 2 ;;
    esac
done

# `--list-files` rather than `--list`: the second rule's population is the files
# the classified tests are declared in, and the first rule's is the same
# classification at two columns. One call, one population.
classification="$(bash "${ROOT}/scripts/check_gpu_tests_ignored.sh" --list-files --root "${ROOT}" 2>/dev/null)"
if [ -z "${classification}" ]; then
    echo "ERROR: check_gpu_tests_ignored.sh --list-files produced no GPU tests — the scan" >&2
    echo "would pass by having nothing to look at." >&2
    exit 2
fi
listing="$(printf '%s\n' "${classification}" | cut -f1,2)"
gpu_files="$(printf '%s\n' "${classification}" | cut -f3 | sort -u)"

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
    awk -v FILE="${f}" \
        -v ANY="${ANY_SKIP}" -v NAMED="${NAMED_SKIP}" -v PLACEHOLDER="${NAMED_SKIP_PLACEHOLDER}" '
        match($0, /^[[:space:]]*(pub[[:space:]]+(\([^)]*\)[[:space:]]*)?)?(async[[:space:]]+)?fn[[:space:]]+[A-Za-z_][A-Za-z0-9_]*/) {
            head = substr($0, RSTART, RLENGTH)
            sub(/^.*fn[[:space:]]+/, "", head)
            cur = head
        }
        # A stand-down notice is a quoted literal carrying SKIP as a whole word.
        # Both halves are required: the bare word appears in prose comments and
        # in variable names, and a quote alone is every other string in the file.
        $0 ~ /"/ && $0 ~ ANY {
            verdict = "none"
            if (match($0, PLACEHOLDER)) {
                verdict = "self"
            } else if (match($0, NAMED)) {
                named = substr($0, RSTART, RLENGTH)
                sub(/^SKIP /, "", named)
                sub(/:$/, "", named)
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

# The guards that stand a cell down and announce nothing. One record per guard
# block that returns with no notice in it, `<file>\t<line>\t<fn>\t<text>`.
#
# The block is followed by brace depth over the line's CODE — comments removed
# and string bodies blanked, so a literal carrying a brace does not close a
# guard early — while the notice and the guarded variable are read from the
# line as written, because both of those ARE literals.
silent=""
n_silent=0
while IFS=$'\t' read -r s_file s_line s_fn s_text; do
    [ -n "${s_file:-}" ] || continue
    rel="${s_file#"${ROOT}"/crates/}"
    silent="${silent}    ${rel}:${s_line} — ${s_fn} returns from an environment guard with no stand-down notice"$'\n'
    silent="${silent}        ${s_text}"$'\n'
    n_silent=$((n_silent + 1))
done <<< "$(printf '%s\n' "${gpu_files}" | while IFS= read -r f; do
    [ -n "${f}" ] || continue
    awk -v FILE="${f}" -v ANY="${ANY_SKIP}" "${AWK_TEXT_FNS}"'
        {
            raw = $0
            code = blank_strings(decomment(raw))
            bare = decomment(raw)
        }
        match(code, /^[[:space:]]*(pub[[:space:]]+(\([^)]*\)[[:space:]]*)?)?(async[[:space:]]+)?fn[[:space:]]+[A-Za-z_][A-Za-z0-9_]*/) {
            head = substr(code, RSTART, RLENGTH)
            sub(/^.*fn[[:space:]]+/, "", head)
            cur = head
        }
        # The process-wide GPU off switch is not a missing-model guard: the
        # runner refuses to start with it set, so a cell behind it never
        # stands down in that suite.
        !g && code ~ /env::var/ && bare !~ /RMLX_SKIP_GPU/ && index(code, "{") > 0 {
            g = 1; gl = NR; gfn = cur; gtext = raw
            sub(/^[[:space:]]+/, "", gtext)
            depth = 0; notice = 0; ret = 0
        }
        g {
            opens = gsub(/\{/, "{", code)
            closes = gsub(/\}/, "}", code)
            depth += opens - closes
            if (raw ~ /"/ && raw ~ ANY) notice = 1
            if (code ~ /(^|[^A-Za-z0-9_])return([[:space:]]*(;|$)|[[:space:]]+(None|Ok\(\(\)\))[[:space:]]*;)/) ret = 1
            if (depth <= 0) {
                if (ret && !notice)
                    printf "%s\t%d\t%s\t%s\n", FILE, gl, gfn, gtext
                g = 0
            }
        }
    ' "${f}"
done)"

# Both kinds are printed before the single exit at the end. They co-occur — a
# suite half-converted has one of each — and a gate that exits inside the first
# block sends the reader back for the second one run later.
red=0

if [ "${n_silent}" -gt 0 ]; then
    echo "ERROR: ${n_silent} environment guard(s) in classified GPU tests' files stand a" >&2
    echo "       cell down and announce nothing:" >&2
    printf '%s' "${silent}" >&2
    echo >&2
    echo "libtest reports such a cell as \`ok\` and scripts/run_gpu_tests.sh never sees" >&2
    echo "it, so a cell that could not run is counted as one that passed. Announce it:" >&2
    echo "  SKIP <the test fn>: <why>" >&2
    echo "A helper takes the caller's test name as an argument and prints" >&2
    echo "\`SKIP {test}: <why>\`; it cannot name itself, because no libtest filter" >&2
    echo "reaches a helper. See docs/GPU_TESTS.md." >&2
    red=1
fi

if [ "${n}" -gt 0 ]; then
    echo "ERROR: ${n} stand-down notice(s) in classified GPU tests do not name their test:" >&2
    printf '%s' "${violations}" >&2
    echo >&2
    echo "scripts/run_gpu_tests.sh counts such a notice and lists it nowhere, so the run" >&2
    echo "is INCOMPLETE with a number and no name. Write it as" >&2
    echo "  SKIP <this test fn>: <why>" >&2
    echo "See docs/GPU_TESTS.md." >&2
    red=1
fi

if [ "${red}" = "1" ]; then
    exit 1
fi

echo "OK: every stand-down notice in a classified GPU test names its own test, and
every environment guard in those files announces the stand-down it takes."
