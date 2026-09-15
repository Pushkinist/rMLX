#!/usr/bin/env bash
# scripts/check_named_skip_notices_fixtures.sh — recall test for
# `check_named_skip_notices.sh`.
#
# WHY
#   The gate it tests is a text scan, and a text scan that stops matching says
#   exactly what a clean tree says. Every case below plants ONE shape and asserts
#   the REASON that reaches the report, not only the exit code: "names no test"
#   and "names X, not itself" are different defects and a gate that reports
#   either for both sends the reader to the wrong line.
#
#   The whitespace cases are the ones that motivated the shared pattern file. The
#   runner's notice carries exactly one space and no space before the colon; a
#   source gate that accepted a wider shape would pass a line the runner then
#   counts as nameless, which is a green CI and an INCOMPLETE run for ever.
#
# Every case builds its own root and passes it with --root. No environment
# variable selects anything here.
#
# Exit 0 = every case reached its expected exit code and reason.
# Exit 1 = at least one did not; each failure is named.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

failures=0
CASE=""
CASE_ROOT=""
OUT=""
STATUS=0

fail() {
    echo "FAIL [${CASE}]: $1" >&2
    failures=$((failures + 1))
}

# A root carrying the gate, the shared pattern file and a stub classifier. The
# pattern file is SYMLINKED rather than copied: a fixture with its own copy would
# keep passing after the real shape changed, which is the drift the one-producer
# rule exists to stop.
new_case() {
    CASE="$1"
    CASE_ROOT="${WORK}/$1"
    mkdir -p "${CASE_ROOT}/scripts/lib" "${CASE_ROOT}/crates"
    cp "${REPO_ROOT}/scripts/check_named_skip_notices.sh" "${CASE_ROOT}/scripts/"
    ln -s "${REPO_ROOT}/scripts/lib/skip_notice_patterns.sh" \
        "${CASE_ROOT}/scripts/lib/skip_notice_patterns.sh"
    : >"${CASE_ROOT}/listing"
    cat >"${CASE_ROOT}/scripts/check_gpu_tests_ignored.sh" <<'STUB'
#!/usr/bin/env bash
# Stub classifier: prints the canned listing beside it.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cat "${root}/listing"
STUB
}

# `classify <crate> <test>...` — what the stub classifier will name.
classify() {
    local crate="$1"; shift
    local t
    for t in "$@"; do printf '%s\t%s\n' "${crate}" "${t}" >>"${CASE_ROOT}/listing"; done
}

# `source_file <crate> <relative path>` — body on stdin.
source_file() {
    local crate="$1" rel="$2"
    mkdir -p "${CASE_ROOT}/crates/${crate}/$(dirname "${rel}")"
    cat >"${CASE_ROOT}/crates/${crate}/${rel}"
}

run_case() {
    OUT="$(bash "${CASE_ROOT}/scripts/check_named_skip_notices.sh" --root "${CASE_ROOT}" 2>&1)"
    STATUS=$?
}

expect_status() {
    [ "${STATUS}" = "$1" ] || fail "expected exit $1, got ${STATUS}: ${OUT}"
}

expect_out() {
    case "${OUT}" in *"$1"*) ;; *) fail "expected output to carry '$1', got: ${OUT}" ;; esac
}

expect_no_out() {
    case "${OUT}" in *"$1"*) fail "expected output NOT to carry '$1', got: ${OUT}" ;; esac
}

# ---------------------------------------------------------------------------
# The clean shape. Everything below is one edit away from this, so a case that
# fails for a reason this one shares is a broken fixture rather than a finding.
new_case named_notice_passes
classify rmlx-models gpu_alpha
source_file rmlx-models src/alpha_tests.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    if no_model() {
        eprintln!("SKIP gpu_alpha: RMLX_O_MODELS_ROOT does not hold the snapshot");
        return;
    }
}
RS
run_case
expect_status 0
expect_out "every stand-down notice"

# ---------------------------------------------------------------------------
# The thirteen sites' own shape: the notice names nothing, so the runner counts
# it and lists it nowhere.
new_case unnamed_notice_fails
classify rmlx-models gpu_alpha
source_file rmlx-models src/alpha_tests.rs <<'RS'
#[test]
fn gpu_alpha() {
    eprintln!("SKIP: RMLX_TEST_MODEL_QWEN36 not set");
}
RS
run_case
expect_status 1
expect_out "gpu_alpha names no test"
expect_out "src/alpha_tests.rs:3"

# ---------------------------------------------------------------------------
# Worse than nameless: the runner WOULD list this, under a name that ran. The
# reason has to separate the two or the reader chases the wrong cell.
new_case notice_naming_another_test_fails
classify rmlx-models gpu_alpha gpu_beta
source_file rmlx-models src/alpha_tests.rs <<'RS'
#[test]
fn gpu_alpha() {
    eprintln!("SKIP gpu_beta: no snapshot");
}
RS
run_case
expect_status 1
expect_out "gpu_alpha names gpu_beta, not itself"
expect_no_out "names no test"

# ---------------------------------------------------------------------------
# A helper's notice already carries the caller's name at run time (the harness's
# `SKIP {test}:` shape), and the helper is not a cell the runner selects. Fail on
# it and the gate is red on every correctly-written suite in the tree.
new_case notice_in_a_helper_is_dropped
classify rmlx-models gpu_alpha
source_file rmlx-models src/alpha_tests.rs <<'RS'
fn resolve(test: &str) -> Option<()> {
    eprintln!("SKIP {test}: no snapshot");
    None
}

#[test]
fn gpu_alpha() {
    let _ = resolve("gpu_alpha");
}
RS
run_case
expect_status 0

# ---------------------------------------------------------------------------
# An empty classification is not a clean tree: the scan would have nothing to
# look at and would report success having read no test at all.
new_case empty_listing_is_exit_2
source_file rmlx-models src/alpha_tests.rs <<'RS'
#[test]
fn gpu_alpha() {
    eprintln!("SKIP: nothing");
}
RS
run_case
expect_status 2
expect_out "produced no GPU tests"

# ---------------------------------------------------------------------------
# No test-bearing sources is the same failure one layer down: a find that stopped
# matching reports every notice as absent.
new_case no_sources_is_exit_2
classify rmlx-models gpu_alpha
run_case
expect_status 2
expect_out "no test-bearing sources"

# ---------------------------------------------------------------------------
# Two spaces. The runner's pattern carries exactly one, so it reads this notice
# as nameless; a gate that called it attributed would pass CI and leave the run
# INCOMPLETE with a number and no name.
new_case double_space_is_not_a_name
classify rmlx-models gpu_alpha
source_file rmlx-models src/alpha_tests.rs <<'RS'
#[test]
fn gpu_alpha() {
    eprintln!("SKIP  gpu_alpha: no snapshot");
}
RS
run_case
expect_status 1
expect_out "gpu_alpha names no test"

# ---------------------------------------------------------------------------
# A space before the colon, for the same reason and from the other side.
new_case space_before_colon_is_not_a_name
classify rmlx-models gpu_alpha
source_file rmlx-models src/alpha_tests.rs <<'RS'
#[test]
fn gpu_alpha() {
    eprintln!("SKIP gpu_alpha : no snapshot");
}
RS
run_case
expect_status 1
expect_out "gpu_alpha names no test"

# ---------------------------------------------------------------------------
# The placeholder a test that passes its own name uses. It expands to the
# runner's shape, so it is accepted — by its exact text.
new_case test_placeholder_is_accepted
classify rmlx-models gpu_alpha
source_file rmlx-models src/alpha_tests.rs <<'RS'
#[test]
fn gpu_alpha() {
    let test = "gpu_alpha";
    eprintln!("SKIP {test}: no snapshot");
}
RS
run_case
expect_status 0

# ---------------------------------------------------------------------------
# Any other placeholder reads identically and can hold the name of a cell that
# ran. The gate cannot check what is inside it, so it does not accept it.
new_case other_placeholder_is_refused
classify rmlx-models gpu_alpha
source_file rmlx-models src/alpha_tests.rs <<'RS'
#[test]
fn gpu_alpha() {
    eprintln!("SKIP {other}: no snapshot");
}
RS
run_case
expect_status 1
expect_out "gpu_alpha names no test"

# ---------------------------------------------------------------------------
# A notice in a fn the classifier does not name is out of scope: the runner never
# selects it, so it never reaches a report.
new_case unclassified_test_is_dropped
classify rmlx-models gpu_beta
source_file rmlx-models src/alpha_tests.rs <<'RS'
#[test]
fn gpu_alpha() {
    eprintln!("SKIP: no snapshot");
}
RS
run_case
expect_status 0

# ---------------------------------------------------------------------------
# Prose. The word appears in comments and in variable names throughout the tree,
# and a gate that fired on those would be turned off within a week.
new_case prose_mentioning_skip_is_not_a_notice
classify rmlx-models gpu_alpha
source_file rmlx-models src/alpha_tests.rs <<'RS'
#[test]
fn gpu_alpha() {
    // This cell used to SKIP when the snapshot was absent; it no longer does.
    let skip_reason_unused = 1;
    let _ = skip_reason_unused;
}
RS
run_case
expect_status 0

# ---------------------------------------------------------------------------
# The crate is read from the path, so a notice under one crate must not be
# attributed to a same-named test classified under another.
new_case crate_is_read_from_the_path
classify rmlx-models gpu_alpha
source_file rmlx-audio src/alpha_tests.rs <<'RS'
#[test]
fn gpu_alpha() {
    eprintln!("SKIP: no snapshot");
}
RS
run_case
expect_status 0

if [ "${failures}" -gt 0 ]; then
    echo "check_named_skip_notices_fixtures: ${failures} case(s) failed" >&2
    exit 1
fi
echo "check_named_skip_notices_fixtures: OK — every planted shape reached its own reason."
