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
# Stub classifier: prints the canned listing beside it, at the width the caller
# asked for — the same two-from-three relationship the real one has.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ "${1:-}" = "--list-files" ]; then cat "${root}/listing"; else cut -f1,2 "${root}/listing"; fi
STUB
    ln -s "${REPO_ROOT}/scripts/lib/awk_text.sh" "${CASE_ROOT}/scripts/lib/awk_text.sh"
}

# `classify <crate> <declaring file, relative to the crate> <test>...` — what
# the stub classifier will name. The file is an argument because the gate's
# second rule is scoped by it: a case that could not state where its test is
# declared could not state which files that rule reads.
classify() {
    local crate="$1" rel="$2"; shift 2
    local t
    for t in "$@"; do
        printf '%s\t%s\t%s\n' "${crate}" "${t}" \
            "${CASE_ROOT}/crates/${crate}/${rel}" >>"${CASE_ROOT}/listing"
    done
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
classify rmlx-models src/alpha_tests.rs gpu_alpha
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
classify rmlx-models src/alpha_tests.rs gpu_alpha
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
classify rmlx-models src/alpha_tests.rs gpu_alpha gpu_beta
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
classify rmlx-models src/alpha_tests.rs gpu_alpha
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
classify rmlx-models src/alpha_tests.rs gpu_alpha
run_case
expect_status 2
expect_out "no test-bearing sources"

# ---------------------------------------------------------------------------
# Two spaces. The runner's pattern carries exactly one, so it reads this notice
# as nameless; a gate that called it attributed would pass CI and leave the run
# INCOMPLETE with a number and no name.
new_case double_space_is_not_a_name
classify rmlx-models src/alpha_tests.rs gpu_alpha
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
classify rmlx-models src/alpha_tests.rs gpu_alpha
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
classify rmlx-models src/alpha_tests.rs gpu_alpha
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
classify rmlx-models src/alpha_tests.rs gpu_alpha
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
classify rmlx-models src/alpha_tests.rs gpu_beta
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
classify rmlx-models src/alpha_tests.rs gpu_alpha
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
classify rmlx-models src/alpha_tests.rs gpu_alpha
source_file rmlx-audio src/alpha_tests.rs <<'RS'
#[test]
fn gpu_alpha() {
    eprintln!("SKIP: no snapshot");
}
RS
run_case
expect_status 0

# ---------------------------------------------------------------------------
# The second rule. A guard that reads an environment variable and returns
# without announcing anything is the shape the first rule cannot see at all:
# there is no notice to read, libtest prints `ok`, and the cell is counted as a
# pass by every gate in the tree.
new_case silent_env_guard_fails
classify rmlx-models src/alpha_tests.rs gpu_alpha
source_file rmlx-models src/alpha_tests.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    let Ok(p) = std::env::var("RMLX_KV_TEST_MODEL") else {
        eprintln!("RMLX_KV_TEST_MODEL not set — skipping");
        return;
    };
    let _ = p;
}
RS
run_case
expect_status 1
expect_out "gpu_alpha returns from an environment guard with no stand-down notice"

# ---------------------------------------------------------------------------
# The same guard, announced. Nothing else changes, so a gate that fired here
# would be firing on the guard rather than on its silence.
new_case announced_env_guard_passes
classify rmlx-models src/alpha_tests.rs gpu_alpha
source_file rmlx-models src/alpha_tests.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    let Ok(p) = std::env::var("RMLX_KV_TEST_MODEL") else {
        eprintln!("SKIP gpu_alpha: RMLX_KV_TEST_MODEL not set");
        return;
    };
    let _ = p;
}
RS
run_case
expect_status 0

# ---------------------------------------------------------------------------
# The guard is routinely in a file-local helper, which is why this rule is
# scoped to the FILE and not to the test fns: a rule that read test bodies only
# would call this a clean scan.
new_case silent_env_guard_in_a_helper_fails
classify rmlx-models src/alpha_tests.rs gpu_alpha
source_file rmlx-models src/alpha_tests.rs <<'RS'
fn model_path() -> Option<String> {
    let Ok(p) = std::env::var("RMLX_KV_TEST_MODEL") else {
        eprintln!("RMLX_KV_TEST_MODEL not set — skipping");
        return None;
    };
    Some(p)
}

#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    let Some(p) = model_path() else {
        return;
    };
    let _ = p;
}
RS
run_case
expect_status 1
expect_out "model_path returns from an environment guard with no stand-down notice"

# ---------------------------------------------------------------------------
# The process-wide GPU off switch is not a missing-model guard: the runner
# refuses to start with it set, so a cell behind it never stands down in that
# suite. Every classified test opens with one, so a rule that fired here would
# be red on the whole tree.
new_case skip_gpu_guard_is_not_a_stand_down
classify rmlx-models src/alpha_tests.rs gpu_alpha
source_file rmlx-models src/alpha_tests.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    if std::env::var("RMLX_SKIP_GPU").as_deref() == Ok("1") {
        return;
    }
}
RS
run_case
expect_status 0

# ---------------------------------------------------------------------------
# A guard in a file that declares no classified GPU test is out of scope, the
# same boundary the first rule has: the runner never selects such a cell, so its
# silence never reaches a report.
new_case silent_guard_in_an_unclassified_file_is_dropped
classify rmlx-models src/alpha_tests.rs gpu_alpha
source_file rmlx-models src/alpha_tests.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    let _ = 1;
}
RS
source_file rmlx-models src/beta_tests.rs <<'RS'
#[test]
fn cpu_beta() {
    let Ok(p) = std::env::var("RMLX_KV_TEST_MODEL") else {
        eprintln!("not set — skipping");
        return;
    };
    let _ = p;
}
RS
run_case
expect_status 0

# ---------------------------------------------------------------------------
# The guard block is followed by brace depth over the line's CODE, so a brace
# inside a string literal does not close it early. Here the literal's `}` would
# end the block one line before the silent return, and the defect would read as
# a clean scan.
new_case a_brace_in_a_literal_does_not_close_the_guard
classify rmlx-models src/alpha_tests.rs gpu_alpha
source_file rmlx-models src/alpha_tests.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    let Ok(p) = std::env::var("RMLX_KV_TEST_MODEL") else {
        eprintln!("the model root is spelled }} on this host");
        return;
    };
    let _ = p;
}
RS
run_case
expect_status 1
expect_out "gpu_alpha returns from an environment guard with no stand-down notice"

# ---------------------------------------------------------------------------
# A guard whose exits all carry a value has answered, not skipped. The
# two-env-key resolvers in this tree are exactly this shape, and a rule that
# read any `return` would be red on correct code the day one of them is
# classified.
new_case value_carrying_return_is_not_a_stand_down
classify rmlx-models src/alpha_tests.rs gpu_alpha
source_file rmlx-models src/alpha_tests.rs <<'RS'
fn chunk() -> usize {
    if let Ok(v) = std::env::var("RMLX_CHUNK") {
        return v.parse().unwrap_or(4096);
    }
    4096
}

#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    let _ = chunk();
}
RS
run_case
expect_status 0

# ---------------------------------------------------------------------------
# The guard body on the opening line. A needle anchored at line start reads
# this as a block with no return in it, which is the whole defect passing as a
# clean scan.
new_case one_line_if_guard_is_seen
classify rmlx-models src/alpha_tests.rs gpu_alpha
source_file rmlx-models src/alpha_tests.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    if std::env::var("RMLX_KV_TEST_MODEL").is_err() { return; }
}
RS
run_case
expect_status 1
expect_out "gpu_alpha returns from an environment guard with no stand-down notice"

# ---------------------------------------------------------------------------
# The other one-line spelling, and the commoner one.
new_case one_line_let_else_guard_is_seen
classify rmlx-models src/alpha_tests.rs gpu_alpha
source_file rmlx-models src/alpha_tests.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    let Ok(p) = std::env::var("RMLX_KV_TEST_MODEL") else { return; };
    let _ = p;
}
RS
run_case
expect_status 1
expect_out "gpu_alpha returns from an environment guard with no stand-down notice"

# ---------------------------------------------------------------------------
# Both defects in one tree. The two blocks are independent and co-occur in a
# half-converted suite; a gate that exits inside the first sends the reader
# back for the second one run later.
new_case both_kinds_are_reported
classify rmlx-models src/alpha_tests.rs gpu_alpha gpu_beta
source_file rmlx-models src/alpha_tests.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    let Ok(p) = std::env::var("RMLX_KV_TEST_MODEL") else {
        return;
    };
    let _ = p;
}

#[test]
#[ignore = "GPU Metal"]
fn gpu_beta() {
    eprintln!("SKIP: no snapshot");
}
RS
run_case
expect_status 1
expect_out "gpu_alpha returns from an environment guard with no stand-down notice"
expect_out "gpu_beta names no test"

if [ "${failures}" -gt 0 ]; then
    echo "check_named_skip_notices_fixtures: ${failures} case(s) failed" >&2
    exit 1
fi
echo "check_named_skip_notices_fixtures: OK — every planted shape reached its own reason."
