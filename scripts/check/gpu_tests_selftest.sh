#!/usr/bin/env bash
# Recall test for check/gpu_tests.sh skip-notices and list: planted shapes, asserting the reason.
#
# Recall test for: check/gpu_tests.sh skip-notices list
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
# Most cases state the classified tests in a listing file (`--listing`), so a case
# isolates the notice rules from the classifier and plants no Device::Gpu. The cases
# at the end classify planted workspaces for real, which is the path the Makefile
# takes, and hold `list` to its columns and its refusals.
#
# Every case builds its own root and passes it with --root. No environment
# variable selects anything here.
#
# LOC-exempt: one planted source per case; the cases are the spec
# Usage:  bash scripts/check/gpu_tests_selftest.sh
# Exit:   0 every case held; 1 a case did not, or none ran; 2 a root or a case could not be built

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TOOL="$REPO_ROOT/scripts/check/gpu_tests.sh"
# shellcheck source=scripts/lib/selftest_support.sh
. "$REPO_ROOT/scripts/lib/selftest_support.sh"

st_init gpu-tests-selftest

CASE=""
CASE_ROOT=""

# new_case <name>: a root with a crates directory and an empty listing of classified tests.
new_case() {
    CASE="$1"
    CASE_ROOT="$ST_WORK/$1"
    mkdir -p "$CASE_ROOT/crates"
    : >"$CASE_ROOT/listing"
}

# classify <crate> <declaring file, relative to the crate> <test>...: what the listing
# names. The file is an argument because the gate's second rule is scoped by it: a case
# that could not state where its test is declared could not state which files that
# rule reads.
classify() {
    local crate="$1" rel="$2" t
    shift 2
    for t in "$@"; do
        printf '%s\t%s\t%s\n' "$crate" "$t" "$CASE_ROOT/crates/$crate/$rel" >>"$CASE_ROOT/listing"
    done
}

# source_file <crate> <relative path>: the body on stdin.
source_file() {
    local crate="$1" rel="$2"
    mkdir -p "$CASE_ROOT/crates/$crate/$(dirname "$rel")"
    cat >"$CASE_ROOT/crates/$crate/$rel"
}

# check_case <exit> [--has S | --lacks S]...: run skip-notices over the case root and its
# listing, and hold the run to the exit code and to its lines.
check_case() {
    local want="$1"
    shift
    st_run bash "$TOOL" skip-notices --root "$CASE_ROOT" --listing "$CASE_ROOT/listing"
    st_verdict "$CASE" "$(st_check "$want" "$@")"
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
check_case 0 \
    --has "every stand-down notice"

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
check_case 1 \
    --has "gpu_alpha names no test" \
    --has "src/alpha_tests.rs:3"

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
check_case 1 \
    --has "gpu_alpha names gpu_beta, not itself" \
    --lacks "names no test"

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
check_case 0

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
check_case 2 \
    --has "produced no GPU tests"

# ---------------------------------------------------------------------------
# No test-bearing sources is the same failure one layer down: a find that stopped
# matching reports every notice as absent.
new_case no_sources_is_exit_2
classify rmlx-models src/alpha_tests.rs gpu_alpha
check_case 2 \
    --has "no test-bearing sources"

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
check_case 1 \
    --has "gpu_alpha names no test"

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
check_case 1 \
    --has "gpu_alpha names no test"

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
check_case 0

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
check_case 1 \
    --has "gpu_alpha names no test"

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
check_case 0

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
check_case 0

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
check_case 0

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
check_case 1 \
    --has "gpu_alpha returns from an environment guard with no stand-down notice"

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
check_case 0

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
check_case 1 \
    --has "model_path returns from an environment guard with no stand-down notice"

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
check_case 0

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
check_case 0

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
check_case 1 \
    --has "gpu_alpha returns from an environment guard with no stand-down notice"

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
check_case 0

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
check_case 1 \
    --has "gpu_alpha returns from an environment guard with no stand-down notice"

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
check_case 1 \
    --has "gpu_alpha returns from an environment guard with no stand-down notice"

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
check_case 1 \
    --has "gpu_alpha returns from an environment guard with no stand-down notice" \
    --has "gpu_beta names no test"

# ---------------------------------------------------------------------------
# The variable is read on one line and the block opens two lines later. A guard
# rule that wants both on one line reads this as no guard at all.
new_case silent_multi_line_let_else_guard_is_seen
classify rmlx-models src/alpha_tests.rs gpu_alpha
source_file rmlx-models src/alpha_tests.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    let Some(p) =
        std::env::var_os("RMLX_TEST_MODEL_ALPHA").map(std::path::PathBuf::from)
    else {
        return;
    };
    let _ = p;
}
RS
check_case 1 \
    --has "src/alpha_tests.rs:5 — gpu_alpha returns from an environment guard with no stand-down notice"

# A variable read into a binding, with no block, is not a guard.
new_case an_env_read_with_no_block_is_not_a_guard
classify rmlx-models src/alpha_tests.rs gpu_alpha
source_file rmlx-models src/alpha_tests.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    let strict =
        std::env::var("RMLX_ALPHA_STRICT")
            .is_ok();
    if strict && done() {
        return;
    }
}
RS
check_case 0

# ---------------------------------------------------------------------------
# The stand-down notice has one form. A test that announces in another form
# prints a line the runner reads as prose, so the cell is counted as passed.
new_case a_stand_down_in_another_form_fails
classify rmlx-models src/alpha_tests.rs gpu_alpha
source_file rmlx-models src/alpha_tests.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    let Some(p) =
        std::env::var_os("RMLX_TEST_MODEL_ALPHA").map(std::path::PathBuf::from)
    else {
        eprintln!("gpu_alpha: skipping: RMLX_TEST_MODEL_ALPHA not set");
        return;
    };
    let _ = p;
}
RS
check_case 1 \
    --has "src/alpha_tests.rs:8 — gpu_alpha prints and returns with no value, and the block holds no stand-down notice" \
    --has 'eprintln!("gpu_alpha: skipping: RMLX_TEST_MODEL_ALPHA not set");' \
    --has "returns from an environment guard with no stand-down notice"

# The rule does not read what guards the block. A missing file and a counter
# that did not move are the same stand-down.
new_case a_stand_down_in_another_form_behind_any_guard_fails
classify rmlx-kv-quant tests/alpha.rs gpu_alpha
source_file rmlx-kv-quant tests/alpha.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    let delta = run();
    if delta == 0 {
        eprintln!(
            "HOLD-soft: the dispatch did not fire, skipping the comparison"
        );
        return;
    }
}
RS
check_case 1 \
    --has "tests/alpha.rs:9 — gpu_alpha prints and returns with no value, and the block holds no stand-down notice" \
    --lacks "environment guard"

# The same block with the notice in its one form passes.
new_case a_printed_return_with_the_notice_passes
classify rmlx-kv-quant tests/alpha.rs gpu_alpha
source_file rmlx-kv-quant tests/alpha.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    let delta = run();
    if delta == 0 {
        eprintln!(
            "SKIP gpu_alpha: the dispatch did not fire"
        );
        return;
    }
}
RS
check_case 0

# A print in an outer block does not make an inner silent return an
# announcement, and a block that returns after a helper announced prints nothing.
new_case a_silent_return_after_an_outer_print_is_not_read
classify rmlx-kv-quant tests/alpha.rs gpu_alpha
source_file rmlx-kv-quant tests/alpha.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    for cell in cells() {
        eprintln!("cell {cell}");
        let Some(loaded) = load("gpu_alpha") else {
            return;
        };
        let _ = loaded;
    }
}
RS
check_case 0

# The population is every fn of a file that declares a classified test. A
# helper that serves several tests stands each of them down.
new_case a_printed_return_in_a_helper_fails
classify rmlx-kv-quant tests/alpha.rs gpu_alpha
source_file rmlx-kv-quant tests/alpha.rs <<'RS'
fn parity(name: &str) {
    if nothing() {
        eprintln!("{name}: HOLD-soft, the counter did not move; check skipped");
        return;
    }
}

#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    parity("alpha");
}
RS
check_case 1 \
    --has "tests/alpha.rs:4 — parity prints and returns with no value, and the block holds no stand-down notice"

# The same helper with the caller's name in the notice passes.
new_case a_helper_that_names_its_caller_passes
classify rmlx-kv-quant tests/alpha.rs gpu_alpha
source_file rmlx-kv-quant tests/alpha.rs <<'RS'
fn parity(test: &str, name: &str) {
    if nothing() {
        eprintln!("SKIP {test}: {name} HOLD-soft, the counter did not move");
        return;
    }
}

#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    parity("gpu_alpha", "alpha");
}
RS
check_case 0

# A file that declares no classified test is not read.
new_case a_printed_return_in_a_file_with_no_classified_test_is_dropped
classify rmlx-kv-quant tests/alpha.rs gpu_alpha
source_file rmlx-kv-quant tests/alpha.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    work();
}
RS
source_file rmlx-kv-quant tests/beta.rs <<'RS'
#[test]
fn cpu_beta() {
    if nothing() {
        eprintln!("nothing to do, skipping");
        return;
    }
}
RS
check_case 0

# Every print macro of the standard library is read, not `eprintln!` alone.
for print_macro in 'println!' 'print!' 'eprint!'; do
    new_case "a_stand_down_printed_with_${print_macro%!}_fails"
    classify rmlx-kv-quant tests/alpha.rs gpu_alpha
    source_file rmlx-kv-quant tests/alpha.rs <<RS
#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    if nothing() {
        ${print_macro}("gpu_alpha: skipping, the dispatch did not fire");
        return;
    }
}
RS
    check_case 1 \
        --has "tests/alpha.rs:6 — gpu_alpha prints and returns with no value, and the block holds no stand-down notice"
done

# Every return with no value is read, as in the guard rule.
new_case a_stand_down_that_returns_ok_fails
classify rmlx-kv-quant tests/alpha.rs gpu_alpha
source_file rmlx-kv-quant tests/alpha.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() -> anyhow::Result<()> {
    if nothing() {
        eprintln!("gpu_alpha: the dispatch did not fire, skipping");
        return Ok(());
    }
    Ok(())
}
RS
check_case 1 \
    --has "tests/alpha.rs:6 — gpu_alpha prints and returns with no value, and the block holds no stand-down notice"

new_case a_stand_down_that_returns_none_fails
classify rmlx-kv-quant tests/alpha.rs gpu_alpha
source_file rmlx-kv-quant tests/alpha.rs <<'RS'
fn loaded(name: &str) -> Option<Model> {
    if nothing() {
        eprintln!("{name}: no snapshot, skipping");
        return None;
    }
    Some(load())
}

#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    let _ = loaded("alpha");
}
RS
check_case 1 \
    --has "tests/alpha.rs:4 — loaded prints and returns with no value, and the block holds no stand-down notice"

new_case a_stand_down_that_ends_in_a_bare_return_fails
classify rmlx-kv-quant tests/alpha.rs gpu_alpha
source_file rmlx-kv-quant tests/alpha.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    let model = match load() {
        Some(model) => model,
        None => {
            eprintln!("gpu_alpha: no snapshot, skipping");
            return
        }
    };
    let _ = model;
}
RS
check_case 1 \
    --has "tests/alpha.rs:8 — gpu_alpha prints and returns with no value, and the block holds no stand-down notice"

# A return that carries a value is a result, not a stand-down.
new_case a_printed_return_with_a_value_is_not_a_stand_down
classify rmlx-kv-quant tests/alpha.rs gpu_alpha
source_file rmlx-kv-quant tests/alpha.rs <<'RS'
fn width() -> usize {
    if narrow() {
        eprintln!("narrow terminal");
        return 80;
    }
    120
}

#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    let _ = width();
}
RS
check_case 0

# A notice excuses the block it is in and no other. Here the first block
# announces in the one form and a later block of the same fn does not: a suite
# converted by half.
new_case a_notice_in_an_earlier_block_does_not_excuse_a_later_one
classify rmlx-kv-quant tests/alpha.rs gpu_alpha
source_file rmlx-kv-quant tests/alpha.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    if no_model() {
        eprintln!("SKIP gpu_alpha: no snapshot");
        return;
    }
    if no_dispatch() {
        eprintln!("gpu_alpha: the dispatch did not fire, skipping");
        return;
    }
}
RS
check_case 1 \
    --has "tests/alpha.rs:10 — gpu_alpha prints and returns with no value, and the block holds no stand-down notice" \
    --lacks "tests/alpha.rs:6"

# A print belongs to the block it is in. A block that prints and goes on does
# not make a later block's silent return an announcement.
new_case a_print_in_an_earlier_block_does_not_convict_a_later_one
classify rmlx-kv-quant tests/alpha.rs gpu_alpha
source_file rmlx-kv-quant tests/alpha.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    if verbose() {
        eprintln!("gpu_alpha: starting");
    }
    let Some(loaded) = load("gpu_alpha") else {
        return;
    };
    let _ = loaded;
}
RS
check_case 0

# The depth of the scan starts again at every fn. Here the first fn leaves the
# scan one block deep: neither text reader knows a character literal.
new_case the_block_depth_starts_again_at_every_fn
classify rmlx-kv-quant tests/alpha.rs gpu_alpha
source_file rmlx-kv-quant tests/alpha.rs <<'RS'
fn opener() -> char {
    '{'
}

#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    eprintln!("gpu_alpha: done");
    return;
}
RS
check_case 0

# The report names the line of the return.
new_case the_other_form_is_reported_at_the_return
classify rmlx-kv-quant tests/alpha.rs gpu_alpha
source_file rmlx-kv-quant tests/alpha.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    if nothing() {
        eprintln!("gpu_alpha: skipping");
        let _ = cleanup();
        return;
    }
}
RS
check_case 1 \
    --has "tests/alpha.rs:7 — gpu_alpha prints"

# A pending environment read opens its guard at the next block, whatever opens
# that block: an `if` whose condition runs over two lines has no `else`.
new_case silent_multi_line_if_guard_is_seen
classify rmlx-models src/alpha_tests.rs gpu_alpha
source_file rmlx-models src/alpha_tests.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    if std::env::var("RMLX_TEST_MODEL_ALPHA")
        .is_err()
    {
        return;
    }
}
RS
check_case 1 \
    --has "src/alpha_tests.rs:4 — gpu_alpha returns from an environment guard with no stand-down notice"

# ---------------------------------------------------------------------------
# A waiver, `WAIVED <test>: <why>`, is held to the same name rule over every fn
# of a classified test's file: the caller's name through the exact `{test}`
# placeholder, or the enclosing fn's own name.
waiver_case() {
    new_case "$1"
    classify rmlx-models src/alpha_tests.rs gpu_alpha
    source_file rmlx-models src/alpha_tests.rs <<RS
#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    check("gpu_alpha");
}

fn check(test: &str) {
    eprintln!("$2");
}
RS
}

waiver_case waiver_with_placeholder_passes 'WAIVED {test}: the streams were not compared'
check_case 0

waiver_case waiver_bare 'WAIVED: the streams were not compared'
check_case 1 \
    --has "src/alpha_tests.rs:8 — rmlx-models check: the waiver names no test"

waiver_case waiver_wrong_placeholder 'WAIVED {name}: the streams were not compared'
check_case 1 \
    --has "src/alpha_tests.rs:8 — rmlx-models check: the waiver names no test"

waiver_case waiver_two_spaces 'WAIVED  check: the streams were not compared'
check_case 1 \
    --has "src/alpha_tests.rs:8 — rmlx-models check: the waiver names no test"

waiver_case waiver_names_another 'WAIVED gpu_beta: the streams were not compared'
check_case 1 \
    --has "src/alpha_tests.rs:8 — rmlx-models check: the waiver names gpu_beta, not itself"

# A root with no crates directory at all is the same refusal as one with no test
# source, and find's own complaint stays out of the report.
new_case no_crates_directory_is_exit_2
classify rmlx-models src/alpha_tests.rs gpu_alpha
rmdir "$CASE_ROOT/crates"
check_case 2 \
    --has "no test-bearing sources" \
    --lacks "No such file"

# KNOWN OPEN BLIND SPOT, pinned so that changing it is a decision: a source awk cannot open
# reads as a source with no findings, so the nameless notice in it passes. Closing the hole
# turns this case red, and the case then takes the exit the closed hole gives.
new_case an_unreadable_source_is_a_known_open_hole
classify rmlx-models src/alpha_tests.rs gpu_alpha
source_file rmlx-models src/alpha_tests.rs <<'RS'
#[test]
fn gpu_alpha() {
    eprintln!("SKIP: no snapshot");
}
RS
chmod 000 "$CASE_ROOT/crates/rmlx-models/src/alpha_tests.rs"
if [ -r "$CASE_ROOT/crates/rmlx-models/src/alpha_tests.rs" ]; then
    printf '  skip %s - %s\n' "$CASE" "this user can read a mode-000 file"
else
    check_case 0 \
        --has "every stand-down notice"
fi

# ---------------------------------------------------------------------------
# The cases above state the classification in a listing. These classify a planted
# workspace for real, which is how the Makefile runs the gate.

# workspace <name>: a one-member workspace, so the classifier reads the planted tests itself.
workspace() {
    CASE="$1"
    CASE_ROOT="$ST_WORK/$1"
    mkdir -p "$CASE_ROOT/crates/fx/src"
    printf '[workspace]\nmembers = [\n    "crates/fx",\n]\n' >"$CASE_ROOT/Cargo.toml"
    printf '[package]\nname = "fx"\nversion = "0.0.0"\nedition = "2021"\n' >"$CASE_ROOT/crates/fx/Cargo.toml"
}

# gpu_test <statement>: one classified test, a GPU test that carries #[ignore], whose body ends in the statement.
gpu_test() {
    source_file fx src/fixture_tests.rs <<RS
#[test]
#[ignore = "GPU Metal context"]
fn gpu_alpha() {
    let device = Device::Gpu;
    $1
}
RS
}

# derived_case <exit> [--has S | --lacks S]...: skip-notices over the workspace, classifying it.
derived_case() {
    local want="$1"
    shift
    st_run bash "$TOOL" skip-notices --root "$CASE_ROOT"
    st_verdict "$CASE" "$(st_check "$want" "$@")"
}

workspace derived_nameless_notice_fails
gpu_test 'eprintln!("SKIP: no snapshot");'
derived_case 1 \
    --has "fx/src/fixture_tests.rs:5 — fx gpu_alpha names no test"

workspace derived_named_notice_passes
gpu_test 'eprintln!("SKIP gpu_alpha: no snapshot");'
derived_case 0 \
    --has "every stand-down notice"

# The second rule reads the declaring files of the classified tests, so it needs the
# third column of the classification.
workspace derived_silent_env_guard_fails
gpu_test 'let Ok(p) = std::env::var("RMLX_KV_TEST_MODEL") else { return; };'
derived_case 1 \
    --has "fx/src/fixture_tests.rs:5 — gpu_alpha returns from an environment guard with no stand-down notice"

# A notice in a test the classifier does not name is out of scope, as it is under a listing.
workspace derived_unclassified_test_is_dropped
source_file fx src/fixture_tests.rs <<'RS'
#[test]
#[ignore = "GPU Metal context"]
fn gpu_alpha() {
    let device = Device::Gpu;
}

#[test]
fn cpu_beta() {
    eprintln!("SKIP: no snapshot");
}
RS
derived_case 0

# Nothing classified is not a clean tree.
workspace derived_no_gpu_test_is_exit_2
source_file fx src/fixture_tests.rs <<'RS'
#[test]
fn cpu_beta() {
    eprintln!("SKIP: no snapshot");
}
RS
derived_case 2 \
    --has "produced no GPU tests"

# A classification that cannot be trusted ends the scan with its own reason, not the
# empty-listing one: an #[ignore] claiming a Metal context with no GPU reachable.
workspace derived_untrusted_classification_is_exit_2
source_file fx src/fixture_tests.rs <<'RS'
#[test]
#[ignore = "GPU Metal context"]
fn orphaned_metal_test() {
    run(Device::Cpu);
}
RS
derived_case 2 \
    --has "claims a Metal context" \
    --lacks "produced no GPU tests"

CASE=derived_root_that_is_no_directory_is_exit_2
CASE_ROOT="$ST_WORK/nowhere"
derived_case 2 \
    --has "is not a directory"

# ---------------------------------------------------------------------------
# list: the compliant set, `<crate> <fn>` per line, and with --files the declaring file.
workspace list_names_the_crate_and_the_fn
gpu_test 'eprintln!("SKIP gpu_alpha: no snapshot");'
st_case list_names_the_crate_and_the_fn 0 \
    --is "$(printf 'fx\tgpu_alpha')" \
    -- bash "$TOOL" list --root "$CASE_ROOT"

st_case list_files_adds_the_declaring_file 0 \
    --is "$(printf 'fx\tgpu_alpha\t%s' "$CASE_ROOT/crates/fx/src/fixture_tests.rs")" \
    -- bash "$TOOL" list --files --root "$CASE_ROOT"

# in_dir <dir> <command>...: run the command with <dir> as the working directory.
in_dir() {
    (cd "$1" && shift && "$@")
}

# A relative --root is read as the directory it names, and the file column is absolute either way.
st_case list_files_names_an_absolute_file_for_a_relative_root 0 \
    --is "$(printf 'fx\tgpu_alpha\t%s' "$CASE_ROOT/crates/fx/src/fixture_tests.rs")" \
    -- in_dir "$ST_WORK" bash "$TOOL" list --files --root "$(basename "$CASE_ROOT")"

workspace list_of_nothing_is_exit_1
source_file fx src/fixture_tests.rs <<'RS'
#[test]
fn cpu_beta() {}
RS
st_case list_of_nothing_is_exit_1 1 \
    --has "classified 0 GPU-touching" \
    -- bash "$TOOL" list --root "$CASE_ROOT"

# ---------------------------------------------------------------------------
# Arguments. Each refusal is exit 2 and names its own reason.
st_case usage_no_mode 2 --has "no mode given" -- bash "$TOOL"
st_case usage_unknown_mode 2 --has "unknown mode: bogus" -- bash "$TOOL" bogus
st_case usage_unknown_option 2 --has "unknown option: --bogus" -- bash "$TOOL" list --bogus
st_case usage_files_belongs_to_list 2 --has "--files belongs to list" -- bash "$TOOL" skip-notices --files
st_case usage_listing_belongs_to_skip_notices 2 --has "--listing belongs to skip-notices" \
    -- bash "$TOOL" list --listing "$ST_WORK/none"
st_case usage_listing_needs_a_file 2 --has "--listing needs a file" -- bash "$TOOL" skip-notices --listing
st_case usage_missing_listing_file 2 --has "--listing: no such file" \
    -- bash "$TOOL" skip-notices --listing "$ST_WORK/none"
st_case usage_root_needs_a_directory 2 --has "--root needs a directory" -- bash "$TOOL" list --root
st_case usage_empty_root 2 --has "--root is empty" -- bash "$TOOL" list --root ""
st_case usage_root_given_twice 2 --has "--root given twice" \
    -- bash "$TOOL" list --root "$ST_WORK" --root "$ST_WORK"

st_done
