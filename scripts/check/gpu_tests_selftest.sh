#!/usr/bin/env bash
# Recall test for check/gpu_tests.sh skip-notices and list: planted shapes, asserting the reason.
#
# Recall test for: check/gpu_tests.sh ignored skip-notices list
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
# Every case plants a workspace and the gate classifies it, which is the path the Makefile
# takes. A classified test carries #[ignore] and names Device::Gpu; a case whose point is
# that a test is NOT classified plants it without. The cases at the end hold `list` to its
# columns and its refusals, and the ones that make a directory or a file unreadable plant
# it at run time, because a mode-000 path cannot be checked in.
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

# member_dir <crate>: where a member lives, relative to the root. A name with a slash is the
# path itself; any other is a directory under crates/.
member_dir() {
    case "$1" in
    */*) printf '%s' "$1" ;;
    *) printf 'crates/%s' "$1" ;;
    esac
}

# workspace <name> [<crate>[=<package>]]...: a root whose members are the named crates, each
# manifest naming its package (the crate name when none is given). Without a crate, one
# member, fx.
workspace() {
    CASE="$1"
    CASE_ROOT="$ST_WORK/$1"
    shift
    [ "$#" -gt 0 ] || set -- fx
    local spec dir
    mkdir -p "$CASE_ROOT"
    printf '[workspace]\nmembers = [\n' >"$CASE_ROOT/Cargo.toml"
    for spec in "$@"; do
        dir="$(member_dir "${spec%%=*}")"
        mkdir -p "$CASE_ROOT/$dir/src"
        printf '    "%s",\n' "$dir" >>"$CASE_ROOT/Cargo.toml"
        printf '[package]\nname = "%s"\nversion = "0.0.0"\nedition = "2021"\n' "${spec#*=}" >"$CASE_ROOT/$dir/Cargo.toml"
    done
    printf ']\n' >>"$CASE_ROOT/Cargo.toml"
}

# source_file <crate> <relative path>: the body on stdin.
source_file() {
    local dir
    dir="$(member_dir "$1")"
    mkdir -p "$CASE_ROOT/$dir/$(dirname "$2")"
    cat >"$CASE_ROOT/$dir/$2"
}

# check_case <exit> [--has S | --lacks S]...: skip-notices over the workspace, which the gate
# classifies, held to the exit code and to its lines.
check_case() {
    local want="$1"
    shift
    st_run bash "$TOOL" skip-notices --root "$CASE_ROOT"
    st_verdict "$CASE" "$(st_check "$want" "$@")"
}

# ---------------------------------------------------------------------------
# The clean shape. Everything below is one edit away from this, so a case that
# fails for a reason this one shares is a broken fixture rather than a finding.
workspace named_notice_passes rmlx-models
source_file rmlx-models src/alpha_tests.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    let device = Device::Gpu;
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
workspace unnamed_notice_fails rmlx-models
source_file rmlx-models src/alpha_tests.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    let device = Device::Gpu;
    eprintln!("SKIP: RMLX_TEST_MODEL_QWEN36 not set");
}
RS
check_case 1 \
    --has "gpu_alpha names no test" \
    --has "src/alpha_tests.rs:5"

# ---------------------------------------------------------------------------
# Worse than nameless: the runner WOULD list this, under a name that ran. The
# reason has to separate the two or the reader chases the wrong cell.
workspace notice_naming_another_test_fails rmlx-models
source_file rmlx-models src/alpha_tests.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    let device = Device::Gpu;
    eprintln!("SKIP gpu_beta: no snapshot");
}

#[test]
#[ignore = "GPU Metal"]
fn gpu_beta() {
    let device = Device::Gpu;
}
RS
check_case 1 \
    --has "gpu_alpha names gpu_beta, not itself" \
    --lacks "names no test"

# ---------------------------------------------------------------------------
# A helper's notice already carries the caller's name at run time (the harness's
# `SKIP {test}:` shape), and the helper is not a cell the runner selects. Fail on
# it and the gate is red on every correctly-written suite in the tree.
workspace notice_in_a_helper_is_dropped rmlx-models
source_file rmlx-models src/alpha_tests.rs <<'RS'
fn resolve(test: &str) -> Option<()> {
    eprintln!("SKIP {test}: no snapshot");
    None
}

#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    let device = Device::Gpu;
    let _ = resolve("gpu_alpha");
}
RS
check_case 0

# ---------------------------------------------------------------------------
# An empty classification is not a clean tree: the scan would have nothing to
# look at and would report success having read no test at all.
workspace empty_listing_is_exit_2 rmlx-models
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
# matching reports every notice as absent. A member outside crates/ is classified and
# still leaves crates/ with none.
workspace no_sources_is_exit_2 tools/alpha
mkdir -p "$CASE_ROOT/crates"
source_file tools/alpha src/alpha_tests.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    let device = Device::Gpu;
    eprintln!("SKIP gpu_alpha: no snapshot");
}
RS
check_case 2 \
    --has "no test-bearing sources"

# ---------------------------------------------------------------------------
# Two spaces. The runner's pattern carries exactly one, so it reads this notice
# as nameless; a gate that called it attributed would pass CI and leave the run
# INCOMPLETE with a number and no name.
workspace double_space_is_not_a_name rmlx-models
source_file rmlx-models src/alpha_tests.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    let device = Device::Gpu;
    eprintln!("SKIP  gpu_alpha: no snapshot");
}
RS
check_case 1 \
    --has "gpu_alpha names no test"

# ---------------------------------------------------------------------------
# A space before the colon, for the same reason and from the other side.
workspace space_before_colon_is_not_a_name rmlx-models
source_file rmlx-models src/alpha_tests.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    let device = Device::Gpu;
    eprintln!("SKIP gpu_alpha : no snapshot");
}
RS
check_case 1 \
    --has "gpu_alpha names no test"

# ---------------------------------------------------------------------------
# The placeholder a test that passes its own name uses. It expands to the
# runner's shape, so it is accepted — by its exact text.
workspace test_placeholder_is_accepted rmlx-models
source_file rmlx-models src/alpha_tests.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    let device = Device::Gpu;
    let test = "gpu_alpha";
    eprintln!("SKIP {test}: no snapshot");
}
RS
check_case 0

# ---------------------------------------------------------------------------
# Any other placeholder reads identically and can hold the name of a cell that
# ran. The gate cannot check what is inside it, so it does not accept it.
workspace other_placeholder_is_refused rmlx-models
source_file rmlx-models src/alpha_tests.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    let device = Device::Gpu;
    eprintln!("SKIP {other}: no snapshot");
}
RS
check_case 1 \
    --has "gpu_alpha names no test"

# ---------------------------------------------------------------------------
# A notice in a fn the classifier does not name is out of scope: the runner never
# selects it, so it never reaches a report.
workspace unclassified_test_is_dropped rmlx-models
source_file rmlx-models src/alpha_tests.rs <<'RS'
#[test]
fn gpu_alpha() {
    eprintln!("SKIP: no snapshot");
}

#[test]
#[ignore = "GPU Metal"]
fn gpu_beta() {
    let device = Device::Gpu;
}
RS
check_case 0

# ---------------------------------------------------------------------------
# Prose. The word appears in comments and in variable names throughout the tree,
# and a gate that fired on those would be turned off within a week.
workspace prose_mentioning_skip_is_not_a_notice rmlx-models
source_file rmlx-models src/alpha_tests.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    let device = Device::Gpu;
    // This cell used to SKIP when the snapshot was absent; it no longer does.
    let skip_reason_unused = 1;
    let _ = skip_reason_unused;
}
RS
check_case 0

# ---------------------------------------------------------------------------
# A notice is matched to its test by the file that declares it, so a notice under one
# crate must not be attributed to a same-named test classified under another.
workspace crate_is_read_from_the_path rmlx-models rmlx-audio
source_file rmlx-models src/alpha_tests.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    let device = Device::Gpu;
    eprintln!("SKIP gpu_alpha: no snapshot");
}
RS
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
workspace silent_env_guard_fails rmlx-models
source_file rmlx-models src/alpha_tests.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    let device = Device::Gpu;
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
workspace announced_env_guard_passes rmlx-models
source_file rmlx-models src/alpha_tests.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    let device = Device::Gpu;
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
workspace silent_env_guard_in_a_helper_fails rmlx-models
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
    let device = Device::Gpu;
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
workspace skip_gpu_guard_is_not_a_stand_down rmlx-models
source_file rmlx-models src/alpha_tests.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    let device = Device::Gpu;
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
workspace silent_guard_in_an_unclassified_file_is_dropped rmlx-models
source_file rmlx-models src/alpha_tests.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    let device = Device::Gpu;
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
workspace a_brace_in_a_literal_does_not_close_the_guard rmlx-models
source_file rmlx-models src/alpha_tests.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    let device = Device::Gpu;
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
workspace value_carrying_return_is_not_a_stand_down rmlx-models
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
    let device = Device::Gpu;
    let _ = chunk();
}
RS
check_case 0

# ---------------------------------------------------------------------------
# The guard body on the opening line. A needle anchored at line start reads
# this as a block with no return in it, which is the whole defect passing as a
# clean scan.
workspace one_line_if_guard_is_seen rmlx-models
source_file rmlx-models src/alpha_tests.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    let device = Device::Gpu;
    if std::env::var("RMLX_KV_TEST_MODEL").is_err() { return; }
}
RS
check_case 1 \
    --has "gpu_alpha returns from an environment guard with no stand-down notice"

# ---------------------------------------------------------------------------
# The other one-line spelling, and the commoner one.
workspace one_line_let_else_guard_is_seen rmlx-models
source_file rmlx-models src/alpha_tests.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    let device = Device::Gpu;
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
workspace both_kinds_are_reported rmlx-models
source_file rmlx-models src/alpha_tests.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    let device = Device::Gpu;
    let Ok(p) = std::env::var("RMLX_KV_TEST_MODEL") else {
        return;
    };
    let _ = p;
}

#[test]
#[ignore = "GPU Metal"]
fn gpu_beta() {
    let device = Device::Gpu;
    eprintln!("SKIP: no snapshot");
}
RS
check_case 1 \
    --has "gpu_alpha returns from an environment guard with no stand-down notice" \
    --has "gpu_beta names no test"

# ---------------------------------------------------------------------------
# The variable is read on one line and the block opens two lines later. A guard
# rule that wants both on one line reads this as no guard at all.
workspace silent_multi_line_let_else_guard_is_seen rmlx-models
source_file rmlx-models src/alpha_tests.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    let device = Device::Gpu;
    let Some(p) =
        std::env::var_os("RMLX_TEST_MODEL_ALPHA").map(std::path::PathBuf::from)
    else {
        return;
    };
    let _ = p;
}
RS
check_case 1 \
    --has "src/alpha_tests.rs:6 — gpu_alpha returns from an environment guard with no stand-down notice"

# A variable read into a binding, with no block, is not a guard.
workspace an_env_read_with_no_block_is_not_a_guard rmlx-models
source_file rmlx-models src/alpha_tests.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    let device = Device::Gpu;
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
workspace a_stand_down_in_another_form_fails rmlx-models
source_file rmlx-models src/alpha_tests.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    let device = Device::Gpu;
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
    --has "src/alpha_tests.rs:9 — gpu_alpha prints and returns with no value, and the block holds no stand-down notice" \
    --has 'eprintln!("gpu_alpha: skipping: RMLX_TEST_MODEL_ALPHA not set");' \
    --has "returns from an environment guard with no stand-down notice"

# The rule does not read what guards the block. A missing file and a counter
# that did not move are the same stand-down.
workspace a_stand_down_in_another_form_behind_any_guard_fails rmlx-kv-quant
source_file rmlx-kv-quant tests/alpha.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    let device = Device::Gpu;
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
    --has "tests/alpha.rs:10 — gpu_alpha prints and returns with no value, and the block holds no stand-down notice" \
    --lacks "environment guard"

# The same block with the notice in its one form passes.
workspace a_printed_return_with_the_notice_passes rmlx-kv-quant
source_file rmlx-kv-quant tests/alpha.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    let device = Device::Gpu;
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
workspace a_silent_return_after_an_outer_print_is_not_read rmlx-kv-quant
source_file rmlx-kv-quant tests/alpha.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    let device = Device::Gpu;
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
workspace a_printed_return_in_a_helper_fails rmlx-kv-quant
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
    let device = Device::Gpu;
    parity("alpha");
}
RS
check_case 1 \
    --has "tests/alpha.rs:4 — parity prints and returns with no value, and the block holds no stand-down notice"

# The same helper with the caller's name in the notice passes.
workspace a_helper_that_names_its_caller_passes rmlx-kv-quant
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
    let device = Device::Gpu;
    parity("gpu_alpha", "alpha");
}
RS
check_case 0

# A file that declares no classified test is not read.
workspace a_printed_return_in_a_file_with_no_classified_test_is_dropped rmlx-kv-quant
source_file rmlx-kv-quant tests/alpha.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    let device = Device::Gpu;
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
    workspace "a_stand_down_printed_with_${print_macro%!}_fails" rmlx-kv-quant
    source_file rmlx-kv-quant tests/alpha.rs <<RS
#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    let device = Device::Gpu;
    if nothing() {
        ${print_macro}("gpu_alpha: skipping, the dispatch did not fire");
        return;
    }
}
RS
    check_case 1 \
        --has "tests/alpha.rs:7 — gpu_alpha prints and returns with no value, and the block holds no stand-down notice"
done

# Every return with no value is read, as in the guard rule.
workspace a_stand_down_that_returns_ok_fails rmlx-kv-quant
source_file rmlx-kv-quant tests/alpha.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() -> anyhow::Result<()> {
    let device = Device::Gpu;
    if nothing() {
        eprintln!("gpu_alpha: the dispatch did not fire, skipping");
        return Ok(());
    }
    Ok(())
}
RS
check_case 1 \
    --has "tests/alpha.rs:7 — gpu_alpha prints and returns with no value, and the block holds no stand-down notice"

workspace a_stand_down_that_returns_none_fails rmlx-kv-quant
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
    let device = Device::Gpu;
    let _ = loaded("alpha");
}
RS
check_case 1 \
    --has "tests/alpha.rs:4 — loaded prints and returns with no value, and the block holds no stand-down notice"

workspace a_stand_down_that_ends_in_a_bare_return_fails rmlx-kv-quant
source_file rmlx-kv-quant tests/alpha.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    let device = Device::Gpu;
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
    --has "tests/alpha.rs:9 — gpu_alpha prints and returns with no value, and the block holds no stand-down notice"

# A return that carries a value is a result, not a stand-down.
workspace a_printed_return_with_a_value_is_not_a_stand_down rmlx-kv-quant
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
    let device = Device::Gpu;
    let _ = width();
}
RS
check_case 0

# A notice excuses the block it is in and no other. Here the first block
# announces in the one form and a later block of the same fn does not: a suite
# converted by half.
workspace a_notice_in_an_earlier_block_does_not_excuse_a_later_one rmlx-kv-quant
source_file rmlx-kv-quant tests/alpha.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    let device = Device::Gpu;
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
    --has "tests/alpha.rs:11 — gpu_alpha prints and returns with no value, and the block holds no stand-down notice" \
    --lacks "tests/alpha.rs:7"

# A print belongs to the block it is in. A block that prints and goes on does
# not make a later block's silent return an announcement.
workspace a_print_in_an_earlier_block_does_not_convict_a_later_one rmlx-kv-quant
source_file rmlx-kv-quant tests/alpha.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    let device = Device::Gpu;
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
workspace the_block_depth_starts_again_at_every_fn rmlx-kv-quant
source_file rmlx-kv-quant tests/alpha.rs <<'RS'
fn opener() -> char {
    '{'
}

#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    let device = Device::Gpu;
    eprintln!("gpu_alpha: done");
    return;
}
RS
check_case 0

# The report names the line of the return.
workspace the_other_form_is_reported_at_the_return rmlx-kv-quant
source_file rmlx-kv-quant tests/alpha.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    let device = Device::Gpu;
    if nothing() {
        eprintln!("gpu_alpha: skipping");
        let _ = cleanup();
        return;
    }
}
RS
check_case 1 \
    --has "tests/alpha.rs:8 — gpu_alpha prints"

# A pending environment read opens its guard at the next block, whatever opens
# that block: an `if` whose condition runs over two lines has no `else`.
workspace silent_multi_line_if_guard_is_seen rmlx-models
source_file rmlx-models src/alpha_tests.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    let device = Device::Gpu;
    if std::env::var("RMLX_TEST_MODEL_ALPHA")
        .is_err()
    {
        return;
    }
}
RS
check_case 1 \
    --has "src/alpha_tests.rs:5 — gpu_alpha returns from an environment guard with no stand-down notice"

# ---------------------------------------------------------------------------
# A waiver, `WAIVED <test>: <why>`, is held to the same name rule over every fn
# of a classified test's file: the caller's name through the exact `{test}`
# placeholder, or the enclosing fn's own name.
waiver_case() {
    workspace "$1" rmlx-models
    source_file rmlx-models src/alpha_tests.rs <<RS
#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    let device = Device::Gpu;
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
    --has "src/alpha_tests.rs:9 — rmlx-models check: the waiver names no test"

waiver_case waiver_wrong_placeholder 'WAIVED {name}: the streams were not compared'
check_case 1 \
    --has "src/alpha_tests.rs:9 — rmlx-models check: the waiver names no test"

waiver_case waiver_two_spaces 'WAIVED  check: the streams were not compared'
check_case 1 \
    --has "src/alpha_tests.rs:9 — rmlx-models check: the waiver names no test"

waiver_case waiver_names_another 'WAIVED gpu_beta: the streams were not compared'
check_case 1 \
    --has "src/alpha_tests.rs:9 — rmlx-models check: the waiver names gpu_beta, not itself"

# A root with no crates directory at all is the same refusal as one with no test
# source, and find's own complaint stays out of the report.
workspace no_crates_directory_is_exit_2 tools/alpha
source_file tools/alpha src/alpha_tests.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    let device = Device::Gpu;
    eprintln!("SKIP gpu_alpha: no snapshot");
}
RS
check_case 2 \
    --has "no test-bearing sources" \
    --lacks "No such file"

# ---------------------------------------------------------------------------
# Further shapes, each planted in its own workspace.

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

workspace derived_nameless_notice_fails
gpu_test 'eprintln!("SKIP: no snapshot");'
check_case 1 \
    --has "fx/src/fixture_tests.rs:5 — fx gpu_alpha names no test"

workspace derived_named_notice_passes
gpu_test 'eprintln!("SKIP gpu_alpha: no snapshot");'
check_case 0 \
    --has "every stand-down notice"

# The second rule reads the declaring files of the classified tests, so it needs the
# third column of the classification.
workspace derived_silent_env_guard_fails
gpu_test 'let Ok(p) = std::env::var("RMLX_KV_TEST_MODEL") else { return; };'
check_case 1 \
    --has "fx/src/fixture_tests.rs:5 — gpu_alpha returns from an environment guard with no stand-down notice"

# A notice in a test the classifier does not name is out of scope.
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
check_case 0

# Nothing classified is not a clean tree.
workspace derived_no_gpu_test_is_exit_2
source_file fx src/fixture_tests.rs <<'RS'
#[test]
fn cpu_beta() {
    eprintln!("SKIP: no snapshot");
}
RS
check_case 2 \
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
check_case 2 \
    --has "claims a Metal context" \
    --lacks "produced no GPU tests"

CASE=derived_root_that_is_no_directory_is_exit_2
CASE_ROOT="$ST_WORK/nowhere"
check_case 2 \
    --has "is not a directory"

# ---------------------------------------------------------------------------
# The guard rule reads past a line break, and the process-wide GPU switch is no stand-down
# there either: the variable on one line and the block on a later one.
workspace multi_line_skip_gpu_guard_is_not_a_stand_down rmlx-models
source_file rmlx-models src/alpha_tests.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    let device = Device::Gpu;
    let Some(_) =
        std::env::var_os("RMLX_SKIP_GPU")
    else {
        return;
    };
}
RS
check_case 0

workspace multi_line_guard_on_another_variable_is_a_stand_down rmlx-models
source_file rmlx-models src/alpha_tests.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    let device = Device::Gpu;
    let Some(_) =
        std::env::var_os("RMLX_KV_TEST_MODEL")
    else {
        return;
    };
}
RS
check_case 1 \
    --has "src/alpha_tests.rs:6 — gpu_alpha returns from an environment guard with no stand-down notice"

# A tests.rs below src/ is a test-bearing source: its notices are read like any other's.
workspace a_nested_tests_rs_is_scanned rmlx-models
source_file rmlx-models src/sub/tests.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    let device = Device::Gpu;
    eprintln!("SKIP: no snapshot");
}
RS
check_case 1 \
    --has "src/sub/tests.rs:5 — rmlx-models gpu_alpha names no test"

workspace a_named_notice_in_a_nested_tests_rs_passes rmlx-models
source_file rmlx-models src/sub/tests.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    let device = Device::Gpu;
    eprintln!("SKIP gpu_alpha: no snapshot");
}
RS
check_case 0

# A package whose name differs from its directory is attributed like any other: the notice is
# matched to its test by the file and the fn, and the report names the package.
workspace package_named_other_than_its_directory_is_attributed fb=fb-pkg
source_file fb src/alpha_tests.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    let device = Device::Gpu;
    eprintln!("SKIP: no snapshot");
}
RS
check_case 1 \
    --has "fb/src/alpha_tests.rs:5 — fb-pkg gpu_alpha names no test"

workspace package_named_other_than_its_directory_passes_when_named fb=fb-pkg
source_file fb src/alpha_tests.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    let device = Device::Gpu;
    eprintln!("SKIP gpu_alpha: no snapshot");
}
RS
check_case 0

# A member is the same member however it is spelled: a leading ./ (repeated), a trailing / (repeated),
# a doubled slash and a /./ in the middle all name crates/fb. The path the classifier reports carries
# none of them, so the file a notice is matched to is the file the scan finds.
for spelled in 'trailing_slash:crates/fb/' 'dot_prefix:./crates/fb' 'double_slash:crates//fb' \
    'inner_dot:crates/./fb' 'repeated_dot_prefix:././crates/fb' 'repeated_trailing_slash:crates/fb//'; do
    tag="${spelled%%:*}"
    workspace "member_with_a_${tag}" fb
    printf '[workspace]\nmembers = [\n    "%s",\n]\n' "${spelled#*:}" >"$CASE_ROOT/Cargo.toml"
    source_file fb src/alpha_tests.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    let device = Device::Gpu;
    eprintln!("SKIP: no snapshot");
}
RS
    st_case "member_with_a_${tag}_has_its_notice_attributed" 1 \
        --has "fb/src/alpha_tests.rs:5 — fb gpu_alpha names no test" \
        -- bash "$TOOL" skip-notices --root "$CASE_ROOT"
    st_case "member_with_a_${tag}_lists_clean_paths" 0 \
        --is "$(printf 'fb\tgpu_alpha\t%s' "$CASE_ROOT/crates/fb/src/alpha_tests.rs")" \
        -- bash "$TOOL" list --files --root "$CASE_ROOT"
done

# The root package is a member too, spelled `.` or `./`. Its files are src/ and tests/ under the
# root, and a member that normalises to nothing is not dropped.
for spelled in 'dot:.' 'dot_slash:./'; do
    tag="${spelled%%:*}"
    workspace "root_member_spelled_${tag}" fb
    printf '[package]\nname = "rootpkg"\nversion = "0.0.0"\n\n[workspace]\nmembers = [\n    "%s",\n    "crates/fb",\n]\n' \
        "${spelled#*:}" >"$CASE_ROOT/Cargo.toml"
    mkdir -p "$CASE_ROOT/src"
    printf '#[test]\nfn gpu_root_no_ignore() {\n    let device = Device::Gpu;\n}\n' >"$CASE_ROOT/src/root_tests.rs"
    printf '#[test]\n#[ignore = "GPU Metal"]\nfn gpu_alpha() {\n    let device = Device::Gpu;\n}\n' >"$CASE_ROOT/crates/fb/src/alpha_tests.rs"
    st_case "root_member_spelled_${tag}_is_scanned" 1 \
        --has "src/root_tests.rs: gpu_root_no_ignore" --lacks "/./" --lacks "//src" \
        -- bash "$TOOL" ignored --root "$CASE_ROOT"
done
workspace the_root_member_lists_its_files_under_the_root fb
printf '[package]\nname = "rootpkg"\nversion = "0.0.0"\n\n[workspace]\nmembers = [\n    "./",\n    "crates/fb",\n]\n' >"$CASE_ROOT/Cargo.toml"
mkdir -p "$CASE_ROOT/src"
printf '#[test]\n#[ignore = "GPU Metal"]\nfn gpu_root() {\n    let device = Device::Gpu;\n}\n' >"$CASE_ROOT/src/root_tests.rs"
printf '#[test]\n#[ignore = "GPU Metal"]\nfn gpu_alpha() {\n    let device = Device::Gpu;\n}\n' >"$CASE_ROOT/crates/fb/src/alpha_tests.rs"
st_case the_root_member_lists_its_files_under_the_root 0 \
    --has "$(printf 'rootpkg\tgpu_root\t%s' "$CASE_ROOT/src/root_tests.rs")" \
    --has "$(printf 'fb\tgpu_alpha\t%s' "$CASE_ROOT/crates/fb/src/alpha_tests.rs")" \
    -- bash "$TOOL" list --files --root "$CASE_ROOT"

# Every rule reads every file. Each case has two classified files with the violation in the one
# read first, so a rule that kept only the last file's findings would pass it.
workspace the_first_of_two_files_with_a_nameless_notice_is_found
source_file fx src/a_tests.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_a() {
    let device = Device::Gpu;
    eprintln!("SKIP: no snapshot");
}
RS
source_file fx src/b_tests.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_b() {
    let device = Device::Gpu;
}
RS
check_case 1 \
    --has "fx/src/a_tests.rs:5 — fx gpu_a names no test"

workspace the_first_of_two_files_with_a_silent_guard_is_found
source_file fx src/a_tests.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_a() {
    let device = Device::Gpu;
    let Ok(p) = std::env::var("RMLX_KV_TEST_MODEL") else { return; };
}
RS
source_file fx src/b_tests.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_b() {
    let device = Device::Gpu;
}
RS
check_case 1 \
    --has "fx/src/a_tests.rs:5 — gpu_a returns from an environment guard with no stand-down notice"

workspace the_first_of_two_files_with_a_print_and_return_is_found
source_file fx src/a_tests.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_a() {
    let device = Device::Gpu;
    if nothing() {
        eprintln!("gpu_a: skipping");
        return;
    }
}
RS
source_file fx src/b_tests.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_b() {
    let device = Device::Gpu;
}
RS
check_case 1 \
    --has "fx/src/a_tests.rs:7 — gpu_a prints and returns with no value, and the block holds no stand-down notice"

workspace the_first_of_two_files_with_a_nameless_waiver_is_found
source_file fx src/a_tests.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_a() {
    let device = Device::Gpu;
    eprintln!("WAIVED: not compared");
}
RS
source_file fx src/b_tests.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_b() {
    let device = Device::Gpu;
}
RS
check_case 1 \
    --has "fx/src/a_tests.rs:5 — fx gpu_a: the waiver names no test"

# The package a finding is attributed to is the one that declares the file: the second member's
# for a notice, and a package named other than its directory for a waiver.
workspace a_finding_in_the_second_member_is_attributed_to_it fa fb
source_file fa src/a_tests.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_a() {
    let device = Device::Gpu;
}
RS
source_file fb src/b_tests.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_b() {
    let device = Device::Gpu;
    eprintln!("SKIP: no snapshot");
}
RS
check_case 1 \
    --has "fb/src/b_tests.rs:5 — fb gpu_b names no test"

workspace a_waiver_is_attributed_to_the_package_not_the_directory fb=fb-pkg
source_file fb src/a_tests.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_a() {
    let device = Device::Gpu;
    eprintln!("WAIVED: not compared");
}
RS
check_case 1 \
    --has "fb/src/a_tests.rs:5 — fb-pkg gpu_a: the waiver names no test"

# ---------------------------------------------------------------------------
# A scan that cannot read is a refusal that names what it could not read, never a tree with
# no findings. Each case below makes a path unreadable for its run and puts it back; a user
# who can read it anyway is told so.

# unreadable <path> <name> <exit> [--has S | --lacks S]... -- <command>...: run the command with
# the path at mode 000, held to the exit code and to its output.
unreadable() {
    local path="$1" name="$2" probe="cat" restore=644
    shift 2
    if [ -d "$path" ]; then
        probe="ls"
        restore=755
    fi
    chmod 000 "$path"
    if "$probe" "$path" >/dev/null 2>&1; then
        st_skip "$name" "this user can read a mode-000 path"
    else
        st_case "$name" "$@"
    fi
    chmod "$restore" "$path"
}

# The classifier's own population: a directory under src/ it cannot list. Without the refusal
# the tests below it are not in the listing, and the listing passes for the whole population.
workspace an_unreadable_directory_under_src
gpu_test 'eprintln!("SKIP gpu_alpha: no snapshot");'
source_file fx src/sub/sub_tests.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_hidden() {
    let device = Device::Gpu;
}
RS
source_file fx tests/sub/integ.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_visible_integ() {
    let device = Device::Gpu;
}
RS
st_case a_readable_directory_under_src_is_listed 0 \
    --has "gpu_hidden" --has "gpu_alpha" --has "gpu_visible_integ" \
    -- bash "$TOOL" list --root "$CASE_ROOT"
# The tool's own line names the directory it listed; the subdirectory needle is `find`'s own
# message, which must reach the operator and so names the directory it could not open.
unreadable "$CASE_ROOT/crates/fx/tests/sub" list_refuses_an_unreadable_directory_under_tests 1 \
    --has "could not list the test files under $CASE_ROOT/crates/fx/tests." --has "crates/fx/tests/sub" \
    -- bash "$TOOL" list --root "$CASE_ROOT"
unreadable "$CASE_ROOT/crates/fx/src/sub" list_refuses_an_unreadable_directory 1 \
    --has "could not list the test files under $CASE_ROOT/crates/fx/src." --has "crates/fx/src/sub" \
    -- bash "$TOOL" list --root "$CASE_ROOT"
unreadable "$CASE_ROOT/crates/fx/src/sub" ignored_refuses_an_unreadable_directory 1 \
    --has "could not list the test files under $CASE_ROOT/crates/fx/src." --has "crates/fx/src/sub" \
    -- bash "$TOOL" ignored --root "$CASE_ROOT"
unreadable "$CASE_ROOT/crates/fx/src/sub" skip_notices_refuses_an_unreadable_directory 2 \
    --has "could not list the test files under $CASE_ROOT/crates/fx/src." --has "crates/fx/src/sub" \
    -- bash "$TOOL" skip-notices --root "$CASE_ROOT"

# A file the classifier declares its tests in, in all three modes.
workspace an_unreadable_source_is_refused
source_file fx src/alpha_tests.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    let device = Device::Gpu;
    eprintln!("SKIP: no snapshot");
}
RS
st_case a_readable_source_with_a_nameless_notice_is_found 1 \
    --has "gpu_alpha names no test" \
    -- bash "$TOOL" skip-notices --root "$CASE_ROOT"
unreadable "$CASE_ROOT/crates/fx/src/alpha_tests.rs" list_refuses_an_unreadable_source 1 \
    --has "cannot read" --has "crates/fx/src/alpha_tests.rs" \
    -- bash "$TOOL" list --root "$CASE_ROOT"
unreadable "$CASE_ROOT/crates/fx/src/alpha_tests.rs" ignored_refuses_an_unreadable_source 1 \
    --has "cannot read" --has "crates/fx/src/alpha_tests.rs" \
    -- bash "$TOOL" ignored --root "$CASE_ROOT"
unreadable "$CASE_ROOT/crates/fx/src/alpha_tests.rs" skip_notices_refuses_an_unreadable_source 2 \
    --has "cannot read" --has "crates/fx/src/alpha_tests.rs" \
    -- bash "$TOOL" skip-notices --root "$CASE_ROOT"

# skip-notices reads more than the classifier does: every test-bearing file under crates/. A
# directory or a file there that only it reads is its refusal alone.
workspace an_unreadable_path_only_skip_notices_reads
gpu_test 'eprintln!("SKIP gpu_alpha: no snapshot");'
mkdir -p "$CASE_ROOT/crates/fx/benches/tests"
printf '#[test]\nfn cpu_bench() {}\n' >"$CASE_ROOT/crates/fx/benches/tests/bench.rs"
st_case a_readable_bench_directory_passes 0 \
    --has "every stand-down notice" \
    -- bash "$TOOL" skip-notices --root "$CASE_ROOT"
unreadable "$CASE_ROOT/crates/fx/benches" skip_notices_refuses_an_unreadable_directory_the_classifier_never_reads 2 \
    --has "could not list the test-bearing sources under" \
    -- bash "$TOOL" skip-notices --root "$CASE_ROOT"
unreadable "$CASE_ROOT/crates/fx/benches" list_ignores_a_directory_the_classifier_never_reads 0 \
    --is "$(printf 'fx\tgpu_alpha')" \
    -- bash "$TOOL" list --root "$CASE_ROOT"
unreadable "$CASE_ROOT/crates/fx/benches/tests/bench.rs" skip_notices_refuses_an_unreadable_file_the_classifier_never_reads 2 \
    --has "cannot read" --has "benches/tests/bench.rs" \
    -- bash "$TOOL" skip-notices --root "$CASE_ROOT"

# Every file of a population is checked, not the first: the unreadable one is read last, and the
# refusal names it.
workspace the_last_unreadable_file_of_a_population_is_named
source_file fx src/a_tests.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_a() {
    let device = Device::Gpu;
}
RS
source_file fx tests/z.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_z() {
    let device = Device::Gpu;
}
RS
mkdir -p "$CASE_ROOT/crates/fx/zz/tests"
printf 'fn helper() {}\n' >"$CASE_ROOT/crates/fx/zz/tests/b.rs"
unreadable "$CASE_ROOT/crates/fx/tests/z.rs" list_names_the_last_unreadable_file_it_classifies 1 \
    --has "cannot read $CASE_ROOT/crates/fx/tests/z.rs." \
    -- bash "$TOOL" list --root "$CASE_ROOT"
unreadable "$CASE_ROOT/crates/fx/zz/tests/b.rs" skip_notices_names_the_last_unreadable_file_it_scans 2 \
    --has "cannot read $CASE_ROOT/crates/fx/zz/tests/b.rs." \
    -- bash "$TOOL" skip-notices --root "$CASE_ROOT"

# A scratch file that cannot be made is a refusal that says so.
workspace a_scratch_file_that_cannot_be_made_is_refused
gpu_test 'eprintln!("SKIP gpu_alpha: no snapshot");'
st_case a_scratch_file_that_cannot_be_made_is_refused 1 \
    --has "cannot make a scratch file" \
    -- env TMPDIR="$ST_WORK/no/such/directory" bash "$TOOL" list --root "$CASE_ROOT"

# The root itself, and a directory under crates/ that is no member: the member floor counts
# manifests there, and a count over a directory it could not list is no count.
workspace an_unreadable_root_is_refused
gpu_test 'eprintln!("SKIP gpu_alpha: no snapshot");'
unreadable "$CASE_ROOT" list_refuses_an_unreadable_root 1 \
    --has "cannot enter the scan root" \
    -- bash "$TOOL" list --root "$CASE_ROOT"
unreadable "$CASE_ROOT" skip_notices_refuses_an_unreadable_root 2 \
    --has "cannot enter the scan root" \
    -- bash "$TOOL" skip-notices --root "$CASE_ROOT"
mkdir -p "$CASE_ROOT/crates/zz"
unreadable "$CASE_ROOT/crates/zz" ignored_refuses_a_crates_directory_it_cannot_count 1 \
    --has "could not list the member manifests under" \
    -- bash "$TOOL" ignored --root "$CASE_ROOT"
unreadable "$CASE_ROOT/crates/zz" skip_notices_refuses_a_crates_directory_it_cannot_count 2 \
    --has "could not list the member manifests under" \
    -- bash "$TOOL" skip-notices --root "$CASE_ROOT"

# The workspace manifest is read by an awk too, so an unreadable one is refused before it, naming it.
workspace an_unreadable_workspace_manifest_is_refused
gpu_test 'eprintln!("SKIP gpu_alpha: no snapshot");'
unreadable "$CASE_ROOT/Cargo.toml" list_refuses_an_unreadable_workspace_manifest 1 \
    --has "cannot read $CASE_ROOT/Cargo.toml" --lacks "parsed 0 workspace members" \
    -- bash "$TOOL" list --root "$CASE_ROOT"
unreadable "$CASE_ROOT/Cargo.toml" ignored_refuses_an_unreadable_workspace_manifest 1 \
    --has "cannot read $CASE_ROOT/Cargo.toml" --lacks "parsed 0 workspace members" \
    -- bash "$TOOL" ignored --root "$CASE_ROOT"
unreadable "$CASE_ROOT/Cargo.toml" skip_notices_refuses_an_unreadable_workspace_manifest 2 \
    --has "cannot read $CASE_ROOT/Cargo.toml" --lacks "parsed 0 workspace members" \
    -- bash "$TOOL" skip-notices --root "$CASE_ROOT"

# A command that fails reaches the verdict. The one that stands in for the real one fails when
# an argument holds a marker (any argument, for an empty marker) and otherwise is the real one,
# so the rest of the run is untouched. Without the check, an awk program that one awk cannot
# compile switches its rule off and the mode passes.
#
# failing_command <command> <marker>: print a directory holding that command, to put first on PATH.
failing_command() {
    local dir real
    dir="$(mktemp -d "$ST_WORK/$1.XXXXXX")"
    real="$(command -v "$1")"
    {
        # shellcheck disable=SC2016 # the script written here expands these, not this one
        printf '#!/bin/sh\nfor arg in "$@"; do\n    case "$arg" in\n    *%s*)\n' "'$2'"
        printf '        echo "%s: planted failure" >&2\n        exit 2\n        ;;\n    esac\ndone\nexec "%s" "$@"\n' "$1" "$real"
    } >"$dir/$1"
    chmod +x "$dir/$1"
    printf '%s' "$dir"
}

workspace a_failing_awk_is_a_refusal rmlx-models
source_file rmlx-models src/alpha_tests.rs <<'RS'
#[test]
#[ignore = "GPU Metal"]
fn gpu_alpha() {
    let device = Device::Gpu;
    eprintln!("SKIP gpu_alpha: no snapshot");
}
RS
st_case an_awk_that_fails_on_nothing_changes_nothing 0 \
    --has "every stand-down notice" \
    -- env PATH="$(failing_command awk 'a marker no program holds'):$PATH" bash "$TOOL" skip-notices --root "$CASE_ROOT"
detector_awk="$(failing_command awk 'flush_attr')"
st_case a_failing_detector_is_refused_by_list 1 \
    --has "the detector failed over" --has "crates/rmlx-models" \
    -- env PATH="$detector_awk:$PATH" bash "$TOOL" list --root "$CASE_ROOT"
st_case a_failing_detector_is_refused_by_ignored 1 \
    --has "the detector failed over" \
    -- env PATH="$detector_awk:$PATH" bash "$TOOL" ignored --root "$CASE_ROOT"
st_case a_failing_detector_is_refused_by_skip_notices 2 \
    --has "the detector failed over" \
    -- env PATH="$detector_awk:$PATH" bash "$TOOL" skip-notices --root "$CASE_ROOT"
for scan in 'notice:verdict = "none"' 'guard:gtext' 'print:printed[sp]' 'waiver:why = "names no test"'; do
    st_case "a_failing_${scan%%:*}_scan_is_refused" 2 \
        --has "the scan of" --has "crates/rmlx-models/src/alpha_tests.rs failed" \
        -- env PATH="$(failing_command awk "${scan#*:}"):$PATH" bash "$TOOL" skip-notices --root "$CASE_ROOT"
done

# The file a scan lists into is removed on every exit, the refusals included.
workspace a_run_leaves_no_scratch_file
gpu_test 'eprintln!("SKIP gpu_alpha: no snapshot");'
scratch="$ST_WORK/scratch"
mkdir -p "$scratch"
st_run env TMPDIR="$scratch" bash "$TOOL" list --root "$CASE_ROOT"
st_run env TMPDIR="$scratch" bash "$TOOL" list --root "$ST_WORK/nowhere"
left="$(ls -A "$scratch")"
st_verdict a_run_leaves_no_scratch_file "$([ -z "$left" ] || printf 'left behind: %s' "$left")"

# The two reads of that file that are not an awk or a find are refusals too, and name the file.
workspace a_count_or_sort_of_the_scratch_file_that_fails_is_refused
gpu_test 'eprintln!("SKIP gpu_alpha: no snapshot");'
st_case a_member_count_that_fails_is_refused 1 \
    --has "could not count the member manifests listed in" --has "/gpu_tests." \
    -- env PATH="$(failing_command wc ''):$PATH" bash "$TOOL" ignored --root "$CASE_ROOT"
st_case a_file_sort_that_fails_is_refused 2 \
    --has "could not sort the test-bearing sources listed in" --has "/gpu_tests." \
    -- env PATH="$(failing_command sort '/gpu_tests.'):$PATH" bash "$TOOL" skip-notices --root "$CASE_ROOT"

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
st_case usage_listing_is_refused 2 --has "unknown option: --listing" \
    -- bash "$TOOL" skip-notices --listing "$ST_WORK/none"
st_case usage_root_needs_a_directory 2 --has "--root needs a directory" -- bash "$TOOL" list --root
st_case usage_empty_root 2 --has "--root is empty" -- bash "$TOOL" list --root ""
st_case usage_root_given_twice 2 --has "--root given twice" \
    -- bash "$TOOL" list --root "$ST_WORK" --root "$ST_WORK"

st_done
