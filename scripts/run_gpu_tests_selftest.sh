#!/usr/bin/env bash
# run_gpu_tests_selftest.sh — recall check for how `scripts/run_gpu_tests.sh`
# reports a red run.
#
# The runner accumulates two independent kinds of red across its crate loop:
# Metal shader-validation diagnostics, and everything that makes a crate fail
# (a failing test, a crate that under-matched its classified population, a crate
# that produced no validation banner). Both have to reach the operator in the
# same run. A tree carrying a standing validation diagnostic otherwise turns
# every genuine test failure into silence — the failing test names are computed,
# held in a shell variable, and thrown away at exit, and each crate's log is
# deleted inside the loop, so nothing survives to re-read.
#
# It also decides which validation hits are a failure at all. The runner accepts
# exactly the census pinned in `scripts/gpu_validation_census.txt` and fails on
# any deviation from it, so both halves of that — the pass and each kind of
# deviation — are checked here too. A pin that could only fail would leave the
# gate as red as it was without one; a pin that could only pass would be a gate
# that cannot fire.
#
# That is a property of the reporting code, not of the GPU, so it is checked
# here against stubs: a stub `cargo` replays a canned libtest log per crate, a
# stub classifier names the population, and each case writes its own pin. No
# Metal device, no snapshot, no compile. The runner is copied into a throwaway
# root per case rather than reimplemented, so this file cannot drift from what
# the gate runs.
#
# Every case asserts the REASON — the strings an operator triages a red gate
# from, taken only from the final report block, not from the interleaved `tee`
# output the report is supposed to summarise — so a runner that exits 1 while
# reporting the wrong half of the run still fails here.
#
# The suite is also partitioned, so a change can pay for the part of it that
# guards what the change touched, and that partition is a gate of its own: the
# cases under THE HALVES below hold it to a union that loses nothing, a census
# slice keyed on the pin's own crate and test, and a stand-down that stays one.
#
# Exit 0 = every case reported exactly what it should.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER="${ROOT}/scripts/run_gpu_tests.sh"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/rmlx_gpu_runner_selftest.XXXXXX")"
trap 'rm -rf "${WORK}"' EXIT

failures=0
CASE=""
CASE_ROOT=""
OUT=""
STATUS=0
REPORT=""
MIX=""
ARGV=""
HALVES_OUT=""

# new_case <name> — build a throwaway repo root: a copy of the runner, a stub
# classifier reading this case's population, and a stub PATH. Sets CASE_ROOT.
# Not a command substitution: the case name has to reach the assertions below,
# and a subshell would drop it.
new_case() {
    CASE="$1"
    CASE_ROOT="${WORK}/$1"
    local root="${CASE_ROOT}"
    mkdir -p "${root}/scripts/lib" "${root}/bin" "${root}/logs" || return 1
    cp "${RUNNER}" "${root}/scripts/run_gpu_tests.sh" || return 1
    # Symlinked, not copied: the runner reads the stand-down notice's shape from
    # this file and so does the source gate, and a fixture carrying its own copy
    # would keep passing after the real shape moved.
    ln -sf "${ROOT}/scripts/lib/skip_notice_patterns.sh" \
        "${root}/scripts/lib/skip_notice_patterns.sh" || return 1
    : >"${root}/classified"
    # The pin is a tracked file and its absence is its own error, so every case
    # starts from an empty one and says so; the cases that pin something
    # overwrite it.
    printf '# kernel | kind | count | crate | test | reference\n' \
        >"${root}/scripts/gpu_validation_census.txt"

    cat >"${root}/scripts/check_gpu_tests_ignored.sh" <<STUB
#!/usr/bin/env bash
cat "${root}/classified"
STUB

    # The half each classified test belongs to. It is a second population on top
    # of the classification, so it gets its own stub: a case states the
    # partition it is testing instead of re-deriving it from a fixture tree,
    # and the one case that does exercise the real rule runs the real producer.
    : >"${root}/halves"
    cat >"${root}/scripts/gpu_test_halves.sh" <<STUB
#!/usr/bin/env bash
cat "${root}/halves"
STUB

    # The stub answers the shader validation canary so the detector's positive
    # control passes, and records its own argv. The runner reports a per-crate
    # count and no executed set, so with one crate declaring a cell in each half
    # the two halves print the same count — the libtest filters the runner
    # actually issued are the only observable that says WHICH cell it asked for.
    : >"${root}/cargo_argv"
    cat >"${root}/bin/cargo" <<STUB
#!/usr/bin/env bash
set -u
printf '%s\n' "\$@" >>"${root}/cargo_argv"
crate=""
prev=""
for a in "\$@"; do
    [ "\$prev" = "-p" ] && crate="\$a"
    if [ "\$a" = "shader-validation-canary" ]; then
        echo 'Metal GPU Validation Enabled'
        echo 'Invalid device store at offset 4000068, executing kernel function: "custom_kernel_rmlx_canary_oob_store"'
        echo 'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out'
        exit 0
    fi
    prev="\$a"
done
if [ ! -f "${root}/logs/\${crate}.log" ]; then
    echo "stub cargo: no canned log for '\${crate}'" >&2
    exit 99
fi
cat "${root}/logs/\${crate}.log"
exit "\$(cat "${root}/logs/\${crate}.rc" 2>/dev/null || echo 0)"
STUB

    chmod +x "${root}/bin/cargo" || return 1
}

# classify <root> <crate> <fn>... — name the GPU tests the classifier reports.
classify() {
    local root="$1" crate="$2" fn
    shift 2
    for fn in "$@"; do
        printf '%s\t%s\n' "${crate}" "${fn}" >>"${root}/classified"
    done
}

# halves <root> — the partition this case runs under, `half<TAB>crate<TAB>test`
# rows on stdin.
halves() {
    cat >"$1/halves"
}

# gpu_fixture_test <file> <fn> [body-line] — a test the classifier reads as
# GPU-touching and compliant, for the cases that run the real half producer over
# a tree. The optional body line is what a file uses to name a codec.
gpu_fixture_test() {
    cat >"$1" <<TEST
#[test]
#[ignore = "GPU: needs the Metal context to itself"]
fn $2() {
    let device = Device::Gpu;
    let _ = device;
    ${3:-}
}
TEST
}

# halves_fixture_tree <dir> — a three-member workspace the real producer and the
# real classifier can both read: one member the model layer depends on, the model
# layer, and one it does not depend on.
#
# One member per line: the classifier reads the list line by line, and a
# single-line array parses as zero members and fails closed — which would make
# every case over this tree permanently red for a reason that has nothing to do
# with the rule.
halves_fixture_tree() {
    local tree="$1" member
    mkdir -p "${tree}/crates/rmlx-kv-quant/src" \
        "${tree}/crates/rmlx-models/src" "${tree}/crates/rmlx-models/tests" \
        "${tree}/crates/rmlx-audio/src" || return 1
    cat >"${tree}/Cargo.toml" <<'TOML'
[workspace]
members = [
    "crates/rmlx-kv-quant",
    "crates/rmlx-models",
    "crates/rmlx-audio",
]
TOML
    for member in rmlx-kv-quant rmlx-models rmlx-audio; do
        printf '[package]\nname = "%s"\n\n[dependencies]\n' "${member}" \
            >"${tree}/crates/${member}/Cargo.toml"
    done
    printf 'rmlx-kv-quant = { workspace = true }\n' \
        >>"${tree}/crates/rmlx-models/Cargo.toml"
}

# expect_placed <half> <crate> <test> — a row the producer must emit for the
# fixture tree most recently read into HALVES_OUT.
expect_placed() {
    case $'\n'"${HALVES_OUT}"$'\n' in
        *$'\n'"$1	$2	$3"$'\n'*) ;;
        *) fail "the producer did not place: $1 $2 $3" ;;
    esac
}

# crate_log <root> <crate> <cargo-exit-code> — canned libtest log on stdin.
crate_log() {
    local root="$1" crate="$2" rc="$3"
    cat >"${root}/logs/${crate}.log"
    printf '%s\n' "${rc}" >"${root}/logs/${crate}.rc"
}

# pin <root> — write this case's shader-validation census pin, contents on stdin.
pin() {
    cat >"$1/scripts/gpu_validation_census.txt"
}

# run_case <root> [runner args...] — run this case's runner; set OUT, STATUS,
# REPORT and MIX.
#
# REPORT is the final block only: everything from the first post-loop ERROR
# header on. The crate logs are teed to the same stream, so asserting against
# OUT would pass on a failing test name that only ever appeared in the 1000-line
# scroll the operator is not reading.
#
# MIX is narrower still — the counted access-kind lines alone. The prose around
# them is static and mentions neither kind, but asserting a kind's ABSENCE over
# the whole report would be a statement about that prose as much as about the
# tally, and would start passing for the wrong reason the day the wording moves.
run_case() {
    local root="$1"
    shift
    : >"${root}/cargo_argv"
    OUT="$(PATH="${root}/bin:${PATH}" env -u RMLX_SKIP_GPU \
        RMLX_O_MODELS_ROOT="${WORK}" bash "${root}/scripts/run_gpu_tests.sh" "$@" 2>&1)"
    STATUS=$?
    ARGV="$(cat "${root}/cargo_argv")"
    REPORT="$(printf '%s\n' "${OUT}" | awk '
        /^ERROR: Metal shader validation reported invalid memory access:/ { seen = 1 }
        /^ERROR: the shader-validation census does not match the pin:/ { seen = 1 }
        /^ERROR: GPU tests failed in:/ { seen = 1 }
        seen')"
    MIX="$(printf '%s\n' "${REPORT}" | awk '
        /^Access mix over the hits above:$/ { in_mix = 1; next }
        in_mix && /^[[:space:]]+[0-9]+[[:space:]]/ { print; next }
        in_mix { in_mix = 0 }')"
}

fail() {
    echo "FAIL [${CASE}] $1" >&2
    failures=$((failures + 1))
}

expect_status() {
    [ "${STATUS}" = "$1" ] || fail "exit ${STATUS}, expected $1"
}

expect_report() {
    case "${REPORT}" in
        *"$1"*) ;;
        *) fail "final report does not mention: $1" ;;
    esac
}

expect_no_report() {
    case "${REPORT}" in
        *"$1"*) fail "final report should not mention: $1" ;;
    esac
}

expect_mix() {
    case "${MIX}" in
        *"$1"*) ;;
        *) fail "access mix does not count: $1" ;;
    esac
}

expect_no_mix() {
    case "${MIX}" in
        *"$1"*) fail "access mix should not count: $1" ;;
    esac
}

expect_out() {
    case "${OUT}" in
        *"$1"*) ;;
        *) fail "output does not mention: $1" ;;
    esac
}

expect_no_out() {
    case "${OUT}" in
        *"$1"*) fail "output should not mention: $1" ;;
    esac
}

# The executed set, read from the libtest filters the runner issued rather than
# from its report, which carries a count and no names.
expect_ran() {
    case $'\n'"${ARGV}"$'\n' in
        *$'\n'"$1"$'\n'*) ;;
        *) fail "the runner did not ask cargo to run: $1" ;;
    esac
}

expect_did_not_run() {
    case $'\n'"${ARGV}"$'\n' in
        *$'\n'"$1"$'\n'*) fail "the runner asked cargo to run: $1" ;;
    esac
}

# ---------------------------------------------------------------------------
# A validation hit and a failing test in the same run: both are reported.
# This is the masking case — the hit alone is enough to fail the run, so a
# runner that reports it and exits never mentions the test that also went red.
#
# The diagnostic lands appended to a libtest line on purpose: that is the shape
# the validation layer actually produces, and an anchored detector misses it.
new_case both_kinds || exit 1
classify "${CASE_ROOT}" rmlx-kv-quant kv_gpu_alpha kv_gpu_beta
crate_log "${CASE_ROOT}" rmlx-kv-quant 101 <<'LOG'
Metal GPU Validation Enabled
running 2 tests
test kv::gpu_alpha ... ok
test kv::gpu_beta ... FAILEDInvalid device load at offset 4000068, executing kernel function: "affine_qmm_t_splitk_bfloat16_t_gs_64_b_8_alN_false"

failures:

---- kv::gpu_beta stdout ----
    Divergence
thread 'kv::gpu_beta' panicked at crates/rmlx-kv-quant/src/codec_tests.rs:165:5:
sorted vs broadcast diverge beyond atol+rtol*|b| by 0.059472658
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace

failures:
    kv::gpu_beta

test result: FAILED. 1 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.00s
LOG
run_case "${CASE_ROOT}"
expect_status 1
expect_report "ERROR: Metal shader validation reported invalid memory access:"
expect_mix "1 device load"
expect_report "ERROR: GPU tests failed in:"
expect_report "kv::gpu_beta"
# The captured-stdout block sits between the same two markers the failing names
# are harvested from, and a panic detail can be indented exactly like a name. A
# harvester that scrapes it reports lines that are not tests as if they were.
expect_no_report "Divergence"

# ---------------------------------------------------------------------------
# A failing test with no validation hit still reports as itself.
new_case failure_only || exit 1
classify "${CASE_ROOT}" rmlx-kv-quant kv_gpu_alpha kv_gpu_beta
crate_log "${CASE_ROOT}" rmlx-kv-quant 101 <<'LOG'
Metal GPU Validation Enabled
running 2 tests
test kv::gpu_alpha ... ok
test kv::gpu_beta ... FAILED

failures:

---- kv::gpu_beta stdout ----
    Divergence
thread 'kv::gpu_beta' panicked at crates/rmlx-kv-quant/src/codec_tests.rs:165:5:
sorted vs broadcast diverge beyond atol+rtol*|b| by 0.059472658
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace

failures:
    kv::gpu_beta

test result: FAILED. 1 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.00s
LOG
run_case "${CASE_ROOT}"
expect_status 1
expect_report "ERROR: GPU tests failed in:"
expect_report "kv::gpu_beta"
expect_no_report "Divergence"
expect_no_report "Metal shader validation reported"

# ---------------------------------------------------------------------------
# Hits only, all of them loads: the banner reports the mix it saw. A hardcoded
# claim of "store" over a run of pure loads sends the reader after the wrong
# kernel, and severity differs between the two. Two of the four diagnostics
# share one output line, which is routine — the layer writes while libtest is
# mid-line — so this also pins that hits are counted per diagnostic and not per
# line, without which the mix would not sum to the count printed beside it.
new_case loads_only || exit 1
classify "${CASE_ROOT}" rmlx-kv-quant kv_gpu_alpha
crate_log "${CASE_ROOT}" rmlx-kv-quant 0 <<'LOG'
Metal GPU Validation Enabled
running 1 test
Invalid device load at offset 4000068, executing kernel function: "affine_qmm_t_splitk"
Invalid device load at offset 4000132, executing kernel function: "affine_qmm_t_splitk"
test kv::gpu_alpha ... okInvalid device load at offset 4000196, executing kernel function: "affine_qmm_t_splitk_bfloat16_t_gs_64_b_8_alN_false"Invalid device load at offset 4000260, executing kernel function: "affine_qmm_t_splitk_bfloat16_t_gs_64_b_8_alN_false"
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.00s
LOG
run_case "${CASE_ROOT}"
expect_status 1
expect_report "rmlx-kv-quant: 4 invalid access(es)"
expect_mix "4 device load"
expect_no_mix "device store"
expect_no_report "ERROR: GPU tests failed in:"

# ---------------------------------------------------------------------------
# The converse: a pure-store run must not be described as loads.
new_case stores_only || exit 1
classify "${CASE_ROOT}" rmlx-kv-quant kv_gpu_alpha
crate_log "${CASE_ROOT}" rmlx-kv-quant 0 <<'LOG'
Metal GPU Validation Enabled
running 1 test
Invalid device store at offset 4000068, executing kernel function: "custom_kernel_rmlx_q8_quantize"
Invalid device store at offset 4000132, executing kernel function: "custom_kernel_rmlx_q8_quantize"
test kv::gpu_alpha ... ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.00s
LOG
run_case "${CASE_ROOT}"
expect_status 1
expect_mix "2 device store"
expect_no_mix "device load"

# ---------------------------------------------------------------------------
# Every kind of access across two crates is counted, not just the first — and
# `device` is not the only spelling the layer emits, so a threadgroup access
# rides along to keep the tally from being read off a hardcoded pair.
new_case mixed_kinds || exit 1
classify "${CASE_ROOT}" rmlx-kv-quant kv_gpu_alpha
classify "${CASE_ROOT}" rmlx-models models_gpu_alpha
crate_log "${CASE_ROOT}" rmlx-kv-quant 0 <<'LOG'
Metal GPU Validation Enabled
running 1 test
Invalid device load at offset 4000068, executing kernel function: "affine_qmm_t_splitk"
Invalid device load at offset 4000132, executing kernel function: "affine_qmm_t_splitk"
test kv::gpu_alpha ... ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.00s
LOG
crate_log "${CASE_ROOT}" rmlx-models 0 <<'LOG'
Metal GPU Validation Enabled
running 1 test
Invalid device store at offset 512, executing kernel function: "custom_kernel_rmlx_q8_quantize"
Invalid device load at offset 640, executing kernel function: "custom_kernel_rmlx_q8_quantize"
Invalid threadgroup load at offset 96, executing kernel function: "custom_kernel_rmlx_q8_quantize"
test models::gpu_alpha ... ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.00s
LOG
run_case "${CASE_ROOT}"
expect_status 1
expect_mix "3 device load"
expect_mix "1 device store"
expect_mix "1 threadgroup load"

# ---------------------------------------------------------------------------
# Two diagnostics adjacent on one output line, under a kernel name short enough
# that the second one starts within the detector's bounded window. The pattern
# is greedy, so a single match spans both and the second access — a store, the
# severe kind — disappears from the count and the mix. Kernel names in this tree
# run from about 20 to 50 characters, so which of the two shapes a run produces
# is not something the reader controls.
new_case adjacent_hits_short_kernel_name || exit 1
classify "${CASE_ROOT}" rmlx-kv-quant kv_gpu_alpha
crate_log "${CASE_ROOT}" rmlx-kv-quant 0 <<'LOG'
Metal GPU Validation Enabled
running 1 test
test kv::gpu_alpha ... okInvalid device load at offset 4096, executing kernel function: "custom_kernel_rmlx_q8_quantize"Invalid device store at offset 8192, executing kernel function: "custom_kernel_rmlx_q8_quantize"
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.00s
LOG
run_case "${CASE_ROOT}"
expect_status 1
expect_report "rmlx-kv-quant: 2 invalid access(es)"
expect_mix "1 device load"
expect_mix "1 device store"

# ---------------------------------------------------------------------------
# A crate that both under-matched and failed tests: an aborting test binary
# produces exactly that pair, since the tests after the abort never run. Both
# lines have to reach the report — the under-match alone says a filter stopped
# matching, which sends the reader looking for a renamed fn rather than at the
# test that took the binary down.
new_case undermatch_plus_failing_test || exit 1
classify "${CASE_ROOT}" rmlx-kv-quant kv_gpu_alpha kv_gpu_beta kv_gpu_gamma
crate_log "${CASE_ROOT}" rmlx-kv-quant 101 <<'LOG'
Metal GPU Validation Enabled
running 3 tests
test kv::gpu_alpha ... ok
test kv::gpu_beta ... FAILED

failures:

---- kv::gpu_beta stdout ----
thread 'kv::gpu_beta' panicked at crates/rmlx-kv-quant/src/codec_tests.rs:165:5:
sorted vs broadcast diverge beyond atol+rtol*|b| by 0.059472658
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace

failures:
    kv::gpu_beta

test result: FAILED. 1 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.00s
LOG
run_case "${CASE_ROOT}"
expect_status 1
expect_report "under-matched (2/3 executed)"
expect_report "kv::gpu_beta"

# ---------------------------------------------------------------------------
# A crate that under-matched its classified population is a second kind of
# crate failure, and it must survive a co-occurring validation hit too — the
# ordering has to hold for every failure kind the runner can report, not for
# the failing-test one alone.
new_case undermatch_with_hit || exit 1
classify "${CASE_ROOT}" rmlx-kv-quant kv_gpu_alpha kv_gpu_beta kv_gpu_gamma
crate_log "${CASE_ROOT}" rmlx-kv-quant 0 <<'LOG'
Metal GPU Validation Enabled
running 1 test
test kv::gpu_alpha ... okInvalid device load at offset 4000068, executing kernel function: "affine_qmm_t_splitk"
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.00s
LOG
run_case "${CASE_ROOT}"
expect_status 1
expect_report "Metal shader validation reported"
expect_report "under-matched (1/3 executed)"

# ---------------------------------------------------------------------------
# So must the third kind: a crate that produced no validation banner ran
# uninstrumented (usually it failed to build), while another crate reported hits.
new_case uninstrumented_with_hit || exit 1
classify "${CASE_ROOT}" rmlx-kv-quant kv_gpu_alpha
classify "${CASE_ROOT}" rmlx-models models_gpu_alpha
crate_log "${CASE_ROOT}" rmlx-kv-quant 0 <<'LOG'
Metal GPU Validation Enabled
running 1 test
test kv::gpu_alpha ... okInvalid device load at offset 4000068, executing kernel function: "affine_qmm_t_splitk"
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.00s
LOG
crate_log "${CASE_ROOT}" rmlx-models 0 <<'LOG'
running 1 test
test models::gpu_alpha ... ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.00s
LOG
run_case "${CASE_ROOT}"
expect_status 1
expect_report "Metal shader validation reported"
expect_report "ran uninstrumented (no validation banner)"

# ---------------------------------------------------------------------------
# The census pin. A tree can carry a validated-benign diagnostic from a kernel
# it does not own, and a gate that stays red on it teaches its readers that
# `Error 1` is background noise. The pin records that census exactly — one entry
# per originating test, carrying that test's own count — and the runner expects
# the sum of the counts whose test actually ran. Anything else is a failure
# naming the delta.
#
# Every case below writes its own pin, so none of them depends on what the tree
# happens to accept today. The diagnostics are glued onto a libtest line,
# because that is the shape the validation layer really produces.

CENSUS_KERNEL="mlx_qmm_stub"

# census_pin <root> <count> [test] [crate] — a one-entry pin.
census_pin() {
    census_pin_line "$1" "${2}" "${3:-kv_gpu_alpha}" "${4:-rmlx-kv-quant}" >"$1/scripts/gpu_validation_census.txt"
}

# census_pin_line <root> <count> <test> <crate> — one entry, on stdout.
census_pin_line() {
    printf '%s | device load | %s | %s | %s | validated benign\n' \
        "${CENSUS_KERNEL}" "$2" "$4" "$3"
}

# census_log <root> <crate> <n> [skip-test] [passed] [note-test] — a clean
# libtest log carrying <n> loads of CENSUS_KERNEL, glued onto the passing test's
# line, and optionally the named test's own skip notice or a plain note naming
# it. <passed> must cover the crate's classified population or the runner
# reports an under-match instead.
census_log() {
    local root="$1" crate="$2" n="$3" skip="${4:-}" passed="${5:-1}" note="${6:-}" i=0 line=""
    while [ "${i}" -lt "${n}" ]; do
        line="${line}Invalid device load at offset $((4096 + i * 64)), executing kernel function: \"${CENSUS_KERNEL}\""
        i=$((i + 1))
    done
    {
        echo 'Metal GPU Validation Enabled'
        echo "running ${passed} tests"
        [ -n "${note}" ] && echo "note ${note}: one of its snapshots is not on this machine"
        [ -n "${skip}" ] && echo "test kv::gpu_alpha ... SKIP ${skip}: no snapshot on this machine"
        echo "test kv::gpu_alpha ... ok${line}"
        echo "test result: ok. ${passed} passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.00s"
    } | crate_log "${root}" "${crate}" 0
}

# The accepted case: the observed tally is exactly the expectation, so the run is
# green and prints the census it accepted. Without this the pin would be a gate
# that can only fail, which is the bug it was written against.
new_case census_exact_match || exit 1
classify "${CASE_ROOT}" rmlx-kv-quant kv_gpu_alpha
census_pin "${CASE_ROOT}" 4
census_log "${CASE_ROOT}" rmlx-kv-quant 4
run_case "${CASE_ROOT}"
expect_status 0
expect_out "census matches the pin"
expect_out "4 device load \"${CENSUS_KERNEL}\" in rmlx-kv-quant"
expect_out "kv_gpu_alpha = 4"
expect_no_out "ERROR:"
# A run that accepted four invalid accesses is not a clean one, and saying so
# would put the operator back where a permanently red gate left them.
expect_no_out "shader validation clean"

# A cell that names an absence without standing down keeps its entry in the
# expectation. A test resolving several snapshots and running on the ones it
# found must not announce `SKIP <itself>:` for the ones it did not: the harvest
# is per test, so that notice drops every entry naming it — including the kernel
# the test did produce hits for — and the run then fails on a count it has
# dropped the expectation for. A plain note is invisible to both scans, which is
# the property this case holds.
new_case census_unnamed_note_keeps_the_entry || exit 1
classify "${CASE_ROOT}" rmlx-kv-quant kv_gpu_alpha
census_pin "${CASE_ROOT}" 4
census_log "${CASE_ROOT}" rmlx-kv-quant 4 "" 1 kv_gpu_alpha
run_case "${CASE_ROOT}"
expect_status 0
expect_out "census matches the pin"
expect_out "kv_gpu_alpha = 4"
expect_no_out "not enforced in full"
expect_no_out "named no test"

# A kernel the pin does not name is a new hit, whatever the pinned ones did.
new_case census_new_kernel || exit 1
classify "${CASE_ROOT}" rmlx-kv-quant kv_gpu_alpha
census_pin "${CASE_ROOT}" 4
crate_log "${CASE_ROOT}" rmlx-kv-quant 0 <<LOG
Metal GPU Validation Enabled
running 1 test
test kv::gpu_alpha ... okInvalid device load at offset 4096, executing kernel function: "${CENSUS_KERNEL}"Invalid device load at offset 4160, executing kernel function: "${CENSUS_KERNEL}"Invalid device load at offset 4224, executing kernel function: "${CENSUS_KERNEL}"Invalid device load at offset 4288, executing kernel function: "${CENSUS_KERNEL}"Invalid device load at offset 8192, executing kernel function: "custom_kernel_rmlx_q8_quantize"
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.00s
LOG
run_case "${CASE_ROOT}"
expect_status 1
expect_report "not pinned: 1 device load \"custom_kernel_rmlx_q8_quantize\" in rmlx-kv-quant"
# The pinned kernel matched, so the report must not send the reader after it.
expect_no_report "\"${CENSUS_KERNEL}\" device load"

# The same total in a different crate is a change in what the suite does, not a
# match: the tally is keyed on the crate too.
new_case census_hits_moved_to_another_crate || exit 1
classify "${CASE_ROOT}" rmlx-kv-quant kv_gpu_alpha
classify "${CASE_ROOT}" rmlx-models models_gpu_alpha
census_pin "${CASE_ROOT}" 4 models_gpu_alpha rmlx-models
census_log "${CASE_ROOT}" rmlx-kv-quant 4
crate_log "${CASE_ROOT}" rmlx-models 0 <<'LOG'
Metal GPU Validation Enabled
running 1 test
test models::gpu_alpha ... ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.00s
LOG
run_case "${CASE_ROOT}"
expect_status 1
expect_report "not pinned: 4 device load \"${CENSUS_KERNEL}\" in rmlx-kv-quant"
expect_report "no longer fires: \"${CENSUS_KERNEL}\" device load in rmlx-models"

# A count above the expectation is a hit the validated analysis does not cover.
new_case census_count_up || exit 1
classify "${CASE_ROOT}" rmlx-kv-quant kv_gpu_alpha
census_pin "${CASE_ROOT}" 4
census_log "${CASE_ROOT}" rmlx-kv-quant 5
run_case "${CASE_ROOT}"
expect_status 1
expect_report "count moved up: \"${CENSUS_KERNEL}\" device load in rmlx-kv-quant — expected 4, observed 5"

# A count BELOW it is a failure too: the pin is then stale, and accepting it
# silently would let the census drift down one hit at a time until it fits
# whatever the tree does today.
new_case census_count_down || exit 1
classify "${CASE_ROOT}" rmlx-kv-quant kv_gpu_alpha
census_pin "${CASE_ROOT}" 4
census_log "${CASE_ROOT}" rmlx-kv-quant 3
run_case "${CASE_ROOT}"
expect_status 1
expect_report "count moved down: \"${CENSUS_KERNEL}\" device load in rmlx-kv-quant — expected 4, observed 3"

# The limit of that: a pinned test that ran and produced nothing. The tally is
# empty, so nothing in the observed set can carry this — it is only visible from
# the pin's side.
new_case census_pinned_kernel_silent || exit 1
classify "${CASE_ROOT}" rmlx-kv-quant kv_gpu_alpha
census_pin "${CASE_ROOT}" 4
census_log "${CASE_ROOT}" rmlx-kv-quant 0
run_case "${CASE_ROOT}"
expect_status 1
expect_report "no longer fires: \"${CENSUS_KERNEL}\" device load in rmlx-kv-quant — expected 4, observed 0"

# Narrowing does not excuse an entry whose test the narrowing KEPT. This is the
# case a population-blind exemption gets wrong: the most targeted run of all
# would be the one that cannot enforce the pin.
new_case census_narrowed_run_enforces_a_selected_entry || exit 1
classify "${CASE_ROOT}" rmlx-kv-quant kv_gpu_alpha
classify "${CASE_ROOT}" rmlx-models models_gpu_alpha
census_pin "${CASE_ROOT}" 4
census_log "${CASE_ROOT}" rmlx-kv-quant 0
run_case "${CASE_ROOT}" --crate rmlx-kv-quant
expect_status 1
expect_report "no longer fires: \"${CENSUS_KERNEL}\" device load in rmlx-kv-quant — expected 4, observed 0"

# An entry whose test the narrowing dropped contributes 0 to the expectation,
# and the run says so rather than claiming a match it did not check.
new_case census_unselected_entry_is_not_counted || exit 1
classify "${CASE_ROOT}" rmlx-kv-quant kv_gpu_alpha
classify "${CASE_ROOT}" rmlx-models models_gpu_alpha
census_pin "${CASE_ROOT}" 4 models_gpu_alpha rmlx-models
census_log "${CASE_ROOT}" rmlx-kv-quant 0
run_case "${CASE_ROOT}" --crate rmlx-kv-quant
expect_status 0
expect_out "not enforced in full"
expect_out "models_gpu_alpha was not selected"
expect_no_out "census matches the pin"
expect_no_out "clean"

# The other way an entry legitimately contributes nothing: its test announced a
# skip, for want of the model it needs. Observed from the test's own notice, not
# inferred from what is on disk — a machine can hold the directory and still
# skip, and can skip while holding it.
new_case census_skipped_entry_is_not_counted || exit 1
classify "${CASE_ROOT}" rmlx-kv-quant kv_gpu_alpha
census_pin "${CASE_ROOT}" 4
census_log "${CASE_ROOT}" rmlx-kv-quant 0 kv_gpu_alpha
run_case "${CASE_ROOT}"
expect_status 0
expect_out "not enforced in full"
expect_out "kv_gpu_alpha skipped"
expect_no_out "census matches the pin"
expect_no_out "clean"

# And a skip does not become a licence: hits from a test that reported skipping
# are above an expectation of zero.
new_case census_skipped_entry_that_still_hit || exit 1
classify "${CASE_ROOT}" rmlx-kv-quant kv_gpu_alpha
census_pin "${CASE_ROOT}" 4
census_log "${CASE_ROOT}" rmlx-kv-quant 4 kv_gpu_alpha
run_case "${CASE_ROOT}"
expect_status 1
expect_report "count moved up: \"${CENSUS_KERNEL}\" device load in rmlx-kv-quant — expected 0, observed 4"

# Two entries on one kernel, one of which ran: the expectation is the one that
# ran, and the run reports both what it checked and what it did not.
new_case census_partial_expectation || exit 1
classify "${CASE_ROOT}" rmlx-kv-quant kv_gpu_alpha kv_gpu_beta
{
    census_pin_line "${CASE_ROOT}" 4 kv_gpu_alpha rmlx-kv-quant
    census_pin_line "${CASE_ROOT}" 6 kv_gpu_beta rmlx-kv-quant
} >"${CASE_ROOT}/scripts/gpu_validation_census.txt"
census_log "${CASE_ROOT}" rmlx-kv-quant 4 kv_gpu_beta 2
run_case "${CASE_ROOT}"
expect_status 0
expect_out "4 device load \"${CENSUS_KERNEL}\" in rmlx-kv-quant"
expect_out "kv_gpu_alpha = 4"
expect_out "kv_gpu_beta skipped"
expect_no_out "census matches the pin"

# A store from a pinned kernel is corruption outright, and the pin's counts say
# nothing about it. It fails even while every pinned load matches.
new_case census_store_on_a_pinned_kernel || exit 1
classify "${CASE_ROOT}" rmlx-kv-quant kv_gpu_alpha
census_pin "${CASE_ROOT}" 4
crate_log "${CASE_ROOT}" rmlx-kv-quant 0 <<LOG
Metal GPU Validation Enabled
running 1 test
test kv::gpu_alpha ... okInvalid device load at offset 4096, executing kernel function: "${CENSUS_KERNEL}"Invalid device load at offset 4160, executing kernel function: "${CENSUS_KERNEL}"Invalid device load at offset 4224, executing kernel function: "${CENSUS_KERNEL}"Invalid device load at offset 4288, executing kernel function: "${CENSUS_KERNEL}"Invalid device store at offset 8192, executing kernel function: "${CENSUS_KERNEL}"
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.00s
LOG
run_case "${CASE_ROOT}"
expect_status 1
expect_report "never accepted: 1 device store \"${CENSUS_KERNEL}\" in rmlx-kv-quant"

# And the pin cannot be edited into accepting one.
new_case census_pin_naming_a_store_is_refused || exit 1
classify "${CASE_ROOT}" rmlx-kv-quant kv_gpu_alpha
pin "${CASE_ROOT}" <<PIN
${CENSUS_KERNEL} | device store | 1 | rmlx-kv-quant | kv_gpu_alpha | validated benign
PIN
crate_log "${CASE_ROOT}" rmlx-kv-quant 0 <<LOG
Metal GPU Validation Enabled
running 1 test
test kv::gpu_alpha ... okInvalid device store at offset 8192, executing kernel function: "${CENSUS_KERNEL}"
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.00s
LOG
run_case "${CASE_ROOT}"
expect_status 1
expect_report "a store is never pinnable"

# A malformed entry is refused rather than skipped. A dropped line would turn
# its kernel's hits into unpinned ones on the next run, sending the reader after
# a delta the file only appears to cover.
new_case census_pin_line_missing_fields_is_refused || exit 1
classify "${CASE_ROOT}" rmlx-kv-quant kv_gpu_alpha
pin "${CASE_ROOT}" <<PIN
${CENSUS_KERNEL} | device load | 4 | rmlx-kv-quant
PIN
census_log "${CASE_ROOT}" rmlx-kv-quant 4
run_case "${CASE_ROOT}"
expect_status 1
expect_report "line 1: expected 6 fields"

new_case census_pin_with_a_bad_count_is_refused || exit 1
classify "${CASE_ROOT}" rmlx-kv-quant kv_gpu_alpha
pin "${CASE_ROOT}" <<PIN
${CENSUS_KERNEL} | device load | some | rmlx-kv-quant | kv_gpu_alpha | validated benign
PIN
census_log "${CASE_ROOT}" rmlx-kv-quant 4
run_case "${CASE_ROOT}"
expect_status 1
expect_report "count 'some' is not a positive integer"

# An entry naming a test that no longer exists would be dropped from every
# expectation for ever, and could then never fail. Refused.
new_case census_pin_naming_an_unknown_test_is_refused || exit 1
classify "${CASE_ROOT}" rmlx-kv-quant kv_gpu_alpha
pin "${CASE_ROOT}" <<PIN
${CENSUS_KERNEL} | device load | 4 | rmlx-kv-quant | kv_gpu_renamed | validated benign
PIN
census_log "${CASE_ROOT}" rmlx-kv-quant 4
run_case "${CASE_ROOT}"
expect_status 1
expect_report "has no classified GPU test 'kv_gpu_renamed'"

# One entry per kernel, kind and test: with two, which count the tally is
# compared against depends on parse order, and the second silently decides it.
new_case census_pin_with_a_duplicate_entry_is_refused || exit 1
classify "${CASE_ROOT}" rmlx-kv-quant kv_gpu_alpha
pin "${CASE_ROOT}" <<PIN
${CENSUS_KERNEL} | device load | 4 | rmlx-kv-quant | kv_gpu_alpha | validated benign
${CENSUS_KERNEL} | device load | 7 | rmlx-kv-quant | kv_gpu_alpha | validated benign
PIN
census_log "${CASE_ROOT}" rmlx-kv-quant 4
run_case "${CASE_ROOT}"
expect_status 1
expect_report "is pinned twice"

# A pin file that is not there is not the same as one that accepts nothing, and
# reading it as such would let a deleted file pass unremarked.
new_case census_missing_pin_file_is_named || exit 1
classify "${CASE_ROOT}" rmlx-kv-quant kv_gpu_alpha
rm -f "${CASE_ROOT}/scripts/gpu_validation_census.txt"
census_log "${CASE_ROOT}" rmlx-kv-quant 4
run_case "${CASE_ROOT}"
expect_status 1
expect_report "not found"

# An empty pin accepts nothing. This is also the state of a tree that has no
# census to carry, where every hit is new by definition.
new_case census_empty_pin_with_hits || exit 1
classify "${CASE_ROOT}" rmlx-kv-quant kv_gpu_alpha
census_log "${CASE_ROOT}" rmlx-kv-quant 4
run_case "${CASE_ROOT}"
expect_status 1
expect_report "not pinned: 4 device load \"${CENSUS_KERNEL}\" in rmlx-kv-quant"

# The TRACKED pin, parsed by the real classifier's population. Every case above
# writes its own file, so without this one a committed pin could be malformed,
# or name a test that has since been renamed, and nothing in `make ci` would say
# so — it would surface only on a machine with a GPU. Every entry is reported as
# skipped here, which is what proves the file parsed into entries at all.
new_case tracked_census_pin || exit 1
cp "${ROOT}/scripts/gpu_validation_census.txt" "${CASE_ROOT}/scripts/gpu_validation_census.txt" || exit 1
bash "${ROOT}/scripts/check_gpu_tests_ignored.sh" --list >"${CASE_ROOT}/classified" || exit 1
while IFS= read -r tracked_crate; do
    [ -n "${tracked_crate}" ] || continue
    {
        echo 'Metal GPU Validation Enabled'
        awk -F'|' -v c="${tracked_crate}" '
            /^[[:space:]]*#/ || /^[[:space:]]*$/ { next }
            { gsub(/^[[:space:]]+|[[:space:]]+$/, "", $4)
              gsub(/^[[:space:]]+|[[:space:]]+$/, "", $5)
              if ($4 == c) print "test " $5 " ... SKIP " $5 ": no snapshot in this fixture" }
        ' "${CASE_ROOT}/scripts/gpu_validation_census.txt"
        printf 'test result: ok. %s passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.00s\n' \
            "$(awk -F'\t' -v c="${tracked_crate}" '$1 == c' "${CASE_ROOT}/classified" | grep -c '')"
    } | crate_log "${CASE_ROOT}" "${tracked_crate}" 0
done < <(cut -f1 "${CASE_ROOT}/classified" | sort -u)
run_case "${CASE_ROOT}"
expect_status 0
expect_out "not enforced in full"
expect_no_out "line 1:"
expect_no_out "expected 6 fields"
expect_no_out "is not a positive integer"
expect_no_out "is pinned twice"
expect_no_out "never pinnable"
expect_no_out "has no classified GPU test"
expect_no_out "not found"

# ---------------------------------------------------------------------------
# A cell that stood down is reported as itself. libtest prints `ok` for a test
# that returned before asserting anything, so without this the operator cannot
# tell a suite that held from one that was never asked — the shape that let a
# green `ci-perf` be quoted over speculative answer-equivalence gates that had
# not run.
#
# The reason is asserted, not just the name: it is where the variable that would
# arm the cell is named, and a report that drops it sends the reader hunting.
new_case stand_down_is_named_with_its_reason || exit 1
classify "${CASE_ROOT}" rmlx-models spec_alpha spec_beta
crate_log "${CASE_ROOT}" rmlx-models 0 <<'LOG'
Metal GPU Validation Enabled
running 2 tests
test spec::alpha ... SKIP spec_alpha: RMLX_DRAFT_TEST_MODEL is unset and this pair's drafter is not resolved by slug
ok
test spec::beta ... ok
test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.00s
LOG
run_case "${CASE_ROOT}"
expect_status 0
expect_out "stood down"
expect_out "rmlx-models spec_alpha: RMLX_DRAFT_TEST_MODEL is unset"
expect_out "INCOMPLETE: 1 selected GPU test(s) stood down"
expect_no_out "spec_beta:"

# ---------------------------------------------------------------------------
# The pair that makes the report falsifiable: the same crate, the same two
# tests, run once with a stand-down and once without. The final line an operator
# quotes must differ between them — if it does not, the notice above is
# decoration.
new_case stand_down_changes_the_final_line || exit 1
classify "${CASE_ROOT}" rmlx-models spec_alpha spec_beta
crate_log "${CASE_ROOT}" rmlx-models 0 <<'LOG'
Metal GPU Validation Enabled
running 2 tests
test spec::alpha ... SKIP spec_alpha: RMLX_O_MODELS_ROOT does not hold a runnable snapshot
ok
test spec::beta ... ok
test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.00s
LOG
run_case "${CASE_ROOT}"
expect_status 0
stood_down_line="$(printf '%s\n' "${OUT}" | grep '^OK:')"
crate_log "${CASE_ROOT}" rmlx-models 0 <<'LOG'
Metal GPU Validation Enabled
running 2 tests
test spec::alpha ... ok
test spec::beta ... ok
test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.00s
LOG
run_case "${CASE_ROOT}"
expect_status 0
ran_line="$(printf '%s\n' "${OUT}" | grep '^OK:')"
[ "${stood_down_line}" = "${ran_line}" ] &&
    fail "a run that stood a test down ends in the same line as one that ran it: ${ran_line}"
case "${ran_line}" in
    *INCOMPLETE*) fail "a run with nothing stood down must not be marked INCOMPLETE: ${ran_line}" ;;
esac
expect_no_out "stood down"

# ---------------------------------------------------------------------------
# A stand-down is not a property of the Metal validation layer, and harvesting
# it only when that layer is on would leave `--no-shader-validation` reporting
# an unqualified pass over cells that never ran.
new_case stand_down_survives_uninstrumented || exit 1
classify "${CASE_ROOT}" rmlx-models spec_alpha
crate_log "${CASE_ROOT}" rmlx-models 0 <<'LOG'
running 1 test
test spec::alpha ... SKIP spec_alpha: RMLX_DRAFT_TEST_MODEL is unset
ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.00s
LOG
run_case "${CASE_ROOT}" --no-shader-validation
expect_status 0
expect_out "rmlx-models spec_alpha: RMLX_DRAFT_TEST_MODEL is unset"
expect_out "INCOMPLETE: 1 selected GPU test(s) stood down"

# ---------------------------------------------------------------------------
# A notice that names no test cannot be attributed. It is counted rather than
# dropped — a report that omits it claims to have seen every stand-down when it
# has not — and it must not be attributed to whichever test happened to be
# nearby, which would be worse than saying nothing.
new_case unattributed_stand_down_is_counted || exit 1
classify "${CASE_ROOT}" rmlx-models spec_alpha
crate_log "${CASE_ROOT}" rmlx-models 0 <<'LOG'
Metal GPU Validation Enabled
running 1 test
test spec::alpha ... SKIP: RMLX_TEST_MODEL_QWEN36 not set
ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.00s
LOG
run_case "${CASE_ROOT}"
expect_status 0
expect_out "1 further stand-down notice(s) named no test"
expect_out "INCOMPLETE: 0 selected GPU test(s) stood down"
expect_no_out "rmlx-models spec_alpha:"

# ---------------------------------------------------------------------------
# A notice whose name is not a test is not an attribution either. A suite that
# announces itself by file or helper name passes the pattern and names nothing
# a libtest filter reaches, so listing it would put a name in the report that
# the reader cannot run. It is counted with the nameless ones.
new_case named_notice_that_is_not_a_test_is_not_attributed || exit 1
classify "${CASE_ROOT}" rmlx-models spec_alpha
crate_log "${CASE_ROOT}" rmlx-models 0 <<'LOG'
Metal GPU Validation Enabled
running 1 test
test loader::published ... SKIP dflash2_loader: the published checkpoint is not on this machine
ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.00s
LOG
run_case "${CASE_ROOT}"
expect_status 0
expect_out "1 further stand-down notice(s)"
expect_out "INCOMPLETE: 0 selected GPU test(s) stood down"
expect_no_out "rmlx-models dflash2_loader:"

# ---------------------------------------------------------------------------
# THE HALVES
#
# The suite is partitioned so a change can pay for the part of it that guards
# what the change touched. A partition is only a gate if three things hold, and
# each is a case below: no classified test falls out of every half, a half's
# census expectation is its own slice of the one pin and not the whole of it,
# and a stand-down inside a half is still a stand-down. The third is the one a
# split can quietly break — a test the selection drops is merely `not enforced
# in full`, which carries no INCOMPLETE, so a half that turned a stand-down into
# a non-selection would buy a green run by not asking.
#
# The partition itself comes from `scripts/gpu_test_halves.sh`, which is the one
# producer of it. Every case here stubs that producer, so the case states its
# own partition; the last case runs the real one over a fixture tree, which is
# what holds the rule to the tree instead of to a list of test names.

# A half runs its own tests and no others.
new_case half_runs_only_its_own_tests || exit 1
classify "${CASE_ROOT}" rmlx-kv-quant kv_gpu_alpha kv_gpu_beta
classify "${CASE_ROOT}" rmlx-models models_gpu_alpha models_gpu_beta
halves "${CASE_ROOT}" <<'HALVES'
codec	rmlx-kv-quant	kv_gpu_alpha
codec	rmlx-kv-quant	kv_gpu_beta
rest	rmlx-models	models_gpu_alpha
rest	rmlx-models	models_gpu_beta
HALVES
crate_log "${CASE_ROOT}" rmlx-kv-quant 0 <<'LOG'
Metal GPU Validation Enabled
running 2 tests
test kv::gpu_alpha ... ok
test kv::gpu_beta ... ok
test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.00s
LOG
run_case "${CASE_ROOT}" --half codec
expect_status 0
expect_out "rmlx-kv-quant (2 GPU tests)"
expect_out "OK: 2 GPU tests passed"
expect_out "codec half"
expect_ran "kv_gpu_alpha"
expect_ran "kv_gpu_beta"
expect_did_not_run "models_gpu_alpha"
# The stub cargo has no canned log for rmlx-models and exits 99 if asked for
# one, so reaching that crate would be loud. This asserts the quieter half of
# the same property: the crate is not even visited.
expect_no_out "rmlx-models ("

# The complement runs the rest, and the two together are the whole suite. A
# split whose halves overlap or leave a gap reports a total here that is not the
# unnarrowed run's.
new_case half_union_equals_the_unnarrowed_run || exit 1
classify "${CASE_ROOT}" rmlx-kv-quant kv_gpu_alpha kv_gpu_beta
classify "${CASE_ROOT}" rmlx-models models_gpu_alpha models_gpu_beta
halves "${CASE_ROOT}" <<'HALVES'
codec	rmlx-kv-quant	kv_gpu_alpha
codec	rmlx-kv-quant	kv_gpu_beta
rest	rmlx-models	models_gpu_alpha
rest	rmlx-models	models_gpu_beta
HALVES
crate_log "${CASE_ROOT}" rmlx-kv-quant 0 <<'LOG'
Metal GPU Validation Enabled
running 2 tests
test kv::gpu_alpha ... ok
test kv::gpu_beta ... ok
test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.00s
LOG
crate_log "${CASE_ROOT}" rmlx-models 0 <<'LOG'
Metal GPU Validation Enabled
running 2 tests
test models::gpu_alpha ... ok
test models::gpu_beta ... ok
test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.00s
LOG
run_case "${CASE_ROOT}" --half codec
expect_status 0
expect_out "OK: 2 GPU tests passed"
run_case "${CASE_ROOT}" --half rest
expect_status 0
expect_out "OK: 2 GPU tests passed"
expect_out "rest half"
run_case "${CASE_ROOT}"
expect_status 0
expect_out "OK: 4 GPU tests passed"
expect_no_out "codec half"
expect_no_out "rest half"

# A classified test the producer places in no half runs under no gate at all,
# which is the shape the whole suite exists to prevent. It is a refusal, not a
# note: a run that quietly skipped it would report the same green line as one
# that ran it.
new_case half_a_test_in_no_half_is_refused || exit 1
classify "${CASE_ROOT}" rmlx-kv-quant kv_gpu_alpha kv_gpu_beta
halves "${CASE_ROOT}" <<'HALVES'
codec	rmlx-kv-quant	kv_gpu_alpha
HALVES
crate_log "${CASE_ROOT}" rmlx-kv-quant 0 <<'LOG'
Metal GPU Validation Enabled
running 1 test
test kv::gpu_alpha ... ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.00s
LOG
run_case "${CASE_ROOT}" --half codec
# The exit code alone is not the assertion here — an unimplemented `--half` also
# exits 1. The refusal must name the stranded test and must run nothing.
expect_status 1
expect_out "kv_gpu_beta"
expect_out "no half"
expect_did_not_run "kv_gpu_alpha"

# The pin stays one file. A half reads the slice of it whose tests the half
# selects, and expects exactly that slice's sum — not the whole pin, which would
# be red on every half, and not a waiver, which would be a pin that cannot fire.
# The other half's entry is silent here rather than `not enforced in full`: that
# note means an entry this run was supposed to check and could not, and an entry
# belonging to the other half is neither.
new_case half_census_slice_is_its_own_entries || exit 1
classify "${CASE_ROOT}" rmlx-kv-quant kv_gpu_alpha
classify "${CASE_ROOT}" rmlx-models models_gpu_alpha
halves "${CASE_ROOT}" <<'HALVES'
codec	rmlx-kv-quant	kv_gpu_alpha
rest	rmlx-models	models_gpu_alpha
HALVES
{
    census_pin_line "${CASE_ROOT}" 4 kv_gpu_alpha rmlx-kv-quant
    census_pin_line "${CASE_ROOT}" 6 models_gpu_alpha rmlx-models
} >"${CASE_ROOT}/scripts/gpu_validation_census.txt"
census_log "${CASE_ROOT}" rmlx-kv-quant 4
run_case "${CASE_ROOT}" --half codec
expect_status 0
expect_out "census matches the pin"
expect_out "kv_gpu_alpha = 4"
expect_no_out "not enforced in full"
expect_no_out "models_gpu_alpha"

# One crate, one kernel, one access kind, two pinned cells — and the half is the
# only thing that separates them. The runner already keys its expectation on
# (crate, kind, kernel), so a case whose two entries sit in different crates
# would pass on that keying alone and say nothing about the slice. Here both
# entries survive it, and only the half decides which is expected: 4, not 10.
new_case half_census_slice_is_the_half_not_the_crate || exit 1
classify "${CASE_ROOT}" rmlx-kv-quant kv_gpu_alpha kv_gpu_beta
halves "${CASE_ROOT}" <<'HALVES'
codec	rmlx-kv-quant	kv_gpu_alpha
rest	rmlx-kv-quant	kv_gpu_beta
HALVES
{
    census_pin_line "${CASE_ROOT}" 4 kv_gpu_alpha rmlx-kv-quant
    census_pin_line "${CASE_ROOT}" 6 kv_gpu_beta rmlx-kv-quant
} >"${CASE_ROOT}/scripts/gpu_validation_census.txt"
census_log "${CASE_ROOT}" rmlx-kv-quant 4
run_case "${CASE_ROOT}" --half codec
expect_status 0
expect_out "census matches the pin"
expect_out "kv_gpu_alpha = 4"
# The other half's entry is not this run's business: not expected, and not
# reported as an entry this run failed to check.
expect_no_out "kv_gpu_beta"
expect_no_out "not enforced in full"

# The other way a slice can go wrong, and a different mutation from the one
# above: matched on the test name alone. Two crates carry a cell of the same
# name — seven names are defined in more than one module today — one per half.
# The case above is blind to this one (its two entries share a crate, so a
# name-only match and a half match agree), and this one is blind to that one (its
# two entries are separated by the runner's own crate key before the half is
# consulted). Both are kept.
new_case half_census_slice_keys_on_the_crate_column || exit 1
classify "${CASE_ROOT}" rmlx-kv-quant kv_gpu_alpha
classify "${CASE_ROOT}" rmlx-models kv_gpu_alpha
halves "${CASE_ROOT}" <<'HALVES'
codec	rmlx-kv-quant	kv_gpu_alpha
rest	rmlx-models	kv_gpu_alpha
HALVES
census_pin "${CASE_ROOT}" 4 kv_gpu_alpha rmlx-models
census_log "${CASE_ROOT}" rmlx-kv-quant 4
run_case "${CASE_ROOT}" --half codec
expect_status 1
expect_report "not pinned: 4 device load \"${CENSUS_KERNEL}\" in rmlx-kv-quant"

# One crate declaring a cell in each half. Both halves report the same count, so
# a partition keyed on the crate rather than on the test passes every count
# assertion in this file — what separates them is WHICH cell each half asked for.
new_case half_one_crate_spans_both_halves || exit 1
classify "${CASE_ROOT}" rmlx-models a_unit_cell an_integration_cell
halves "${CASE_ROOT}" <<'HALVES'
codec	rmlx-models	a_unit_cell
rest	rmlx-models	an_integration_cell
HALVES
crate_log "${CASE_ROOT}" rmlx-models 0 <<'LOG'
Metal GPU Validation Enabled
running 1 test
test models::a_unit_cell ... ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.00s
LOG
run_case "${CASE_ROOT}" --half codec
expect_status 0
expect_out "rmlx-models (1 GPU tests)"
expect_ran "a_unit_cell"
expect_did_not_run "an_integration_cell"
run_case "${CASE_ROOT}" --half rest
expect_status 0
expect_out "rmlx-models (1 GPU tests)"
expect_ran "an_integration_cell"
expect_did_not_run "a_unit_cell"

# Property 5 at the census: one pin, three runs, and the two halves' accepted
# expectations sum to the unnarrowed run's. A slice that double-counts or drops
# an entry is invisible to every per-half case above — each of them is
# self-consistent — and shows up only here.
new_case half_census_sum_equals_the_whole || exit 1
classify "${CASE_ROOT}" rmlx-kv-quant kv_gpu_alpha kv_gpu_beta
halves "${CASE_ROOT}" <<'HALVES'
codec	rmlx-kv-quant	kv_gpu_alpha
rest	rmlx-kv-quant	kv_gpu_beta
HALVES
{
    census_pin_line "${CASE_ROOT}" 4 kv_gpu_alpha rmlx-kv-quant
    census_pin_line "${CASE_ROOT}" 6 kv_gpu_beta rmlx-kv-quant
} >"${CASE_ROOT}/scripts/gpu_validation_census.txt"
census_log "${CASE_ROOT}" rmlx-kv-quant 4
run_case "${CASE_ROOT}" --half codec
expect_status 0
expect_out "4 device load \"${CENSUS_KERNEL}\" in rmlx-kv-quant"
census_log "${CASE_ROOT}" rmlx-kv-quant 6
run_case "${CASE_ROOT}" --half rest
expect_status 0
expect_out "6 device load \"${CENSUS_KERNEL}\" in rmlx-kv-quant"
census_log "${CASE_ROOT}" rmlx-kv-quant 10 "" 2
run_case "${CASE_ROOT}"
expect_status 0
expect_out "10 device load \"${CENSUS_KERNEL}\" in rmlx-kv-quant"
expect_out "census matches the pin"

# A stand-down inside a half is still a stand-down. This is the one the split
# can launder: a test the selection never asked for is only `not enforced in
# full`, which carries no INCOMPLETE and lets `make ci-perf` print `ok`.
new_case half_stand_down_stays_incomplete || exit 1
classify "${CASE_ROOT}" rmlx-kv-quant kv_gpu_alpha
classify "${CASE_ROOT}" rmlx-models models_gpu_alpha
halves "${CASE_ROOT}" <<'HALVES'
codec	rmlx-kv-quant	kv_gpu_alpha
rest	rmlx-models	models_gpu_alpha
HALVES
census_log "${CASE_ROOT}" rmlx-kv-quant 0 kv_gpu_alpha
run_case "${CASE_ROOT}" --half codec
expect_status 0
expect_out "INCOMPLETE: 1 selected GPU test(s) stood down"
expect_out "rmlx-kv-quant kv_gpu_alpha:"

# The rule reads the tree, not a list of names. This case and the next run the
# real producer, over a fixture tree it is pointed at by argument: a crate the
# model layer stands on is codec whichever target its test is declared in, the
# model layer's own unit test is codec, its integration binary is codec when the
# binary selects a codec and rest when it does not, and a crate the model layer
# does not depend on is neither. Renaming a test moves nothing, which is what a
# producer carrying a literal list cannot do.
new_case half_rule_is_read_from_the_tree || exit 1
tree="${CASE_ROOT}/tree"
halves_fixture_tree "${tree}" || exit 1
gpu_fixture_test "${tree}/crates/rmlx-kv-quant/src/codec_tests.rs" codec_decodes
gpu_fixture_test "${tree}/crates/rmlx-models/src/arch_tests.rs" arch_forwards
gpu_fixture_test "${tree}/crates/rmlx-models/tests/pipeline.rs" pipeline_agrees
gpu_fixture_test "${tree}/crates/rmlx-models/tests/codec_sweep.rs" sweep_holds \
    'let q = rmlx_kv_quant::KvQuant::K8V8;'
gpu_fixture_test "${tree}/crates/rmlx-audio/src/asr_tests.rs" asr_transcribes
HALVES_OUT="$(bash "${ROOT}/scripts/gpu_test_halves.sh" --root "${tree}" 2>&1)"
expect_placed codec rmlx-kv-quant codec_decodes
expect_placed codec rmlx-models arch_forwards
expect_placed codec rmlx-models sweep_holds
expect_placed rest rmlx-models pipeline_agrees
expect_placed rest rmlx-audio asr_transcribes
# The same test under a different name keeps its half, because the half came
# from where the test is declared.
gpu_fixture_test "${tree}/crates/rmlx-kv-quant/src/codec_tests.rs" codec_decodes_renamed
HALVES_OUT="$(bash "${ROOT}/scripts/gpu_test_halves.sh" --root "${tree}" 2>&1)"
expect_placed codec rmlx-kv-quant codec_decodes_renamed

# The codec-name set comes from `ALL_KV_QUANTS`, not from a pattern. `FromStr` is
# a trait path under the same `KvQuant::` prefix and names no codec, so a file
# whose only mention is that one has not selected anything and stays in rest. A
# bare `KvQuant::<ident>` needle places it in codec and the half loses its
# meaning.
new_case half_a_non_codec_kv_quant_item_stays_rest || exit 1
tree="${CASE_ROOT}/tree"
halves_fixture_tree "${tree}" || exit 1
gpu_fixture_test "${tree}/crates/rmlx-models/tests/parses_a_flag.rs" flag_parses \
    'let q = <rmlx_kv_quant::KvQuant as std::str::FromStr>::from_str("none");'
gpu_fixture_test "${tree}/crates/rmlx-models/tests/pins_none.rs" none_is_pinned \
    'let q = rmlx_kv_quant::KvQuant::None;'
HALVES_OUT="$(bash "${ROOT}/scripts/gpu_test_halves.sh" --root "${tree}" 2>&1)"
expect_placed rest rmlx-models flag_parses
expect_placed rest rmlx-models none_is_pinned

# ---------------------------------------------------------------------------
# The half's marker line. It is the split's ONE structural defence — a half
# narrows the classified population in lockstep with the executed one, so no
# check inside the runner can tell a half from a complete run, and what keeps a
# half-run's record honest is that it never reads `ci-perf ok`. Nothing else in
# the tree greps that string, so without this case the marker can be replaced by
# the whole gate's and every gate stays green.
#
# `make -n`: the recipe is read, nothing is executed, no GPU is touched.
new_case half_marker_is_pinned || exit 1
OUT="$(cd "${ROOT}" && make -n ci-perf HALF=codec 2>&1)"
STATUS=$?
expect_status 0
expect_out "ci-perf codec-half ok — NOT the whole gate"
expect_no_out "ci-perf ok"

# ---------------------------------------------------------------------------
# The harness's own positive control: with nothing wrong, the same stubs produce
# a green run. Without this, every case above could be passing because the stub
# crates never ran at all.
new_case clean || exit 1
classify "${CASE_ROOT}" rmlx-kv-quant kv_gpu_alpha kv_gpu_beta
crate_log "${CASE_ROOT}" rmlx-kv-quant 0 <<'LOG'
Metal GPU Validation Enabled
running 2 tests
test kv::gpu_alpha ... ok
test kv::gpu_beta ... ok
test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.00s
LOG
run_case "${CASE_ROOT}"
expect_status 0
expect_out "OK: 2 GPU tests passed"
expect_out "shader validation clean"
expect_no_out "INCOMPLETE"
expect_no_out "ERROR:"

if [ "${failures}" -ne 0 ]; then
    echo >&2
    echo "run_gpu_tests_selftest: ${failures} assertion(s) failed." >&2
    exit 1
fi

echo "run_gpu_tests_selftest: OK — every kind of red is reported, the access mix is the one observed, the census pin accepts only what it names, and each half runs its own tests against its own slice of that pin."
