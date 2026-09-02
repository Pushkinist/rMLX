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
# That is a property of the reporting code, not of the GPU, so it is checked
# here against stubs: a stub `cargo` replays a canned libtest log per crate and
# a stub classifier names the population. No Metal device, no snapshot, no
# compile. The runner is copied into a throwaway root per case rather than
# reimplemented, so this file cannot drift from what the gate runs.
#
# Every case asserts the REASON — the strings an operator triages a red gate
# from, taken only from the final report block, not from the interleaved `tee`
# output the report is supposed to summarise — so a runner that exits 1 while
# reporting the wrong half of the run still fails here.
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

# new_case <name> — build a throwaway repo root: a copy of the runner, a stub
# classifier reading this case's population, and a stub PATH. Sets CASE_ROOT.
# Not a command substitution: the case name has to reach the assertions below,
# and a subshell would drop it.
new_case() {
    CASE="$1"
    CASE_ROOT="${WORK}/$1"
    local root="${CASE_ROOT}"
    mkdir -p "${root}/scripts" "${root}/bin" "${root}/logs" || return 1
    cp "${RUNNER}" "${root}/scripts/run_gpu_tests.sh" || return 1
    : >"${root}/classified"

    cat >"${root}/scripts/check_gpu_tests_ignored.sh" <<STUB
#!/usr/bin/env bash
cat "${root}/classified"
STUB

    # The GPU-free preconditions are not what this file exercises, and one of
    # them reads the whole host: stub the process check so a live MLX server
    # cannot decide the outcome of a reporting test, and answer the shader
    # validation canary so the detector's positive control passes.
    cat >"${root}/bin/pgrep" <<'STUB'
#!/usr/bin/env bash
exit 1
STUB

    cat >"${root}/bin/cargo" <<STUB
#!/usr/bin/env bash
set -u
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

    chmod +x "${root}/bin/pgrep" "${root}/bin/cargo" || return 1
}

# classify <root> <crate> <fn>... — name the GPU tests the classifier reports.
classify() {
    local root="$1" crate="$2" fn
    shift 2
    for fn in "$@"; do
        printf '%s\t%s\n' "${crate}" "${fn}" >>"${root}/classified"
    done
}

# crate_log <root> <crate> <cargo-exit-code> — canned libtest log on stdin.
crate_log() {
    local root="$1" crate="$2" rc="$3"
    cat >"${root}/logs/${crate}.log"
    printf '%s\n' "${rc}" >"${root}/logs/${crate}.rc"
}

# run_case <root> — run this case's runner; set OUT, STATUS, REPORT and MIX.
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
    OUT="$(PATH="${root}/bin:${PATH}" env -u RMLX_SKIP_GPU \
        RMLX_O_MODELS_ROOT="${WORK}" bash "${root}/scripts/run_gpu_tests.sh" 2>&1)"
    STATUS=$?
    REPORT="$(printf '%s\n' "${OUT}" | awk '
        /^ERROR: Metal shader validation reported invalid memory access:/ { seen = 1 }
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

echo "run_gpu_tests_selftest: OK — every kind of red is reported, and the access mix is the one observed."
