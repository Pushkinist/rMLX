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
    OUT="$(PATH="${root}/bin:${PATH}" env -u RMLX_SKIP_GPU \
        RMLX_O_MODELS_ROOT="${WORK}" bash "${root}/scripts/run_gpu_tests.sh" "$@" 2>&1)"
    STATUS=$?
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
# `Error 1` is background noise. The pin records that census exactly, so a hit
# that matches it passes and anything else — a new kernel, a count that moved in
# either direction, any store, a pinned kernel that stopped firing — is a
# failure naming the delta.
#
# Every case below writes its own pin, so none of them depends on what the tree
# happens to accept today.

CENSUS_KERNEL="mlx_qmm_stub"
CENSUS_SNAPSHOT="stub-snapshot"

# census_pin <root> <count> [snapshot] — a one-entry pin for CENSUS_KERNEL.
census_pin() {
    local root="$1" count="$2" snapshot="${3:-${CENSUS_SNAPSHOT}}"
    mkdir -p "${WORK}/${CENSUS_SNAPSHOT}"
    pin "${root}" <<PIN
# kernel | kind | count | crate | test | snapshot | reference
${CENSUS_KERNEL} | device load | ${count} | rmlx-kv-quant | kv_gpu_alpha | ${snapshot} | validated benign
PIN
}

# census_log <root> <crate> <n> — a clean libtest log carrying <n> loads of
# CENSUS_KERNEL.
census_log() {
    local root="$1" crate="$2" n="$3" i=0
    {
        echo 'Metal GPU Validation Enabled'
        echo 'running 1 test'
        while [ "${i}" -lt "${n}" ]; do
            echo "Invalid device load at offset $((4096 + i * 64)), executing kernel function: \"${CENSUS_KERNEL}\""
            i=$((i + 1))
        done
        echo 'test kv::gpu_alpha ... ok'
        echo 'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.00s'
    } | crate_log "${root}" "${crate}" 0
}

# The accepted case: the observed tally is exactly the pinned one, so the run is
# green and prints the census it accepted. Without this the pin would be a
# gate that can only fail, which is the bug it was written against.
new_case census_exact_match || exit 1
classify "${CASE_ROOT}" rmlx-kv-quant kv_gpu_alpha
census_pin "${CASE_ROOT}" 4
census_log "${CASE_ROOT}" rmlx-kv-quant 4
run_case "${CASE_ROOT}"
expect_status 0
expect_out "census matches the pin"
expect_out "4 device load \"${CENSUS_KERNEL}\""
expect_no_out "ERROR:"
# A run that accepted four invalid accesses is not a clean one, and saying so
# would put the operator back where a permanently red gate left them.
expect_no_out "shader validation clean"

# A kernel the pin does not name is a new hit, whatever the pinned ones did.
new_case census_new_kernel || exit 1
classify "${CASE_ROOT}" rmlx-kv-quant kv_gpu_alpha
census_pin "${CASE_ROOT}" 4
crate_log "${CASE_ROOT}" rmlx-kv-quant 0 <<LOG
Metal GPU Validation Enabled
running 1 test
Invalid device load at offset 4096, executing kernel function: "${CENSUS_KERNEL}"
Invalid device load at offset 4160, executing kernel function: "${CENSUS_KERNEL}"
Invalid device load at offset 4224, executing kernel function: "${CENSUS_KERNEL}"
Invalid device load at offset 4288, executing kernel function: "${CENSUS_KERNEL}"
Invalid device load at offset 8192, executing kernel function: "custom_kernel_rmlx_q8_quantize"
test kv::gpu_alpha ... ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.00s
LOG
run_case "${CASE_ROOT}"
expect_status 1
expect_report "not pinned: 1 device load \"custom_kernel_rmlx_q8_quantize\""
# The pinned kernel matched, so the report must not send the reader after it.
expect_no_report "\"${CENSUS_KERNEL}\" device load"

# A count that moved up is a hit the validated analysis does not cover.
new_case census_count_up || exit 1
classify "${CASE_ROOT}" rmlx-kv-quant kv_gpu_alpha
census_pin "${CASE_ROOT}" 4
census_log "${CASE_ROOT}" rmlx-kv-quant 5
run_case "${CASE_ROOT}"
expect_status 1
expect_report "count moved up: \"${CENSUS_KERNEL}\" device load — pinned 4, observed 5"

# A count that moved DOWN is a failure too: the pin is then stale, and accepting
# it silently would let the census drift down one hit at a time until it fits
# whatever the tree does today.
new_case census_count_down || exit 1
classify "${CASE_ROOT}" rmlx-kv-quant kv_gpu_alpha
census_pin "${CASE_ROOT}" 4
census_log "${CASE_ROOT}" rmlx-kv-quant 3
run_case "${CASE_ROOT}"
expect_status 1
expect_report "count moved down: \"${CENSUS_KERNEL}\" device load — pinned 4, observed 3"

# The limit of that: a pinned kernel that stopped firing entirely. The tally is
# empty, so nothing in the observed set can carry this — it is only visible from
# the pin's side.
new_case census_pinned_kernel_silent || exit 1
classify "${CASE_ROOT}" rmlx-kv-quant kv_gpu_alpha
census_pin "${CASE_ROOT}" 4
crate_log "${CASE_ROOT}" rmlx-kv-quant 0 <<'LOG'
Metal GPU Validation Enabled
running 1 test
test kv::gpu_alpha ... ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.00s
LOG
run_case "${CASE_ROOT}"
expect_status 1
expect_report "no longer fires: \"${CENSUS_KERNEL}\" device load — pinned 4, observed 0"

# A store from a pinned kernel is corruption outright, and the pin's counts say
# nothing about it. It fails even while every pinned load matches.
new_case census_store_on_a_pinned_kernel || exit 1
classify "${CASE_ROOT}" rmlx-kv-quant kv_gpu_alpha
census_pin "${CASE_ROOT}" 4
crate_log "${CASE_ROOT}" rmlx-kv-quant 0 <<LOG
Metal GPU Validation Enabled
running 1 test
Invalid device load at offset 4096, executing kernel function: "${CENSUS_KERNEL}"
Invalid device load at offset 4160, executing kernel function: "${CENSUS_KERNEL}"
Invalid device load at offset 4224, executing kernel function: "${CENSUS_KERNEL}"
Invalid device load at offset 4288, executing kernel function: "${CENSUS_KERNEL}"
Invalid device store at offset 8192, executing kernel function: "${CENSUS_KERNEL}"
test kv::gpu_alpha ... ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.00s
LOG
run_case "${CASE_ROOT}"
expect_status 1
expect_report "never accepted: 1 device store \"${CENSUS_KERNEL}\""

# And the pin cannot be edited into accepting one.
new_case census_pin_naming_a_store_is_refused || exit 1
classify "${CASE_ROOT}" rmlx-kv-quant kv_gpu_alpha
mkdir -p "${WORK}/${CENSUS_SNAPSHOT}"
pin "${CASE_ROOT}" <<PIN
${CENSUS_KERNEL} | device store | 1 | rmlx-kv-quant | kv_gpu_alpha | ${CENSUS_SNAPSHOT} | validated benign
PIN
crate_log "${CASE_ROOT}" rmlx-kv-quant 0 <<LOG
Metal GPU Validation Enabled
running 1 test
Invalid device store at offset 8192, executing kernel function: "${CENSUS_KERNEL}"
test kv::gpu_alpha ... ok
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
expect_report "line 1: expected 7 fields"

new_case census_pin_with_a_bad_count_is_refused || exit 1
classify "${CASE_ROOT}" rmlx-kv-quant kv_gpu_alpha
mkdir -p "${WORK}/${CENSUS_SNAPSHOT}"
pin "${CASE_ROOT}" <<PIN
${CENSUS_KERNEL} | device load | some | rmlx-kv-quant | kv_gpu_alpha | ${CENSUS_SNAPSHOT} | validated benign
PIN
census_log "${CASE_ROOT}" rmlx-kv-quant 4
run_case "${CASE_ROOT}"
expect_status 1
expect_report "count 'some' is not a positive integer"

# One entry per kernel and kind: with two, which count the tally is compared
# against depends on parse order, and the second silently decides it.
new_case census_pin_with_a_duplicate_entry_is_refused || exit 1
classify "${CASE_ROOT}" rmlx-kv-quant kv_gpu_alpha
mkdir -p "${WORK}/${CENSUS_SNAPSHOT}"
pin "${CASE_ROOT}" <<PIN
${CENSUS_KERNEL} | device load | 4 | rmlx-kv-quant | kv_gpu_alpha | ${CENSUS_SNAPSHOT} | validated benign
${CENSUS_KERNEL} | device load | 7 | rmlx-kv-quant | kv_gpu_alpha | ${CENSUS_SNAPSHOT} | validated benign
PIN
census_log "${CASE_ROOT}" rmlx-kv-quant 4
run_case "${CASE_ROOT}"
expect_status 1
expect_report "is pinned twice"

# An empty pin accepts nothing. This is also the state of a tree that has no
# census to carry, where every hit is new by definition.
new_case census_empty_pin_with_hits || exit 1
classify "${CASE_ROOT}" rmlx-kv-quant kv_gpu_alpha
pin "${CASE_ROOT}" <<'PIN'
# kernel | kind | count | crate | test | snapshot | reference
PIN
census_log "${CASE_ROOT}" rmlx-kv-quant 4
run_case "${CASE_ROOT}"
expect_status 1
expect_report "not pinned: 4 device load \"${CENSUS_KERNEL}\""

# A narrowed run visits a subset of the population, so it observes fewer hits by
# construction and its shortfall says nothing about the pin. Enforcing the
# downward direction there would make every `CRATE=`-narrowed iteration red,
# which is the failure this whole mechanism exists to remove.
new_case census_narrowed_run_does_not_enforce_downward || exit 1
classify "${CASE_ROOT}" rmlx-kv-quant kv_gpu_alpha
classify "${CASE_ROOT}" rmlx-models models_gpu_alpha
census_pin "${CASE_ROOT}" 4
crate_log "${CASE_ROOT}" rmlx-kv-quant 0 <<'LOG'
Metal GPU Validation Enabled
running 1 test
test kv::gpu_alpha ... ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.00s
LOG
run_case "${CASE_ROOT}" --crate rmlx-kv-quant
expect_status 0
expect_out "not enforced downward"

# The exemption is one-directional: a narrowed run still fails on a count that
# moved up, and on a kernel the pin does not name.
new_case census_narrowed_run_fails_on_excess || exit 1
classify "${CASE_ROOT}" rmlx-kv-quant kv_gpu_alpha
classify "${CASE_ROOT}" rmlx-models models_gpu_alpha
census_pin "${CASE_ROOT}" 3
census_log "${CASE_ROOT}" rmlx-kv-quant 4
run_case "${CASE_ROOT}" --crate rmlx-kv-quant
expect_status 1
expect_report "count moved up: \"${CENSUS_KERNEL}\" device load — pinned 3, observed 4"

new_case census_narrowed_run_fails_on_an_unpinned_kernel || exit 1
classify "${CASE_ROOT}" rmlx-kv-quant kv_gpu_alpha
classify "${CASE_ROOT}" rmlx-models models_gpu_alpha
census_pin "${CASE_ROOT}" 4
crate_log "${CASE_ROOT}" rmlx-kv-quant 0 <<'LOG'
Metal GPU Validation Enabled
running 1 test
Invalid device load at offset 8192, executing kernel function: "custom_kernel_rmlx_q8_quantize"
test kv::gpu_alpha ... ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.00s
LOG
run_case "${CASE_ROOT}" --crate rmlx-kv-quant
expect_status 1
expect_report "not pinned: 1 device load \"custom_kernel_rmlx_q8_quantize\""

# The other way an entry legitimately stops firing: the snapshot that reaches it
# is not on this machine. The suite's own contract is that a model-gated cell
# skips and counts as passed, so a developer without the weights must not be
# told the pin is stale.
new_case census_missing_snapshot_does_not_enforce_downward || exit 1
classify "${CASE_ROOT}" rmlx-kv-quant kv_gpu_alpha
census_pin "${CASE_ROOT}" 4 a-snapshot-this-host-does-not-have
crate_log "${CASE_ROOT}" rmlx-kv-quant 0 <<'LOG'
Metal GPU Validation Enabled
running 1 test
test kv::gpu_alpha ... ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.00s
LOG
run_case "${CASE_ROOT}"
expect_status 0
expect_out "not enforced downward"

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

echo "run_gpu_tests_selftest: OK — every kind of red is reported, the access mix is the one observed, and the census pin accepts only what it names."
