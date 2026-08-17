#!/usr/bin/env bash
# scripts/eval_lock_stress.sh — drive the evaluation-lock reproducer across many
# FRESH PROCESSES and fail on any non-zero exit.
#
# WHY A DRIVER AND NOT JUST A TEST
#   `concurrent_first_eval_reproducer` is probabilistic: without the lock it
#   fails about 1 run in 12 (15/180 measured). Running it once — which is what
#   `make ci` does — is therefore weak evidence in the green direction. Its
#   power comes only from repetition, and specifically from repetition across
#   *processes*: the defect is an unsynchronised insert into a process-global
#   hash map, and the map rehashes hardest while it is still growing from
#   empty. Once a process has populated it, later bursts in that same process
#   mostly find existing entries and stop inserting. So looping inside the test
#   buys almost nothing; re-exec does.
#
#   At the measured ~8% per-run rate, N=60 gives ~99.3% detection and N=100
#   ~99.98%. The default below is 60.
#
# WHY IT COUNTS TIMEOUTS AS FAILURES
#   Corruption does not always segfault. Three failure shapes were observed:
#   SIGSEGV, SIGTRAP, and an infinite spin (a bucket chain that became
#   circular). The last one hangs forever and would otherwise stall the caller
#   rather than reporting, so every run is bounded and a timeout is a failure.
#
# WHAT THIS CANNOT REACH
#   Same scope as the test it drives: the CPU evaluation path only. It says
#   nothing about concurrent GPU evaluation, and nothing about races in lazy
#   graph construction. It is also a *statistical* instrument — a clean run at
#   N=60 bounds the failure rate to roughly <5% at 95% confidence, it does not
#   prove zero.
#
# USAGE
#   bash scripts/eval_lock_stress.sh [RUNS] [TIMEOUT_SECS]
#   make eval-lock-stress RUNS=100

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS="${1:-${RUNS:-60}}"
TMO="${2:-${TIMEOUT_SECS:-60}}"
TEST_NAME="lib_tests::concurrent_first_eval_reproducer"

cd "${REPO_ROOT}"

echo "eval-lock-stress: building rmlx-mlx test binary"

# Take the path cargo reports for the binary it just built, rather than globbing
# target/deps — a stale artifact from an earlier build must never be the thing
# under test. Cargo prints `Executable unittests src/lib.rs (<path>)` on stderr.
build_out="$(cargo test -p rmlx-mlx --lib --no-run 2>&1)" || {
    echo "${build_out}" >&2
    echo "eval-lock-stress: build failed" >&2
    exit 1
}
BIN="$(printf '%s\n' "${build_out}" \
    | sed -n 's/^ *Executable unittests .*(\(.*\))$/\1/p' | tail -1)"

if [ -z "${BIN}" ] || [ ! -x "${BIN}" ]; then
    echo "eval-lock-stress: could not resolve the rmlx-mlx test binary from cargo output" >&2
    printf '%s\n' "${build_out}" >&2
    exit 1
fi

# A timeout is mandatory: one of the failure shapes is an infinite spin.
TIMEOUT_BIN=""
for cand in timeout gtimeout; do
    if command -v "${cand}" >/dev/null 2>&1; then TIMEOUT_BIN="$(command -v "${cand}")"; break; fi
done
if [ -z "${TIMEOUT_BIN}" ]; then
    echo "eval-lock-stress: need 'timeout' or 'gtimeout' (brew install coreutils) — a hang is one of the failure shapes and must be bounded" >&2
    exit 1
fi

# Anti-vacuity. libtest exits 0 when a filter selects nothing — it prints
# "0 passed; 0 failed; N filtered out" and is, to `$?`, indistinguishable from a
# clean run. Output here goes to /dev/null and only the exit code is read, so a
# rename, a move out of `lib_tests`, or a change to the `#[ignore]` status would
# otherwise yield "OK: 60/60 clean" having run the reproducer zero times.
selected="$("${BIN}" --exact "${TEST_NAME}" --ignored --list 2>/dev/null \
    | grep -c ': test$' || true)"
if [ "${selected}" != "1" ]; then
    echo "eval-lock-stress: filter '${TEST_NAME}' selected ${selected} tests, expected exactly 1." >&2
    echo "  The reproducer was renamed, moved, or is no longer #[ignore]d." >&2
    echo "  Update TEST_NAME in this script — a filter that matches nothing" >&2
    echo "  would make every run exit 0 without executing anything." >&2
    exit 1
fi

echo "eval-lock-stress: ${BIN}"
echo "eval-lock-stress: ${RUNS} fresh processes, ${TMO}s cap each"

ok=0; crash=0; hang=0; other=0
for i in $(seq 1 "${RUNS}"); do
    set +e
    "${TIMEOUT_BIN}" -s KILL "${TMO}" "${BIN}" --exact "${TEST_NAME}" \
        --ignored --test-threads=1 >/dev/null 2>&1
    rc=$?
    set -e
    case "${rc}" in
        0)       ok=$((ok+1)) ;;
        124|137) hang=$((hang+1)); echo "  run ${i}: HANG (rc=${rc})" ;;
        *)       if [ "${rc}" -gt 128 ]; then
                     crash=$((crash+1)); echo "  run ${i}: signal $((rc-128))"
                 else
                     other=$((other+1)); echo "  run ${i}: exit ${rc}"
                 fi ;;
    esac
done

fails=$(( crash + hang + other ))
echo "eval-lock-stress: runs=${RUNS} ok=${ok} crashes=${crash} hangs=${hang} other=${other}"

if [ "${fails}" -ne 0 ]; then
    echo "eval-lock-stress: FAILED — ${fails}/${RUNS} runs did not exit cleanly." >&2
    echo "  Evaluation is no longer serialised, or the lock no longer covers" >&2
    echo "  every path that reaches mlx::core::eval_impl. See docs/FFI.md." >&2
    exit 1
fi

echo "OK: ${RUNS}/${RUNS} clean."
