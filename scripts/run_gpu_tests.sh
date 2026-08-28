#!/usr/bin/env bash
# scripts/run_gpu_tests.sh — execute the GPU (Metal-context) tests that
# `scripts/check_gpu_tests_ignored.sh` mandates be `#[ignore]`d, serialized.
#
# WHY
#   Every test reaching `Device::Gpu` is REQUIRED to carry `#[ignore]`, because a
#   shared Metal context driven from parallel `cargo test` threads aborts the
#   whole test binary. `make ci` enforces that rule. Nothing ever ran the tests
#   it mandates: `make test` is a bare `cargo test --workspace` (no `--ignored`),
#   and the hosted CI runs no tests and has no Metal. The result was a category
#   of test that is mandatory to write, forbidden to run automatically, and
#   therefore never run — every KV codec's on-GPU decode correctness lives in
#   that category, and tests in it have gone red on `main` and stayed red while
#   every gate reported green.
#
#   `--test-threads=1` is what makes running them safe: the Metal context is
#   driven from one thread at a time, so the abort the `#[ignore]` rule protects
#   against cannot happen.
#
# WHICH TESTS
#   Exactly the compliant set from `check_gpu_tests_ignored.sh --list` — the
#   NAMED `#[test]` fns that reach `Device::Gpu` AND carry `#[ignore]`. Deriving
#   the population from the enforcing gate's own classifier is the point: a
#   second, hand-maintained list would drift, and the rule would end up
#   mandating `#[ignore]` on tests this runner never visits.
#
#   "Named" is the one exclusion, and it is structural: a macro-generated test
#   has no fn name until the compiler expands it, so no substring filter can
#   select it and the coverage check below would read the whole crate as
#   under-matched. Those tests are still ENFORCED by the classifier, at their
#   `macro_rules!` body; the enforcing run prints the excluded set. See
#   docs/TESTING.md.
#
#   It deliberately does NOT run every `#[ignore]` test. Plenty are ignored for
#   reasons that have nothing to do with Metal — live network access, a missing
#   cargo feature, `ignore`-marked doc-comment pseudo-code — and sweeping those
#   in would make this gate permanently red for reasons it cannot speak to.
#
# FAIL-CLOSED
#   A gate that silently runs nothing passes everything. This one refuses to
#   report OK unless it executed every test the classifier named: an empty
#   classification, a `--filter` that selects nothing, and a crate that executed
#   FEWER tests than were classified for it are all errors, not green runs. The
#   last one is the important case — checking only for "executed zero" lets 237
#   of 238 run and still call it green, which is the classifier/runner divergence
#   this design exists to prevent, surviving one layer down.
#
#   `RMLX_SKIP_GPU=1` is refused outright rather than tolerated. Every classified
#   test opens with `if skip_if_no_gpu_env() { return; }`, so with it set the
#   whole suite returns before touching Metal and the gate would report a
#   comfortable "OK: N GPU tests passed" having dispatched nothing. This gate
#   exists to prove the GPU path RAN.
#
#   Model-gated cells that skip because no snapshot is present still count as
#   executed; that is the suite's own documented contract and is not a
#   process-wide off switch.
#
# SHADER VALIDATION (--shader-validation, on by default)
#   An out-of-bounds device store from a Metal kernel is silently dropped: the
#   command buffer completes, `cb.error` is nil, the process exits 0, and the
#   assertions downstream of the frozen buffer still pass. That is this repo's
#   documented silent-corruption class, and no amount of `cargo test` sees it —
#   measured on a deliberately broken q8 quantize kernel, both GPU tests over it
#   passed with no output at all.
#
#   Metal's shader validation instruments every pipeline and reports the invalid
#   access, naming the kernel function. Two details decide the implementation:
#
#     * The report never reaches the exit code. Even with validation on, cargo
#       exits 0 and the tests report `ok`. The signal is a diagnostic in the
#       output, so this gate SCANS for it; a gate that trusts the exit code
#       passes while the diagnostic scrolls past.
#     * MTL_SHADER_VALIDATION=1 alone is not enough. Reports go to Unified
#       Logging unless MTL_SHADER_VALIDATION_REPORT_TO_STDERR is set — it
#       defaults to 0 (`man MetalValidation`). Setting only the first variable
#       yields a gate that runs, prints its banner, and can never fire.
#
#   So this script owns the whole MTL_SHADER_VALIDATION_* environment rather
#   than inheriting it: every knob that can disable instrumentation or muzzle
#   reporting is pinned explicitly, so no stale export in a dev shell can leave
#   the gate looking armed while it is blind. It then asserts the validation
#   banner actually appeared — if the layer did not load, the run proved nothing
#   and reporting green would be a lie.
#
#   Validation costs throughput, so it belongs here and not in any cell whose
#   numbers get recorded. Pass --no-shader-validation to opt out.
#
# USAGE
#   bash scripts/run_gpu_tests.sh
#   bash scripts/run_gpu_tests.sh --crate rmlx-kv-quant
#   bash scripts/run_gpu_tests.sh --crate rmlx-kv-quant --filter rotor_flash
#   bash scripts/run_gpu_tests.sh --no-shader-validation
#   bash scripts/run_gpu_tests.sh --preflight   # preconditions only, no tests
#
#   The narrowing options exist for iterating by hand. A gate must invoke this
#   script with NO arguments — `--crate`/`--filter` narrow the classified
#   population in lockstep with the executed one, so the coverage check below
#   cannot tell a narrowed run from a complete one, and `--no-shader-validation`
#   disarms the instrumentation entirely.
#
# Exit 0 = every selected GPU test passed. Exit 1 = a failure, a shader
# validation diagnostic, or a run that executed nothing.

# No `-e`: cargo's exit code is captured explicitly via PIPESTATUS, and one
# failing crate must not abort the remaining crates. The `[ ... ] && echo` guard
# idioms below also return 1 when the guard is false, which `-e` would treat as
# a fatal error.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
    cat <<'USAGE'
Usage: run_gpu_tests.sh [--crate <name>] [--filter <substring>]
                        [--shader-validation | --no-shader-validation]
       run_gpu_tests.sh --preflight

  --crate <name>          restrict to one workspace member (e.g. rmlx-kv-quant)
  --filter <substring>    restrict to GPU test fns whose name contains <substring>
  --shader-validation     instrument every Metal pipeline and fail on an invalid
                          memory access (default)
  --no-shader-validation  run the tests uninstrumented
  --preflight             check only the environment preconditions (no GPU
                          variable set, no competing MLX process, a non-empty
                          classification) and exit; runs no tests
USAGE
}

ONLY_CRATE=""
FILTER=""
SHADER_VALIDATION=1
PREFLIGHT=0
while [ $# -gt 0 ]; do
    case "$1" in
        --crate)  ONLY_CRATE="${2:?--crate needs a value}"; shift 2 ;;
        --filter) FILTER="${2:?--filter needs a value}"; shift 2 ;;
        --shader-validation)    SHADER_VALIDATION=1; shift ;;
        --no-shader-validation) SHADER_VALIDATION=0; shift ;;
        --preflight) PREFLIGHT=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown argument '$1'" >&2; usage >&2; exit 1 ;;
    esac
done

# Pinned rather than inherited. Every one of these can silently disarm the
# gate: DEFAULT_STATE=none skips instrumenting pipelines, DISABLE_PIPELINES
# exempts named ones, GLOBAL_MEMORY=0 / THREADGROUP_MEMORY=0 drop the
# instrumentation this gate is about, ENABLE_ERROR_REPORTING=0 detects and says
# nothing, and REPORT_TO_STDERR defaults to 0 so reports land in Unified
# Logging where nothing here reads them. FAIL_MODE=zerofill keeps the run alive
# after the first hit (invalid writes are dropped), so one run reports every
# offending kernel instead of the first. See `man MetalValidation`.
#
# Naming these is not enough on its own: `env NAME=VALUE` overrides exactly what
# it names and passes the rest of the environment through, so any other MTL_*
# knob — a stale export, or one a future toolchain adds — still reaches the test
# process. `mtl_unset` below clears the whole MTL_* namespace first, so the
# pinned set is the entire Metal configuration the tests run under.
mtl_validation_env=(
    MTL_SHADER_VALIDATION=1
    MTL_SHADER_VALIDATION_DEFAULT_STATE=all
    MTL_SHADER_VALIDATION_DISABLE_PIPELINES=
    MTL_SHADER_VALIDATION_ENABLE_ERROR_REPORTING=1
    MTL_SHADER_VALIDATION_GLOBAL_MEMORY=1
    MTL_SHADER_VALIDATION_THREADGROUP_MEMORY=1
    MTL_SHADER_VALIDATION_FAIL_MODE=zerofill
    MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1
)
# Every inherited MTL_* name, as `-u NAME` pairs for `env`. Built once, before
# any crate runs. It is non-empty in practice only when the caller has such
# exports, so it is expanded alongside a non-empty array and never alone — an
# empty array expansion under `set -u` is an error on the bash 3.2 that
# `/usr/bin/env bash` resolves to here.
mtl_unset=()
while IFS='=' read -r mtl_name _; do
    case "${mtl_name}" in
    MTL_*) mtl_unset+=(-u "${mtl_name}") ;;
    esac
done < <(env)

# Printed by Metal at device creation once the layer is live. Its absence means
# the run was uninstrumented no matter what the variables above say.
VALIDATION_BANNER='Metal GPU Validation Enabled'
# Shape of every report seen from this layer, across device and threadgroup
# memory, loads and stores:
#   Invalid device store at offset 4000068, executing kernel function: "..."
# Deliberately NOT anchored at line start: the layer writes to the process's
# stderr while libtest is mid-line, so a report routinely lands appended to a
# `test some::name ... ` prefix, and `^Invalid` sees only the ones that happen
# to start a line. Requiring one of the two markers within a bounded distance
# is what keeps a test's own "invalid" wording from forging a hit.
VALIDATION_DIAGNOSTIC='Invalid .{0,120}(at offset [0-9]+|executing kernel function:)'

# Same environment the crates run under, hoisted so the canary above can use it.
# `${arr[@]+"${arr[@]}"}` is the portable spelling: mtl_unset is empty whenever
# the caller exported no MTL_* names, and expanding an empty array is an
# unbound-variable error under `set -u` on the bash 3.2 that `/usr/bin/env bash`
# resolves to here — being adjacent to a non-empty array does not help.
validation_prefix_canary=(env ${mtl_unset[@]+"${mtl_unset[@]}"} "${mtl_validation_env[@]}")

# Every classified test starts with `if skip_if_no_gpu_env() { return; }`, so
# this variable turns the entire suite into no-ops that still report as passed.
# It is a documented setting for Metal-less environments, which means a stale
# export in a dev shell would silently disarm the one step that proves the GPU
# path works. Refuse rather than run a hollow suite.
if [ "${RMLX_SKIP_GPU:-}" = "1" ]; then
    echo "ERROR: RMLX_SKIP_GPU=1 — every GPU test would return before touching Metal." >&2
    echo "This gate proves the GPU path RAN; unset it (or use 'make test' instead)." >&2
    exit 1
fi

# A model-gated cell — every golden-token gate, and the snapshot-backed decode
# suites — returns early and reports `ok` when its snapshot is not on disk, and
# the coverage check below counts that as executed. That is the suite's own
# documented contract (a developer without the weights must not be blocked by
# this gate) and it is deliberately NOT a refusal. What it must not do is look
# identical to a run that checked those cells: a `make ci-perf` on a machine
# with no snapshot root reports exactly the same "OK: N GPU tests passed" as one
# that ran every golden, which is how a green ci-perf came to be reported over a
# red golden suite. So the state is captured here and printed with the result.
snapshot_root_note=""
if [ -z "${RMLX_O_MODELS_ROOT:-}" ]; then
    snapshot_root_note="RMLX_O_MODELS_ROOT is UNSET — every snapshot-gated cell (all golden-token gates) skipped and counted as passed"
elif [ ! -d "${RMLX_O_MODELS_ROOT}" ]; then
    snapshot_root_note="RMLX_O_MODELS_ROOT='${RMLX_O_MODELS_ROOT}' does not exist — every snapshot-gated cell skipped and counted as passed"
fi

# CLAUDE.md hard rule 8 — a single MLX process per Mac. These tests build their
# own Metal context; a co-resident server already holding the GPU makes any
# failure here unattributable. Refuse rather than pkill: killing a process this
# script does not own is not its call.
if pgrep -f 'rmlx serve|mlx_lm|paroquant|omlx' >/dev/null 2>&1; then
    echo "ERROR: another MLX process is live — the GPU tests need the Metal context to themselves." >&2
    pgrep -fl 'rmlx serve|mlx_lm|paroquant|omlx' >&2 || true
    echo >&2
    echo "Stop it first, e.g.:" >&2
    echo "  pkill -f 'rmlx serve'; pkill -f mlx_lm; rm -f /tmp/rmlx.*.claim" >&2
    exit 1
fi

listing="$(bash "${REPO_ROOT}/scripts/check_gpu_tests_ignored.sh" --list)"
if [ -z "${listing}" ]; then
    echo "ERROR: check_gpu_tests_ignored.sh --list produced no GPU tests." >&2
    exit 1
fi

# Everything above this line is a precondition on the environment rather than a
# test: the two refusals and a non-empty classification, all of them
# milliseconds. `--preflight` stops here so a caller that is about to spend a
# long time on something else can find out FIRST that this suite would refuse to
# start. `make ci-perf` runs it before its release-perf half for exactly that
# reason — discovering a live `rmlx serve` after the workspace suite has already
# run wastes the whole of it.
if [ "${PREFLIGHT}" = "1" ]; then
    echo "preflight OK: GPU free, no skip variable, $(printf '%s\n' "${listing}" | grep -c '') tests classified."
    [ -n "${snapshot_root_note}" ] && echo "preflight WARNING: ${snapshot_root_note}." >&2
    exit 0
fi
[ -n "${snapshot_root_note}" ] && echo "WARNING: ${snapshot_root_note}." >&2

# Apply the narrowing options, then group what is left by crate.
# The canary is the gate's own positive control, not part of the correctness
# suite: it is run separately, first, with its own feature enabled. Left in this
# population the coverage check would demand it run in the ordinary pass, where
# the feature is off and it is not compiled at all.
CANARY_TEST="shader_validation_canary_emits_an_invalid_access_report"

selected=""
while IFS=$'\t' read -r crate fn_name; do
    [ -z "${crate}" ] && continue
    [ "${fn_name}" = "${CANARY_TEST}" ] && continue
    if [ -n "${ONLY_CRATE}" ] && [ "${crate}" != "${ONLY_CRATE}" ]; then continue; fi
    case "${fn_name}" in
        *"${FILTER}"*) selected="${selected}${crate}"$'\t'"${fn_name}"$'\n' ;;
    esac
done <<< "${listing}"

# `sort -u`, not `uniq`: uniq collapses only ADJACENT duplicates, which happens
# to hold because the classifier's outer loop is per-member — but nothing
# enforces that emit order, and a future change to it would silently run a crate
# twice. Crate order is irrelevant to the run.
crates=()
while IFS= read -r c; do
    [ -n "$c" ] && crates+=("$c")
done < <(printf '%s' "${selected}" | cut -f1 | sort -u)

if [ ${#crates[@]} -eq 0 ]; then
    echo "ERROR: selection matched no GPU tests." >&2
    [ -n "${ONLY_CRATE}" ] && echo "  --crate '${ONLY_CRATE}'" >&2
    [ -n "${FILTER}" ] && echo "  --filter '${FILTER}'" >&2
    echo "A gate that runs nothing passes everything; refusing to report OK." >&2
    exit 1
fi

failed_crates=""
total_passed=0
total_failed=0
validation_hits=""

if [ "${SHADER_VALIDATION}" = "1" ]; then
    echo "shader validation: ON (invalid Metal memory access fails this run)"
    # A positive control, run before anything is trusted to be clean. The
    # banner proves the instrumentation LOADED; it says nothing about whether
    # the detector below still matches what this toolchain emits, and
    # VALIDATION_DIAGNOSTIC is a hand-written pattern over an undocumented,
    # version-specific message. Without this, a wording change at the next Xcode
    # bump turns the gate into one that runs, banners, and never fires.
    #
    # The canary kernel stores out of bounds on purpose and lives behind its own
    # feature, so it is not built into any ordinary test run.
    canary_log="$(mktemp -t rmlx-gpu-canary)"
    "${validation_prefix_canary[@]}" cargo test --no-fail-fast -p rmlx-kv-quant \
        --features shader-validation-canary --lib -- \
        --ignored --test-threads=1 "${CANARY_TEST}" >"${canary_log}" 2>&1
    if ! grep -Eq "${VALIDATION_DIAGNOSTIC}" "${canary_log}"; then
        echo "ERROR: the out-of-bounds canary produced no diagnostic this scan would" >&2
        echo "  match, so a clean scan proves nothing. Either the canary did not run" >&2
        echo "  or this toolchain's report format changed." >&2
        echo "  Pattern: ${VALIDATION_DIAGNOSTIC}" >&2
        echo "  Log: ${canary_log}" >&2
        exit 1
    fi
    rm -f "${canary_log}"
    echo "shader validation: detector confirmed against a deliberate OOB store"
else
    echo "shader validation: OFF (--no-shader-validation) — an out-of-bounds"
    echo "  device store will be dropped silently and this run will not see it."
fi

for crate in "${crates[@]}"; do
    # `classified` counts the classifier's LINES, not its unique names, because
    # each line is a distinct `#[test]` fn: seven names are defined in more than
    # one module today (e.g. `gpu_two_append_multi_head_roundtrip` x3), and those
    # are three separate tests that all run. Deduping the coverage target would
    # accept two of the three vanishing. The filter LIST is deduped, since
    # passing one substring twice selects nothing extra.
    classified=0
    filters=()
    while IFS=$'\t' read -r c fn_name; do
        [ "$c" = "${crate}" ] || continue
        classified=$((classified + 1))
    done <<< "${selected}"
    while IFS= read -r fn_name; do
        [ -n "${fn_name}" ] && filters+=("${fn_name}")
    done < <(printf '%s' "${selected}" | awk -F'\t' -v c="${crate}" '$1 == c {print $2}' | sort -u)
    echo "── ${crate} (${classified} GPU tests) ──────────────────────────"

    log="$(mktemp -t "rmlx-gpu-test-${crate}")"
    # `--tests` selects every target with `test = true` — the lib's unit tests,
    # each bin's unit tests, and the `tests/*.rs` integration binaries. That is
    # exactly the set the classifier scans, so the runner cannot be pointed at
    # more or less than the rule covers. It is also the only spelling that gets
    # all three without breaking:
    #   * `--lib` hard-errors with "no library targets found in package" on a
    #     bin-only member such as rmlx-cli, which the classifier does scan.
    #   * `--all-targets` drags in benches, and criterion harnesses reject
    #     `--ignored` outright ("error: unexpected argument found").
    #   * doc-tests stay out either way, deliberately: `--ignored` makes rustdoc
    #     compile ```ignore blocks, which are prose, not tests.
    #
    # Names come from the classifier as bare fn identifiers, while libtest
    # matches against the full `module::path::fn` — so these are substring
    # filters, not `--exact`. Over-matching a sibling is harmless (it runs one
    # extra test); under-matching is what the coverage check below catches.
    #
    # `--no-fail-fast` is load-bearing for that check: without it cargo stops
    # after the first test binary that fails, so every later binary in the crate
    # silently never runs and the coverage shortfall reports as "a filter
    # stopped matching" when the real cause was an earlier failure.
    # `env` prefix, not `export`: the validation settings apply to the test
    # process and nothing else, and the empty-valued DISABLE_PIPELINES entry
    # overrides a stale export rather than merging with it. The prefix keeps a
    # bare `env` when validation is off, because expanding an empty array is an
    # unbound-variable error under `set -u` on the bash 3.2 that
    # `/usr/bin/env bash` resolves to here.
    validation_prefix=(env)
    [ "${SHADER_VALIDATION}" = "1" ] &&
        validation_prefix=(env ${mtl_unset[@]+"${mtl_unset[@]}"} "${mtl_validation_env[@]}")
    "${validation_prefix[@]}" cargo test --no-fail-fast -p "${crate}" --tests -- \
        --ignored --test-threads=1 "${filters[@]}" 2>&1 | tee "${log}"
    rc=${PIPESTATUS[0]}

    counts="$(awk '
        /^test result:/ {
            for (i = 1; i <= NF; i++) {
                if ($(i+1) ~ /^passed/) { p += $i }
                if ($(i+1) ~ /^failed/) { f += $i }
            }
        }
        END { printf "%d %d", p, f }
    ' "${log}")"
    # Harvest the failing test names while the log still exists — a red gate
    # that only names the crate leaves an operator unable to tell their own
    # regression from the known baseline without re-running by hand.
    crate_fails="$(awk '
        /^failures:$/ { f = 1; next }
        /^test result:/ { f = 0 }
        f && /^    [A-Za-z_]/ { print $1 }
    ' "${log}" | sort -u)"

    if [ "${SHADER_VALIDATION}" = "1" ]; then
        # Per crate, matching the coverage check's granularity. A single global
        # OR would let one crate's banner vouch for a crate whose tests all
        # returned before creating a Metal device.
        if ! grep -qF "${VALIDATION_BANNER}" "${log}"; then
            failed_crates="${failed_crates}  ${crate}: ran uninstrumented (no validation banner)"$'\n'
        fi
        # Report the kernel each hit names, deduped — a codec's kernel name is
        # the actionable identifier here. The buffer field is not: MLX owns the
        # allocator, so KV stores come through as `buffer: <unnamed>`.
        hits="$(grep -Eo "${VALIDATION_DIAGNOSTIC}[^\"]*\"[^\"]*\"" "${log}" \
                | grep -Eo 'kernel function: "[^"]*"' | sort | uniq -c | sort -rn)"
        n_hits="$(grep -Ec "${VALIDATION_DIAGNOSTIC}" "${log}")"
        if [ "${n_hits}" -gt 0 ]; then
            validation_hits="${validation_hits}  ${crate}: ${n_hits} invalid access(es)"$'\n'
            if [ -n "${hits}" ]; then
                validation_hits="${validation_hits}$(printf '%s\n' "${hits}" | sed 's/^/    /')"$'\n'
            fi
        fi
    fi
    rm -f "${log}"
    crate_passed=${counts% *}
    crate_failed=${counts#* }
    total_passed=$((total_passed + crate_passed))
    total_failed=$((total_failed + crate_failed))
    executed=$((crate_passed + crate_failed))

    # Coverage check: every classified test must have actually run. A shortfall
    # means a filter stopped matching — a renamed fn, a target that is no longer
    # built, a test compiled out behind a feature — which is silence, not
    # success. Over-matching inflates `executed` and is harmless, so this is a
    # one-sided `-lt`.
    if [ "${executed}" -lt "${classified}" ]; then
        echo "ERROR: ${crate} classified ${classified} GPU tests but executed ${executed} — a filter stopped matching." >&2
        failed_crates="${failed_crates}  ${crate}: under-matched (${executed}/${classified} executed)"$'\n'
        continue
    fi
    if [ "${rc}" -ne 0 ]; then
        failed_crates="${failed_crates}  ${crate}:"$'\n'
        if [ -n "${crate_fails}" ]; then
            # Read loop, not an unquoted expansion: splitting on newlines that
            # way also glob-expands each test name against the cwd.
            while IFS= read -r fail_name; do
                [ -n "${fail_name}" ] &&
                    failed_crates="${failed_crates}    ${fail_name}"$'\n'
            done <<< "${crate_fails}"
        fi
    fi
done

echo
# An invalid access is a failure even though every test reported `ok` and cargo
# exited 0 — that is the whole point: the store is dropped, the buffer freezes,
# and the assertions downstream of it still pass.
if [ "${SHADER_VALIDATION}" = "1" ] && [ -n "${validation_hits}" ]; then
    echo "ERROR: Metal shader validation reported invalid memory access:" >&2
    printf '%s' "${validation_hits}" >&2
    echo >&2
    echo "An out-of-bounds device store is DROPPED, not raised: the tests above" >&2
    echo "still passed and cargo still exited 0. Treat this as corruption." >&2
    exit 1
fi

if [ -n "${failed_crates}" ]; then
    echo "ERROR: GPU tests failed in:" >&2
    printf '%s' "${failed_crates}" >&2
    echo >&2
    echo "Reproduce one crate with:" >&2
    echo "  cargo test --no-fail-fast -p <crate> --tests -- --ignored --test-threads=1 <filter>" >&2
    echo "This suite is NOT known to be green on main, and this runner tracks no" >&2
    echo "known-red list. Before attributing a failure above to your change, re-run" >&2
    echo "the same crate and filter on a clean checkout of your base commit and" >&2
    echo "compare: that is the only thing that separates a regression you caused" >&2
    echo "from one you inherited, and it is cheap. See docs/TESTING.md." >&2
    echo "A crate reported as 'ran uninstrumented' usually failed to BUILD: no test" >&2
    echo "binary means no Metal device and therefore no validation banner." >&2
    exit 1
fi

# The note rides on the OK line, not only on stderr: a summary an operator reads
# as "green" has to say what it did not check, or the next reader repeats the
# mistake of quoting the exit code.
if [ -n "${snapshot_root_note}" ]; then
    incomplete=" — INCOMPLETE: ${snapshot_root_note}"
else
    incomplete=""
fi
if [ "${SHADER_VALIDATION}" = "1" ]; then
    echo "OK: ${total_passed} GPU tests passed across ${#crates[@]} workspace member(s), shader validation clean.${incomplete}"
else
    echo "OK: ${total_passed} GPU tests passed across ${#crates[@]} workspace member(s) (uninstrumented).${incomplete}"
fi
