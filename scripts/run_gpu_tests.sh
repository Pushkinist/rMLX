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
#   Exactly the compliant set from `check_gpu_tests_ignored.sh --list` —
#   `#[test]` fns that reach `Device::Gpu` AND carry `#[ignore]`. Deriving the
#   population from the enforcing gate's own classifier is the point: a second,
#   hand-maintained list would drift, and the rule would end up mandating
#   `#[ignore]` on tests this runner never visits.
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
# USAGE
#   bash scripts/run_gpu_tests.sh
#   bash scripts/run_gpu_tests.sh --crate rmlx-kv-quant
#   bash scripts/run_gpu_tests.sh --crate rmlx-kv-quant --filter rotor_flash
#
# Exit 0 = every selected GPU test passed. Exit 1 = a failure, or a run that
# executed nothing.

# No `-e`: cargo's exit code is captured explicitly via PIPESTATUS, and one
# failing crate must not abort the remaining crates. The `[ ... ] && echo` guard
# idioms below also return 1 when the guard is false, which `-e` would treat as
# a fatal error.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
    cat <<'USAGE'
Usage: run_gpu_tests.sh [--crate <name>] [--filter <substring>]

  --crate <name>       restrict to one workspace member (e.g. rmlx-kv-quant)
  --filter <substring> restrict to GPU test fns whose name contains <substring>
USAGE
}

ONLY_CRATE=""
FILTER=""
while [ $# -gt 0 ]; do
    case "$1" in
        --crate)  ONLY_CRATE="${2:?--crate needs a value}"; shift 2 ;;
        --filter) FILTER="${2:?--filter needs a value}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown argument '$1'" >&2; usage >&2; exit 1 ;;
    esac
done

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

# Apply the narrowing options, then group what is left by crate.
selected=""
while IFS=$'\t' read -r crate fn_name; do
    [ -z "${crate}" ] && continue
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
    cargo test --no-fail-fast -p "${crate}" --tests -- \
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
            failed_crates="${failed_crates}$(printf '    %s\n' ${crate_fails})"$'\n'
        fi
    fi
done

echo
if [ -n "${failed_crates}" ]; then
    echo "ERROR: GPU tests failed in:" >&2
    printf '%s' "${failed_crates}" >&2
    echo >&2
    echo "Reproduce one crate with:" >&2
    echo "  cargo test --no-fail-fast -p <crate> --tests -- --ignored --test-threads=1 <filter>" >&2
    echo "Known-red baseline (tracked separately): the four rotor fused-QK dispatch" >&2
    echo "tests in rmlx-kv-quant reporting 'dispatch delta = 0'. See docs/TESTING.md." >&2
    exit 1
fi

echo "OK: ${total_passed} GPU tests passed across ${#crates[@]} workspace member(s)."
