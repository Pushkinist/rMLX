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
#   report OK unless it executed tests: an empty classification, a crate whose
#   listed tests all failed to match, and a `--filter` that selects nothing are
#   all errors, not green runs.
#
#   Tests that skip internally (model-gated cells with no snapshot present,
#   `RMLX_SKIP_GPU=1`) still count as executed. That is accepted: this gate
#   proves the suite RAN; per-test skip logic is the suite's own contract.
#
# USAGE
#   bash scripts/run_gpu_tests.sh
#   bash scripts/run_gpu_tests.sh --crate rmlx-kv-quant
#   bash scripts/run_gpu_tests.sh --crate rmlx-kv-quant --filter rotor_flash
#
# Exit 0 = every selected GPU test passed. Exit 1 = a failure, or a run that
# executed nothing.

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

crates=()
while IFS= read -r c; do
    [ -n "$c" ] && crates+=("$c")
done < <(printf '%s' "${selected}" | cut -f1 | uniq)

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
    filters=()
    while IFS=$'\t' read -r c fn_name; do
        [ "$c" = "${crate}" ] && filters+=("${fn_name}")
    done <<< "${selected}"
    echo "── ${crate} (${#filters[@]} GPU tests) ──────────────────────────"

    log="$(mktemp -t "rmlx-gpu-test-${crate}")"
    # Names come from the classifier as bare fn identifiers, while libtest
    # matches against the full `module::path::fn` — so these are substring
    # filters, not `--exact`. Over-matching a sibling is harmless (it runs one
    # extra test); under-matching is what fail-closed below catches.
    # `--lib --bins --tests` excludes doc-tests: `--ignored` makes rustdoc
    # compile ```ignore blocks, which are prose, not tests.
    cargo test -p "${crate}" --lib --bins --tests -- \
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
    rm -f "${log}"
    crate_passed=${counts% *}
    crate_failed=${counts#* }
    total_passed=$((total_passed + crate_passed))
    total_failed=$((total_failed + crate_failed))

    # Per-crate fail-closed: the classifier said this crate has GPU tests, so a
    # run that executed none of them means the filters stopped matching (a
    # renamed test, a moved target) — silence, not success.
    if [ "$((crate_passed + crate_failed))" -eq 0 ]; then
        echo "ERROR: ${crate} has ${#filters[@]} classified GPU tests but executed 0." >&2
        failed_crates="${failed_crates}  ${crate} (executed 0)"$'\n'
        continue
    fi
    if [ "${rc}" -ne 0 ]; then
        failed_crates="${failed_crates}  ${crate}"$'\n'
    fi
done

echo
if [ -n "${failed_crates}" ]; then
    echo "ERROR: GPU tests failed in:" >&2
    printf '%s' "${failed_crates}" >&2
    echo >&2
    echo "Reproduce one crate with:" >&2
    echo "  cargo test -p <crate> --lib --bins --tests -- --ignored --test-threads=1 <filter>" >&2
    exit 1
fi

echo "OK: ${total_passed} GPU tests passed across ${#crates[@]} workspace member(s)."
