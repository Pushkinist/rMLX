#!/usr/bin/env bash
# scripts/gpu_test_threadgroup.sh — which classified GPU tests run with Metal's
# threadgroup-memory validation. One row per test, `<on|off><TAB><crate><TAB><test>`.
#
# WHY TWO SETTINGS
#   `MTL_SHADER_VALIDATION_THREADGROUP_MEMORY=1` makes MLX's own NAX kernels
#   return wrong values, differently on each run: the routed-expert
#   `gather_qmm` (`affine_gather_qmm_rhs_nax`) in a Qwen3.6 MoE prefill, and
#   the f32 `affine_qmm_t_nax` in a PARO forward, report addresses no index in
#   those kernels can produce and change the output of the test over them. With
#   the threadgroup instrumentation off and everything else on, the same calls
#   are bit-identical to an unvalidated run on every repeat, and the device
#   diagnostics are still reported at the same count. So a test that runs a
#   checkpoint cannot be judged with it on.
#
#   rMLX's own `.metal` kernels are small, run on synthetic operands in their
#   tests, and are the kernels this repo can get wrong in threadgroup memory.
#   Their tests keep the instrumentation.
#
# THE RULE
#   A classified GPU test runs with threadgroup validation `on` if and only if
#   its declaring file is under a workspace member that owns a gated `.metal`
#   directory (`scripts/metal_dirs.sh`), and either
#     * that member is not the model layer, or
#     * the file is the test file of a source that includes a `.metal` file
#       from a gated directory — itself, or the `<name>.rs` beside a
#       `<name>_tests.rs`.
#   Every other classified GPU test runs with it `off`.
#
#   The members below the model layer that ship MSL test their kernels on
#   operands they build, and load no checkpoint. The model layer's tests run
#   checkpoints, except the tests of its own kernel dispatchers, which sit
#   beside the source that embeds the kernel.
#
#   Every fact is read from the tree and none is a test name: the gated
#   directories from `scripts/metal_dirs.sh`, the member and its package name
#   from the directory's manifest, the declaring file from the classifier, the
#   embedding from the `include_str!` in the source.
#
# WHAT IT CANNOT SEE
#   A test below the model layer that loads a checkpoint runs instrumented, and
#   an own-kernel test in the model layer that is not its dispatcher's test file
#   runs uninstrumented. Neither exists in the tree today.
#
# A classification in which either setting is empty is refused: the whole
# point is the split, and a producer that put every test on one side would run
# the suite with one setting and report a clean split.
#
# USAGE
#   bash scripts/gpu_test_threadgroup.sh
#   bash scripts/gpu_test_threadgroup.sh --root <dir>
#
# Exit 0 = the rows, on stdout. Exit 2 = they could not be computed, with the
# reason named.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${REPO_ROOT}"
MODEL_LAYER="rmlx-models"

# shellcheck source=lib/awk_text.sh
. "${REPO_ROOT}/scripts/lib/awk_text.sh"

while [ $# -gt 0 ]; do
    case "$1" in
        --root) ROOT="${2:?--root needs a value}"; shift 2 ;;
        -h|--help)
            echo "Usage: gpu_test_threadgroup.sh [--root <dir>]"
            exit 0 ;;
        *) echo "ERROR: unknown argument '$1' (expected --root <dir>)." >&2; exit 2 ;;
    esac
done

if [ ! -d "${ROOT}" ]; then
    echo "ERROR: --root '${ROOT}' is not a directory." >&2
    exit 2
fi
ROOT="$(cd "${ROOT}" && pwd)"

# The gated directories, resolved against the tree being classified.
METAL_DIRS=()
SCRIPTS_DIR="${REPO_ROOT}/scripts"
REPO_ROOT="${ROOT}"
# shellcheck source=metal_dirs.sh
. "${SCRIPTS_DIR}/metal_dirs.sh"
REPO_ROOT="$(cd "${SCRIPTS_DIR}/.." && pwd)"
if [ ${#METAL_DIRS[@]} -eq 0 ]; then
    echo "ERROR: scripts/metal_dirs.sh names no gated directory." >&2
    exit 2
fi

# `<package>` per gated member, from the manifest two levels above its
# `src/metal` directory.
gated_pkgs=""
for dir in "${METAL_DIRS[@]}"; do
    manifest="${dir%/src/metal}/Cargo.toml"
    pkg=""
    [ -f "${manifest}" ] &&
        pkg="$(awk -F'"' '/^name[[:space:]]*=/{print $2; exit}' "${manifest}")"
    if [ -z "${pkg}" ]; then
        echo "ERROR: gated directory ${dir} has no member manifest naming a package (${manifest})." >&2
        exit 2
    fi
    gated_pkgs="${gated_pkgs}${pkg}"$'\n'
done

# True when <src> embeds a `.metal` file that lives in a gated directory.
embeds_gated_metal() {
    local src="$1" lit resolved gated
    [ -f "${src}" ] || return 1
    while IFS= read -r lit; do
        [ -n "${lit}" ] || continue
        resolved="$(cd "$(dirname "${src}")/$(dirname "${lit}")" 2>/dev/null && pwd)" || continue
        [ -f "${resolved}/$(basename "${lit}")" ] || continue
        for gated in "${METAL_DIRS[@]}"; do
            case "${resolved}/" in
                "${gated}/"*) return 0 ;;
            esac
        done
    done < <(awk "${AWK_TEXT_FNS}"'
        {
            code = decomment($0)
            while (match(code, /include_str!\("[^"]*\.metal"\)/)) {
                print substr(code, RSTART + 14, RLENGTH - 16)
                code = substr(code, RSTART + RLENGTH)
            }
        }' "${src}")
    return 1
}

listing="$(bash "${REPO_ROOT}/scripts/check_gpu_tests_ignored.sh" --list-files --root "${ROOT}" 2>/dev/null)"
if [ -z "${listing}" ]; then
    echo "ERROR: check_gpu_tests_ignored.sh --list-files produced no GPU tests for ${ROOT}." >&2
    exit 2
fi

rows=""
n_on=0
n_off=0
while IFS=$'\t' read -r crate test file; do
    [ -n "${crate}" ] || continue
    setting="off"
    case $'\n'"${gated_pkgs}" in
        *$'\n'"${crate}"$'\n'*)
            if [ "${crate}" != "${MODEL_LAYER}" ] || embeds_gated_metal "${file}" ||
                { case "${file}" in *_tests.rs) true ;; *) false ;; esac &&
                  embeds_gated_metal "${file%_tests.rs}.rs"; }; then
                setting="on"
            fi ;;
    esac
    if [ "${setting}" = "on" ]; then n_on=$((n_on + 1)); else n_off=$((n_off + 1)); fi
    rows="${rows}${setting}"$'\t'"${crate}"$'\t'"${test}"$'\n'
done <<< "${listing}"

if [ "${n_on}" -eq 0 ] || [ "${n_off}" -eq 0 ]; then
    echo "ERROR: the threadgroup split collapsed — ${n_on} test(s) on, ${n_off} off." >&2
    echo "Every classified GPU test would run with one setting while the split reads" >&2
    echo "as computed. See docs/GPU_TESTS.md." >&2
    exit 2
fi

printf '%s' "${rows}"
