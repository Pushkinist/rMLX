#!/usr/bin/env bash
# Recall test for the Rust-source gates: planted scan roots from a table, asserting exit and reason.
#
# Recall test for: check_eval_lock.sh
# Recall test for: check_gpu_tests_ignored.sh
# Recall test for: check_kernel_dtype_contract.sh
# Recall test for: check_no_kernel_input_eval.sh
# Recall test for: check_kv_layer_quants.sh
# Recall test for: check_no_decode_swallow.sh
# Recall test for: check_no_inline_tests.sh
#
# The mode is a directory under scripts/fixtures/ that holds a planted tree for
# each case and a cases.tsv: one row per case, tab-separated, no empty field
# ("-" for none). The table's own header comment says what each column holds and
# why its gate needs the cases it has. The gate's path is named here, in the
# mode's arm below, and nowhere in a table. A new source gate adds an arm and a
# table, never a file.
#
# A row whose tree is missing is a red case, and so is a tree under the mode's
# directory that no row names: a gate that refuses a root it cannot read exits
# with the same code a violation does, so a deleted tree must not pass for it.
#
# Usage:  bash scripts/check/rust_gates_fixtures.sh <mode>
#         mode: eval_lock | gpu_tests_ignored | kernel_dtype_contract |
#               no_kernel_input_eval | kv_layer_quants | no_decode_swallow |
#               no_inline_tests
# Exit:   0 every case held; 1 a case did not, or none ran; 2 an unknown mode, a
#         missing table, a malformed row or no awk on the machine
# Caller: by hand

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/lib/selftest_support.sh
. "$REPO_ROOT/scripts/lib/selftest_support.sh"

MODE="${1:-}"
case "$MODE" in
eval_lock) GATE="$REPO_ROOT/scripts/check_eval_lock.sh" ;;
gpu_tests_ignored) GATE="$REPO_ROOT/scripts/check_gpu_tests_ignored.sh" ;;
kernel_dtype_contract) GATE="$REPO_ROOT/scripts/check_kernel_dtype_contract.sh" ;;
no_kernel_input_eval) GATE="$REPO_ROOT/scripts/check_no_kernel_input_eval.sh" ;;
kv_layer_quants) GATE="$REPO_ROOT/scripts/check_kv_layer_quants.sh" ;;
no_decode_swallow) GATE="$REPO_ROOT/scripts/check_no_decode_swallow.sh" ;;
no_inline_tests) GATE="$REPO_ROOT/scripts/check_no_inline_tests.sh" ;;
*)
    echo "usage: bash scripts/check/rust_gates_fixtures.sh <mode>  (modes: see the header)" >&2
    exit 2
    ;;
esac
FIX="$REPO_ROOT/scripts/fixtures/$MODE"
if [ ! -f "$FIX/cases.tsv" ]; then
    echo "rust_gates_fixtures: no table scripts/fixtures/$MODE/cases.tsv" >&2
    exit 2
fi

st_init "$MODE"

# The gate is an awk program and awk implementations disagree, so its cases run
# once under each implementation on the machine; the other gates run under the default.
AWKS=("")
if [ "$MODE" = gpu_tests_ignored ]; then
    AWKS=()
    for cand in awk gawk mawk; do
        found="$(command -v "$cand" 2>/dev/null)" || continue
        # one file under two names (Debian's awk -> mawk) is one implementation
        dup=0
        for seen in ${AWKS[@]+"${AWKS[@]}"}; do
            if [ "$found" -ef "$seen" ]; then dup=1; fi
        done
        if [ "$dup" -eq 0 ]; then AWKS+=("$found"); fi
    done
    if [ "${#AWKS[@]}" -eq 0 ]; then
        echo "rust_gates_fixtures: no awk found, so this run proves nothing" >&2
        exit 2
    fi
    if [ "${#AWKS[@]}" -eq 1 ]; then
        echo "NOTE: only one awk implementation here (${AWKS[0]}). This run does NOT"
        echo "      cover the gate's awk portability: a construct BSD awk accepts and"
        echo "      gawk rejects outright leaves this suite green and CI hard-down."
        echo "      \`brew install gawk mawk\` (or apt) to check it here, not in CI."
    fi
fi

stdout_of() { "$@" 2>/dev/null; }
# shellcheck disable=SC2069 # the order is the point: stderr to the capture, stdout dropped
stderr_of() { "$@" 2>&1 >/dev/null; }

# add_needles <--has|--lacks> <needles separated by ;;> — append them to CHECKS; "-" adds none.
add_needles() {
    local flag="$1" rest="$2" needle
    [ "$rest" != - ] || return 0
    while [ -n "$rest" ]; do
        case "$rest" in
        *';;'*)
            needle="${rest%%;;*}"
            rest="${rest#*;;}"
            ;;
        *)
            needle="$rest"
            rest=""
            ;;
        esac
        CHECKS+=("$flag" "$needle")
    done
}

SHIMS=0
for impl in "${AWKS[@]}"; do
    RUN_PATH="$PATH"
    if [ -n "$impl" ]; then
        SHIMS=$((SHIMS + 1))
        shim="$ST_WORK/awk.$SHIMS"
        mkdir -p "$shim"
        ln -sf "$impl" "$shim/awk"
        RUN_PATH="$shim:$PATH"
        echo "== awk: $impl =="
    fi
    while IFS=$'\t' read -u 3 -r fixture args stream want has lacks is what || [ -n "$fixture" ]; do
        case "$fixture" in
        '' | '#'*) continue ;;
        esac
        if [ -z "$what" ]; then
            st_die "scripts/fixtures/$MODE/cases.tsv: the row for '$fixture' has fewer than 8 fields"
        fi
        case "$stream" in
        both | out | err) ;;
        *) st_die "scripts/fixtures/$MODE/cases.tsv: '$fixture': stream '$stream' is not both, out or err" ;;
        esac
        case "$want" in
        *[!0-9]*) st_die "scripts/fixtures/$MODE/cases.tsv: '$fixture': exit '$want' is not a number" ;;
        esac
        if [ ! -d "$FIX/$fixture" ]; then
            st_verdict "$fixture" "fixture directory is missing: scripts/fixtures/$MODE/$fixture" "$what"
            continue
        fi
        ARGV=()
        read -r -a words <<<"$args"
        for word in "${words[@]}"; do
            if [ "$word" = '{}' ]; then ARGV+=("$FIX/$fixture"); else ARGV+=("$word"); fi
        done
        CHECKS=(--what "$what")
        add_needles --has "$has"
        add_needles --lacks "$lacks"
        if [ "$is" != - ]; then CHECKS+=(--is "$(printf '%b' "$is")"); fi
        case "$stream" in
        both) st_case "$fixture" "$want" "${CHECKS[@]}" -- env PATH="$RUN_PATH" bash "$GATE" "${ARGV[@]}" ;;
        out) st_case "$fixture" "$want" "${CHECKS[@]}" -- stdout_of env PATH="$RUN_PATH" bash "$GATE" "${ARGV[@]}" ;;
        err) st_case "$fixture" "$want" "${CHECKS[@]}" -- stderr_of env PATH="$RUN_PATH" bash "$GATE" "${ARGV[@]}" ;;
        esac
    done 3<"$FIX/cases.tsv"
done

NAMED="$(cut -f1 "$FIX/cases.tsv")"
for dir in "$FIX"/*/; do
    [ -d "$dir" ] || continue
    name="$(basename "$dir")"
    if ! grep -qxF -- "$name" <<<"$NAMED"; then
        st_verdict "$name" "fixture directory is not named by any row of scripts/fixtures/$MODE/cases.tsv"
    fi
done

st_done
