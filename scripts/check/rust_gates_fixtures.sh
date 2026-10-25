#!/usr/bin/env bash
# Recall test for the Rust-source gates: planted scan roots from a table, asserting exit and reason.
#
# Recall test for: check_eval_lock.sh
#                  check/gpu_tests.sh ignored
#                  check/kernel_dispatch.sh input-eval dtype-contract scalar-f32
#                  check_kv_layer_quants.sh
#                  check_no_decode_swallow.sh
#                  check_no_inline_tests.sh
#
# The mode is a directory under scripts/fixtures/ that holds a planted tree for
# each case and a cases.tsv: one row per case, tab-separated, no empty field
# ("-" for none). The table's own header comment says what each column holds and
# why its gate needs the cases it has. The gate's path is named here, in the
# mode's arm below, and nowhere in a table. A new source gate adds an arm and a
# table, never a file.
#
# A row whose tree is missing is a red case, and so is a tree under the mode's
# directory that no row names, a hidden one included: a gate that refuses a root
# it cannot read exits with the same code a violation does, so a deleted tree must
# not pass for it.
#
# A needle in the has column is a fixed string; "A;>B" is one needle that holds
# when B appears after the first A, for output whose order is part of the contract.
#
# A gate that takes a mode word gets the mode of its table. A first word "!name" in args
# replaces it ("!bogus --root {}" runs an unknown mode, "! --root {}" runs none, "!list
# --root {}" runs a sibling mode of the same gate), for the rows that prove the refusal of a
# mode and the rows that read another mode's output.
#
# Usage:  bash scripts/check/rust_gates_fixtures.sh <mode>
#         mode: eval_lock | gpu_tests_ignored | kernel_dtype_contract | no_kernel_input_eval |
#               kv_layer_quants | no_decode_swallow | no_inline_tests | scalar_f32_leak
# Exit:   0 every case held; 1 a case did not, or none ran; 2 an unknown mode, a
#         missing table, a malformed row (not 8 fields, a fixture name that is a
#         path, args without "{}", a "!mode" word for a gate with no mode word, a needle that
#         is blank) or no awk on the machine

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/lib/selftest_support.sh
. "$REPO_ROOT/scripts/lib/selftest_support.sh"

MODE="${1:-}"
GATE_MODE=""
case "$MODE" in
eval_lock) GATE="$REPO_ROOT/scripts/check_eval_lock.sh" ;;
gpu_tests_ignored)
    GATE="$REPO_ROOT/scripts/check/gpu_tests.sh"
    GATE_MODE=ignored
    ;;
kernel_dtype_contract)
    GATE="$REPO_ROOT/scripts/check/kernel_dispatch.sh"
    GATE_MODE=dtype-contract
    ;;
no_kernel_input_eval)
    GATE="$REPO_ROOT/scripts/check/kernel_dispatch.sh"
    GATE_MODE=input-eval
    ;;
kv_layer_quants) GATE="$REPO_ROOT/scripts/check_kv_layer_quants.sh" ;;
no_decode_swallow) GATE="$REPO_ROOT/scripts/check_no_decode_swallow.sh" ;;
no_inline_tests) GATE="$REPO_ROOT/scripts/check_no_inline_tests.sh" ;;
scalar_f32_leak)
    GATE="$REPO_ROOT/scripts/check/kernel_dispatch.sh"
    GATE_MODE=scalar-f32
    ;;
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

# add_needles <--has|--lacks> <needles separated by ;;> — append them to CHECKS, or to ORDERED
# when a has needle holds an "A;>B" chain; "-" adds none. A blank needle matches any output.
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
        case "$needle" in
        *[![:space:]]*) ;;
        *) st_die "scripts/fixtures/$MODE/cases.tsv: '$fixture': a $flag needle is blank" ;;
        esac
        case "$flag$needle" in
        --has*';>'*) ORDERED+=("$needle") ;;
        *) CHECKS+=("$flag" "$needle") ;;
        esac
    done
}

# order_why <output> <A;>B;>...> — print why when a part is not found after the one before it.
order_why() {
    local rest="$1" chain="$2" part
    while [ -n "$chain" ]; do
        case "$chain" in
        *';>'*)
            part="${chain%%;>*}"
            chain="${chain#*;>}"
            ;;
        *)
            part="$chain"
            chain=""
            ;;
        esac
        case "$part" in
        *[![:space:]]*) ;;
        *) st_die "scripts/fixtures/$MODE/cases.tsv: an ordered needle has a blank part" ;;
        esac
        case "$rest" in
        *"$part"*) rest="${rest#*"$part"}" ;;
        *)
            echo "no '$part' after the part before it in the output"
            return 0
            ;;
        esac
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
    while IFS=$'\t' read -u 3 -r fixture args stream want has lacks is what extra || [ -n "$fixture" ]; do
        case "$fixture" in
        '' | '#'*) continue ;;
        esac
        if [ -z "$what" ]; then
            st_die "scripts/fixtures/$MODE/cases.tsv: the row for '$fixture' has fewer than 8 fields"
        fi
        if [ -n "$extra" ]; then
            st_die "scripts/fixtures/$MODE/cases.tsv: the row for '$fixture' has more than 8 fields"
        fi
        case "$fixture" in
        */* | .*) st_die "scripts/fixtures/$MODE/cases.tsv: '$fixture' is a path, not the name of a tree beside the table" ;;
        esac
        case "$args" in
        *'{}'*) ;;
        *) st_die "scripts/fixtures/$MODE/cases.tsv: '$fixture': args '$args' has no {}, so the gate would not run on the tree" ;;
        esac
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
        mode_word="$GATE_MODE"
        case "${words[0]}" in
        '!'*)
            if [ -z "$GATE_MODE" ]; then
                st_die "scripts/fixtures/$MODE/cases.tsv: '$fixture': args '$args' name a mode, and this gate has none"
            fi
            mode_word="${words[0]#!}"
            words=("${words[@]:1}")
            ;;
        esac
        for word in "${words[@]}"; do
            ARGV+=("${word//\{\}/$FIX/$fixture}")
        done
        CHECKS=()
        ORDERED=()
        add_needles --has "$has"
        add_needles --lacks "$lacks"
        if [ "$is" != - ]; then CHECKS+=(--is "$(printf '%b' "$is")"); fi
        case "$stream" in
        both) st_run env PATH="$RUN_PATH" bash "$GATE" ${mode_word:+"$mode_word"} "${ARGV[@]}" ;;
        out) st_run stdout_of env PATH="$RUN_PATH" bash "$GATE" ${mode_word:+"$mode_word"} "${ARGV[@]}" ;;
        err) st_run stderr_of env PATH="$RUN_PATH" bash "$GATE" ${mode_word:+"$mode_word"} "${ARGV[@]}" ;;
        esac
        why="$(st_check "$want" ${CHECKS[@]+"${CHECKS[@]}"})"
        for chain in ${ORDERED[@]+"${ORDERED[@]}"}; do
            ordered="$(order_why "$ST_OUT" "$chain")"
            why="${why:+$why$'\n'}$ordered"
        done
        st_verdict "$fixture" "$why" "$what"
    done 3<"$FIX/cases.tsv"
done

NAMED="$(cut -f1 "$FIX/cases.tsv")"
for dir in "$FIX"/*/ "$FIX"/.[!.]*/; do
    [ -d "$dir" ] || continue
    name="$(basename "$dir")"
    if ! grep -qxF -- "$name" <<<"$NAMED"; then
        st_verdict "$name" "fixture directory is not named by any row of scripts/fixtures/$MODE/cases.tsv"
    fi
done

st_done
