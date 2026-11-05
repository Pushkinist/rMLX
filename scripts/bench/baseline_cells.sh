#!/usr/bin/env bash
# baseline_cells.sh — decode-rate cells through `rmlx baseline`: the canary, or one model and codec
#
# Usage:  bash scripts/bench/baseline_cells.sh --preset canary --models-root DIR [--include-26b]
#             [--binary PATH]
#         bash scripts/bench/baseline_cells.sh --model PATH --kv-quant CODEC [--prompt-tokens N]
#             [--max-tokens N] [--warmup N] [--runs N] [--record diagnostic] [--label L]
#             [--binary PATH]
# Exit:   0 every run measured; 1 a run failed, gave no reading, or the gate record did not land;
#         2 a usage error or a missing model; 11 passthrough of rmlx (another process holds the
#         Metal claim); 75 a run measured and its record did not land; 125 the binary is missing
# Not a mode of: bench/codec_inertness_probe.sh — that one reads residency and a token digest at
#         temperature 0; this one reads decode rate
#
# --preset canary measures the three test-target models (and the 26b MoE with --include-26b), each
# on the default KV codec and on k8vturbo3: 1 warmup and 3 measured calls, median decode_tps ±
# sample stddev, one CSV row per arm in <data root>/bench/perf_canary.csv
# (ts_utc,git_sha,model,kv_quant,prompt_tokens,decode_tps,stddev,build_profile). One further
# default-codec run is recorded `gate`, which `make canary-gate` reads; every other run is recorded
# `diagnostic` under the label `canary sha=<sha> <phase>`. The one-token run that names the
# resolved codec is not recorded: a one-token generation has no decode rate and the recorder
# refuses a record without one.
#
# A snapshot whose identity cannot be recorded stops the canary before its first run. A gate
# record that does not land stops it with exit 1 and names its buffer file; a replay adds the row.
# A run that measured and whose record did not land keeps its reading, names its buffer file, and
# the tool ends 75 after the last model (docs/METRICS_DB.md section 13, rule 24).
#
# --model with --kv-quant measures one cell: mean ± sample stddev of the measured runs. --record
# diagnostic records every run, the warmup included, under one --label (default
# codec_cell:<utc stamp>:<codec>) so the runs are found together.
#
# This tool measures one build. Comparing two builds needs scripts/perf_ab.sh, which interleaves
# the arms: ordering and thermal drift land on whichever arm ran second here.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# shellcheck source=scripts/lib/bench_server.sh
source "${REPO_ROOT}/scripts/lib/bench_server.sh"
# shellcheck source=scripts/lib/identity.sh
source "${REPO_ROOT}/scripts/lib/identity.sh"
# shellcheck source=scripts/lib/record_pending.sh
source "${REPO_ROOT}/scripts/lib/record_pending.sh"

PRESET=""
MODELS_ROOT=""
INCLUDE_26B=false
BINARY=""
MODEL_PATH=""
KV_QUANT=""
RECORD_CLASS=""
LABEL=""
CELL_OPTIONS=""
PROMPT_TOKENS=4096
MAX_TOKENS=100
WARMUP_RUNS=1
MEASURED_RUNS=3
MAX_CTX=""
BUILD_PROFILE="release-perf"

usage_error() {
    echo "ERROR: $1" >&2
    exit 2
}

value_of() {
    [[ $# -ge 2 ]] || usage_error "$1 requires a value"
    printf '%s' "$2"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --include-26b) INCLUDE_26B=true; shift; continue ;;
        --preset) PRESET="$(value_of "$@")" ;;
        --models-root) MODELS_ROOT="$(value_of "$@")" ;;
        --binary) BINARY="$(value_of "$@")" ;;
        --model) MODEL_PATH="$(value_of "$@")"; CELL_OPTIONS+=" $1" ;;
        --kv-quant) KV_QUANT="$(value_of "$@")"; CELL_OPTIONS+=" $1" ;;
        --prompt-tokens) PROMPT_TOKENS="$(value_of "$@")"; CELL_OPTIONS+=" $1" ;;
        --max-tokens) MAX_TOKENS="$(value_of "$@")"; CELL_OPTIONS+=" $1" ;;
        --warmup) WARMUP_RUNS="$(value_of "$@")"; CELL_OPTIONS+=" $1" ;;
        --runs) MEASURED_RUNS="$(value_of "$@")"; CELL_OPTIONS+=" $1" ;;
        --record) RECORD_CLASS="$(value_of "$@")"; CELL_OPTIONS+=" $1" ;;
        --label) LABEL="$(value_of "$@")"; CELL_OPTIONS+=" $1" ;;
        *) usage_error "unknown argument: $1" ;;
    esac
    shift 2
done

case "${PRESET}" in
    canary)
        [[ -z "${CELL_OPTIONS}" ]] || usage_error "--preset canary takes none of:${CELL_OPTIONS}"
        [[ -n "${MODELS_ROOT}" ]] || usage_error "--preset canary needs --models-root
set RMLX_O_MODELS_ROOT (see .env.example) or pass --models-root"
        MAX_CTX=8192
        RECORD_CLASS=diagnostic
        ;;
    "")
        [[ -n "${MODEL_PATH}" ]] || usage_error "--model is required without --preset"
        [[ -n "${KV_QUANT}" ]] || usage_error "--kv-quant is required without --preset"
        if [[ -n "${MODELS_ROOT}" ]] || ${INCLUDE_26B}; then
            usage_error "--models-root and --include-26b belong to --preset canary"
        fi
        [[ -z "${RECORD_CLASS}" || "${RECORD_CLASS}" == diagnostic ]] || usage_error "--record takes diagnostic"
        [[ -d "${MODEL_PATH}" ]] || usage_error "model path not found: ${MODEL_PATH}"
        [[ -n "${LABEL}" ]] || LABEL="codec_cell:$(date -u +%Y%m%dT%H%M%SZ):${KV_QUANT}"
        ;;
    *) usage_error "unknown preset: ${PRESET}" ;;
esac
if ! [[ "${PROMPT_TOKENS}" =~ ^[1-9][0-9]*$ && "${MAX_TOKENS}" =~ ^[1-9][0-9]*$ &&
    "${MEASURED_RUNS}" =~ ^[1-9][0-9]*$ && "${WARMUP_RUNS}" =~ ^[0-9]+$ ]]; then
    usage_error "--prompt-tokens, --max-tokens and --runs take a positive integer, --warmup a non-negative one"
fi

TARGET_DIR="${REPO_ROOT}/target"
if [[ -z "${BINARY}" ]]; then
    for profile in release-perf release; do
        BINARY="${TARGET_DIR}/${profile}/rmlx"
        # The canary files build_profile=release-perf, so it takes no other build.
        if [[ -x "${BINARY}" || -n "${PRESET}" ]]; then
            break
        fi
    done
fi
if [[ ! -x "${BINARY}" ]]; then
    echo "ERROR: binary not found at ${BINARY}. Run: make build-perf" >&2
    exit 125
fi

# The measured calls hide rmlx's stderr, so a Metal claim another process holds
# is found here: the probe takes the claim and releases it, or exits 11 naming
# the holder, and the run stops with that status.
"${BINARY}" claim run -- true

DATA_ROOT="$("${BINARY}" metrics path --home)"
LOG_DIR="${DATA_ROOT}/logs"
SCRATCH_DIR="${DATA_ROOT}/tmp"
CSV="${DATA_ROOT}/bench/perf_canary.csv"

ctx_args=()
git_sha_args=()
if [[ -n "${PRESET}" ]]; then
    ctx_args=(--max-ctx "${MAX_CTX}")
    mkdir -p "${DATA_ROOT}/bench"
    if [[ ! -f "${CSV}" ]]; then
        echo "ts_utc,git_sha,model,kv_quant,prompt_tokens,decode_tps,stddev,build_profile" > "${CSV}"
    fi
    rmlx_export_identity "${BINARY}"
    GIT_SHA="$(rmlx_git_sha "${REPO_ROOT}")"
    # "unknown" is a fallback for the label text, never provenance: a checkout
    # without .git must not pass --git-sha at all.
    if [[ "${GIT_SHA}" != unknown* ]]; then
        git_sha_args=(--git-sha "${GIT_SHA}")
    fi
fi

# One line per run that measured and whose record did not land.
PENDING="$(mktemp "${TMPDIR:-/tmp}/baseline_cells_pending.XXXXXX")"
trap 'rm -f "${PENDING}"' EXIT

label_for() {
    if [[ -n "${PRESET}" ]]; then
        echo "canary sha=${GIT_SHA}${1:+ $1}"
    else
        echo "${LABEL}"
    fi
}

# One `rmlx baseline`, recorded under <class> when it is not empty; prints
# "<decode_tps>:<prefill_tps>", a missing reading as 0. A run that measured and
# whose record did not land (exit RECORD_PENDING_EXIT) keeps its reading and is
# noted in PENDING, except a gate run, which the canary cannot go on without.
# Any other failure stops the tool.
# Usage: run_once <phase> <class> <model_path> [baseline args...]
run_once() {
    local phase="$1" class="$2" model_path="$3"
    shift 3
    local record_args=() out rc=0 err dtps ptps
    if [[ -n "${class}" ]]; then
        record_args=(--record --run-class "${class}")
    fi
    err="$(mktemp "${TMPDIR:-/tmp}/baseline_cells_err.XXXXXX")"
    out="$("${BINARY}" baseline \
        --model "${model_path}" \
        --prompt-tokens "${PROMPT_TOKENS}" \
        --max-tokens "${MAX_TOKENS}" \
        ${ctx_args[@]+"${ctx_args[@]}"} \
        --label "$(label_for "${phase}")" \
        ${git_sha_args[@]+"${git_sha_args[@]}"} \
        ${record_args[@]+"${record_args[@]}"} \
        "$@" \
        2>"${err}")" || rc=$?
    if [[ "${rc}" -eq "${RECORD_PENDING_EXIT}" && "${class}" != gate ]]; then
        record_pending_note "${err}"
        echo "${model_path##*/} ${phase}" >>"${PENDING}"
    elif [[ "${rc}" -ne 0 ]]; then
        if [[ "${rc}" -eq "${RECORD_PENDING_EXIT}" ]]; then
            record_pending_note "${err}"
            echo "  replaying it adds the gate row this canary reports as missing" >&2
        fi
        rm -f "${err}"
        if [[ "${class}" == gate ]]; then
            echo "ERROR: ${model_path##*/}: the gate record failed (exit ${rc}); the canary has no row to gate on" >&2
        else
            echo "ERROR: ${model_path##*/}: the ${phase} run failed (exit ${rc}; see ${LOG_DIR})" >&2
        fi
        exit 1
    fi
    rm -f "${err}"
    dtps="$(printf '%s\n' "${out}" | grep -o 'decode_tps=[0-9.]*' | cut -d= -f2 || true)"
    ptps="$(printf '%s\n' "${out}" | grep -o 'prefill_tps=[0-9.]*' | cut -d= -f2 || true)"
    echo "${dtps}:${ptps:-0}"
}

# Warmup runs, then measured runs, filling DECODE and PREFILL with the readings.
# <arm> names the arm in the progress lines and run phases; empty is the default
# arm. A measured run with no reading stops the tool when <strict> is 1, else it
# is warned about and skipped.
# Usage: measure_arm <model_path> <arm> <strict> [baseline args...]
measure_arm() {
    local model_path="$1" arm="$2" strict="$3" i result dtps ptps ok
    shift 3
    local tag="${arm:+[${arm}] }"
    DECODE=()
    PREFILL=()
    for ((i = 1; i <= WARMUP_RUNS; i++)); do
        echo "  ${tag}warmup ${i}/${WARMUP_RUNS}..." >&2
        run_once "${arm:+${arm}-}warmup" "${RECORD_CLASS}" "${model_path}" "$@" > /dev/null
    done
    for ((i = 1; i <= MEASURED_RUNS; i++)); do
        echo "  ${tag}measured run ${i}/${MEASURED_RUNS}..." >&2
        result="$(run_once "${arm:-measured}" "${RECORD_CLASS}" "${model_path}" "$@")"
        dtps="${result%%:*}"
        ptps="${result##*:}"
        # A zero reading is a silent model failure, not a measurement.
        ok="$(awk -v v="${dtps}" 'BEGIN { print (v+0 > 0.001) ? "yes" : "no" }')"
        if [[ "${ok}" != yes ]]; then
            if [[ "${strict}" -eq 1 ]]; then
                if [[ -z "${dtps}" ]]; then
                    echo "ERROR: failed to parse decode_tps for ${model_path##*/} run ${i}" >&2
                else
                    echo "ERROR: ${model_path##*/} [${KV_QUANT:-default codec}] run ${i} read ${dtps} tok/s decode_tps; a zero is not a measurement" >&2
                fi
                exit 1
            fi
            echo "  WARN: ${arm} run ${i} failed to parse decode_tps — skipping" >&2
            continue
        fi
        DECODE+=("${dtps}")
        PREFILL+=("${ptps}")
    done
}

# Median of space-separated values (sort via sort(1), compute via awk)
median() {
    echo "$@" | tr ' ' '\n' | sort -n | awk '
    { a[NR]=$1 }
    END {
        n=NR
        if (n % 2 == 1) print a[(n+1)/2]
        else printf "%.4f\n", (a[n/2] + a[n/2+1]) / 2
    }'
}

mean() {
    echo "$@" | tr ' ' '\n' | awk '{ s += $1; n++ } END { printf "%.4f\n", s / n }'
}

# Sample stddev of space-separated values
stddev() {
    echo "$@" | tr ' ' '\n' | awk '
    { sum+=$1; sumsq+=$1*$1; n++ }
    END {
        if (n < 2) { print "0.0000"; exit }
        mean = sum/n
        variance = (sumsq - n*mean*mean) / (n-1)
        if (variance < 0) variance = 0
        printf "%.4f\n", sqrt(variance)
    }'
}

# Which KV codec a one-token run resolves, read out of that run's own log: the
# engine states the codec once, in one field, and lib/bench_server.sh owns the
# reader.
resolve_kv_quant() {
    local model_path="$1" probe_pid log
    mkdir -p "${LOG_DIR}" "${SCRATCH_DIR}"
    snapshot_logs

    "${BINARY}" baseline \
        --model "${model_path}" \
        --prompt-tokens "${PROMPT_TOKENS}" \
        --max-tokens 1 \
        --max-ctx "${MAX_CTX}" \
        > /dev/null 2>&1 &
    probe_pid=$!
    wait "${probe_pid}" || true

    log="$(phase_log "${probe_pid}")" || return 1
    log_kv_quant "${log}"
}

canary_model() {
    local model_path="$1" name="${1##*/}" med sd kv_quant_tag ts_utc turbo_med turbo_sd

    echo "==> ${name}" >&2
    measure_arm "${model_path}" "" 1
    med="$(median ${DECODE[@]+"${DECODE[@]}"})"
    sd="$(stddev ${DECODE[@]+"${DECODE[@]}"})"

    # The engine's own name for the codec goes into the row verbatim.
    kv_quant_tag="$(resolve_kv_quant "${model_path}" 2>/dev/null || true)"
    if [[ -z "${kv_quant_tag}" ]]; then
        kv_quant_tag="auto"
    fi

    ts_utc="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo "${ts_utc},${GIT_SHA},${name},${kv_quant_tag},${PROMPT_TOKENS},${med},${sd},${BUILD_PROFILE}" >> "${CSV}"

    # The runs.db record `make canary-gate` reads: one run, same protocol as the
    # measured ones.
    echo "  recording into runs.db..." >&2
    run_once "" gate "${model_path}" > /dev/null
    echo "  runs.db: ok" >&2

    printf "%s  decode_tps=%.2f ± %.2f\n" "${name}" "${med}" "${sd}"

    # Informational: `auto` resolves to bf16, so the turbo arm is the compressed-KV rate.
    measure_arm "${model_path}" k8vturbo3 0 --kv-quant k8vturbo3
    if [[ ${#DECODE[@]} -ge 1 ]]; then
        turbo_med="$(median ${DECODE[@]+"${DECODE[@]}"})"
        turbo_sd="$(stddev ${DECODE[@]+"${DECODE[@]}"})"
        echo "${ts_utc},${GIT_SHA},${name},k8vturbo3,${PROMPT_TOKENS},${turbo_med},${turbo_sd},${BUILD_PROFILE}" >> "${CSV}"
        printf "%s  [k8vturbo3] decode_tps=%.2f ± %.2f\n" "${name}" "${turbo_med}" "${turbo_sd}"
    fi
}

if [[ -n "${PRESET}" ]]; then
    MODEL_DIRS=(
        mlx-community__gemma-4-e4b-it-mxfp8
        mlx-community__Qwen3.8-27B-4bit
        prism-ml__Ternary-Bonsai-2-27B-mlx-2bit
    )
    if ${INCLUDE_26B}; then
        MODEL_DIRS+=(mlx-community__gemma-4-26b-a4b-it-mxfp8)
    fi

    # Every run records, so an identity the DB would refuse stops the canary
    # here, before any model is loaded, and not at its first measured run.
    for dir in ${MODEL_DIRS[@]+"${MODEL_DIRS[@]}"}; do
        [[ -d "${MODELS_ROOT}/${dir}" ]] || continue
        if ! python3 -c 'import sys; sys.path.insert(0, sys.argv[1]); import snapshot_identity; sys.argv[1:] = sys.argv[2:]; sys.exit(snapshot_identity.main())' \
            "${REPO_ROOT}/scripts/lib" "${MODELS_ROOT}/${dir}" > /dev/null; then
            echo "ERROR: ${MODELS_ROOT}/${dir}: its identity cannot be recorded (above); nothing was measured" >&2
            exit 1
        fi
    done

    for dir in ${MODEL_DIRS[@]+"${MODEL_DIRS[@]}"}; do
        if [[ ! -d "${MODELS_ROOT}/${dir}" ]]; then
            echo "SKIP ${dir}: model path not found at ${MODELS_ROOT}/${dir}" >&2
            continue
        fi
        canary_model "${MODELS_ROOT}/${dir}"
    done
else
    echo "==> ${KV_QUANT} × ${MODEL_PATH##*/}" >&2
    measure_arm "${MODEL_PATH}" "" 1 --kv-quant "${KV_QUANT}"
    printf "%s × %s: mean decode_tps=%.1f (±%.2f), prefill_tps=%.1f\n" \
        "${KV_QUANT}" "${MODEL_PATH##*/}" \
        "$(mean ${DECODE[@]+"${DECODE[@]}"})" "$(stddev ${DECODE[@]+"${DECODE[@]}"})" "$(mean ${PREFILL[@]+"${PREFILL[@]}"})"
fi

echo "" >&2
if [[ -n "${PRESET}" ]]; then
    echo "LEGACY CSV: ${CSV}" >&2
    echo "DB: ${DATA_ROOT}/metrics/runs.db (use 'make canary-gate SHA=<sha>' to gate)" >&2
fi
if [[ -s "${PENDING}" ]]; then
    echo "${PRESET:-cell} INCOMPLETE: $(wc -l <"${PENDING}" | tr -d ' ') diagnostic record(s) did not land (named above)" >&2
    exit "${RECORD_PENDING_EXIT}"
fi
