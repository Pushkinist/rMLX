#!/usr/bin/env bash
# baseline_cells.sh — `rmlx baseline` cells: canary, one codec, inertness sweep, cache-type matrix
#
# Usage:  bash scripts/bench/baseline_cells.sh --preset canary --models-root DIR [--include-26b]
#             [--binary PATH]
#         bash scripts/bench/baseline_cells.sh --model PATH --kv-quant CODEC [--prompt-tokens N]
#             [--max-tokens N] [--warmup N] [--runs N] [--record diagnostic] [--label L]
#             [--binary PATH]
#         bash scripts/bench/baseline_cells.sh --measure inertness --model PATH [--kv-quant CODEC]...
#             [--prompt-tokens N] [--max-tokens N] [--max-ctx N] [--kv-boundary-layers H,T]
#             [--csv PATH] [--binary PATH]
#         bash scripts/bench/baseline_cells.sh --preset cache-types --model PATH --arch-class A
#             [--prompt-tokens N] [--binary PATH]
# Exit:   0 every run measured; 1 a run failed, gave no reading, or the gate record did not land;
#         2 a usage error or a missing model; 11 passthrough of rmlx (another process holds the
#         Metal claim); 75 a run measured and its record did not land; 125 the binary is missing.
#         The inertness sweep ends 0 or 75 whatever its runs did: a run that failed keeps its row.
#         The cache-types matrix ends 0 or 75 whatever its cells did: a cell that failed keeps its row.
# Not a mode of: perf_ab.sh — that one interleaves the arms of two builds; this one measures one build
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
# --measure inertness asks whether a KV codec changes anything a caller can see. For one model and
# one prompt length it runs `rmlx baseline` once per codec (no --kv-quant: the default list, one
# representative per parameterised family) at temperature 0, no warmup, and appends one row per
# codec to <data root>/bench/codec_inertness.csv (or --csv). Every run is recorded `diagnostic`
# under the label codec_inertness:<utc stamp>:<codec>, so no row reaches a champion or a gate.
#   kv_cache_bytes  resident KV: a codec that saves no bytes is not a memory lever
#   ids_sha         sha256 of the greedy token ids, 16 hex digits, empty when the run printed none:
#                   the `none` arm's digest means the codec is not on the decode read path, which
#                   decoded text cannot show
#   store_skipped   1 when any layer logged `exit_prefill: packed store skipped`. Corroboration only,
#                   not a per-codec fact: the head/tail layers of the iso and rotor codecs are
#                   promoted to K8V8, so those store-reading codecs set it too
# Columns, in order: timestamp,model,prompt_tokens,prompt_tokens_measured,max_ctx,max_tokens,codec,
# exit_code,kv_cache_bytes,ids_sha,store_skipped,ttft_ms,decode_tps,prefill_tps,binary_sha,
# kv_boundary,recorded_run_class,record_status. `docs/KV_QUANT.md` § "Codec disposition" says what a
# row decides; `ingest/ingest.py codec-inertness` reads the CSV.
#   exit_code       0 when the run measured (a pending record included), else its own exit code
#   record_status   landed, pending (the run measured and its record did not land: its buffer file
#                   is named on stderr and the tool ends 75), or empty for a run that did not measure
#   kv_boundary     the --kv-boundary-layers value, quoted because it holds a comma; empty is the
#                   engine default. It is a cell-identity term: another head/tail count is another cell
#   prompt_tokens   the fixture name (4096 selects prompts/longctx_4k.json); prompt_tokens_measured
#                   is what the run tokenized, ~5% more after the chat template
#   ttft_ms, decode_tps, prefill_tps are context only: single unpaired runs on a shared host
# --max-ctx defaults to 1.25 x the prompt + --max-tokens + 2048: a ceiling of prompt + generation
# is refused mid-prefill, because the chat template lengthens the prompt and the ring rounds up.
# --max-tokens 200 or more for a disposition run: at short lengths, codecs that differ can give
# one digest. A CSV on disk under another header is refused with exit 2.
#
# --preset cache-types runs one model through a fixed list of KV cache-type combinations. Per combo:
#   1. a smoke probe, 16 tokens at 4096 prompt tokens, recorded `diagnostic` under the label
#      ctype-smoke-<combo>. It is rejected when the run failed (exit 78 is a resolver skip), the
#      decode rate is below 0.5, it printed no `baseline:` line, its `output:` text is not UTF-8, or
#      one four-word sequence of it repeats more than 3 times.
#   2. 2 warmup and 3 measured runs of 32 tokens, none recorded one by one. One `gate` record holds
#      the median decode rate (with the stddev), ttft, load time and peak RSS of the last measured
#      run, filed through `rmlx metrics record --file`.
# A median below 0.5 files no record. Every combo ends in one summary row:
# label, kv_quant, decode_tps, decode_stddev, status (ok, skip, smoke_fail, runtime_fail or
# fail_rc<N>, N the exit code of the run, or 9 when the record did not land). --prompt-tokens
# takes a multiple of 1024 and picks prompts/longctx_<N/1024>k.json; the context is 8192 up to 4096
# prompt tokens, 16384 up to 8192, 32768 up to 16384 and twice the prompt above that.
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
KV_QUANTS=()
MEASURE=""
CSV_PATH=""
KV_BOUNDARY=""
RECORD_CLASS=""
LABEL=""
CELL_OPTIONS=""
RATE_OPTIONS=""
INERTNESS_OPTIONS=""
ARCH_CLASS=""
KEEP_FAILED_RUNS=false
PROMPT_TOKENS=4096
MAX_TOKENS=100
WARMUP_RUNS=1
MEASURED_RUNS=3
MAX_CTX=""
BUILD_PROFILE="release-perf"

INERTNESS_HEADER="timestamp,model,prompt_tokens,prompt_tokens_measured,max_ctx,max_tokens,codec,exit_code,kv_cache_bytes,ids_sha,store_skipped,ttft_ms,decode_tps,prefill_tps,binary_sha,kv_boundary,recorded_run_class,record_status"
# One representative per parameterised family: 27 of the 28 codecs `ALL_KV_QUANTS` holds, with no
# `rotor_k_3_asym_*` entry.
DEFAULT_CODECS=(
    none
    k8v8 k8v4 planar planar3 planar_k
    k8vturbo3 k8vturbo3tcq k8vturbo2 k8vturbo2tcq
    tsym3 tsym4
    iso3 iso4 rotor3 rotor4
    rotor_k_4_asym_v4_g64
    mixed_k8g64_v4g64 rot_k_v8g64
    iso3_sym iso4_sym k_iso3 k_iso4
    rotor3_sym rotor4_sym k_rotor3 k_rotor4
)

# The cache-types matrix, one combo per entry:
#   label|cache-type-k|cache-type-v|kv_quant preset|kv_quant the summary shows when no record was filed
# A combo names a per-side pair or a preset, never both.
CACHE_TYPE_COMBOS=(
    "auto|||auto|auto"
    "bf16|bf16|bf16||ct_bf16_bf16"
    "k8v4|||k8v4|k8v4"
    "k8v8|||k8v8|k8v8"
    "planar|||planar|planar"
    "q8g128_q4g64|q8_g128|q4_g64||mixed_k8g128_v4g64"
    "q8g128_q8g64|q8_g128|q8_g64||mixed_k8g128_v8g64"
    "q8g128_q4g128|q8_g128|q4_g128||mixed_k8g128_v4g128"
    "q8g128_q6g64|q8_g128|q6_g64||mixed_k8g128_v6g64"
    "q8g128_q8g32|q8_g128|q8_g32||mixed_k8g128_v8g32"
    "q4g64_q4g64|q4_g64|q4_g64||mixed_k4g64_v4g64"
    "q8g128_tq4|q8_g128|tq4||k8v4"
    "q8g128_planar4|q8_g128|planar4||planar"
)

usage_error() {
    echo "ERROR: $1" >&2
    exit 2
}

# The identity columns of a snapshot, as `name=value` lines. lib/ is imported, never run.
read_snapshot_identity() {
    python3 -c 'import sys; sys.path.insert(0, sys.argv[1]); import snapshot_identity; sys.argv[1:] = sys.argv[2:]; sys.exit(snapshot_identity.main())' \
        "${REPO_ROOT}/scripts/lib" "$1"
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
        --arch-class) ARCH_CLASS="$(value_of "$@")"; CELL_OPTIONS+=" $1" ;;
        --kv-quant) KV_QUANTS+=("$(value_of "$@")"); CELL_OPTIONS+=" $1" ;;
        --prompt-tokens) PROMPT_TOKENS="$(value_of "$@")"; CELL_OPTIONS+=" $1" ;;
        --max-tokens) MAX_TOKENS="$(value_of "$@")"; CELL_OPTIONS+=" $1" ;;
        --warmup) WARMUP_RUNS="$(value_of "$@")"; CELL_OPTIONS+=" $1"; RATE_OPTIONS+=" $1" ;;
        --runs) MEASURED_RUNS="$(value_of "$@")"; CELL_OPTIONS+=" $1"; RATE_OPTIONS+=" $1" ;;
        --record) RECORD_CLASS="$(value_of "$@")"; CELL_OPTIONS+=" $1"; RATE_OPTIONS+=" $1" ;;
        --label) LABEL="$(value_of "$@")"; CELL_OPTIONS+=" $1"; RATE_OPTIONS+=" $1" ;;
        --measure) MEASURE="$(value_of "$@")"; CELL_OPTIONS+=" $1" ;;
        --max-ctx) MAX_CTX="$(value_of "$@")"; CELL_OPTIONS+=" $1"; INERTNESS_OPTIONS+=" $1" ;;
        --kv-boundary-layers) KV_BOUNDARY="$(value_of "$@")"; CELL_OPTIONS+=" $1"; INERTNESS_OPTIONS+=" $1" ;;
        --csv) CSV_PATH="$(value_of "$@")"; CELL_OPTIONS+=" $1"; INERTNESS_OPTIONS+=" $1" ;;
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
    cache-types)
        [[ -n "${MODEL_PATH}" ]] || usage_error "--preset cache-types needs --model"
        [[ -n "${ARCH_CLASS}" ]] || usage_error "--preset cache-types needs --arch-class"
        other_options=""
        for option in ${CELL_OPTIONS}; do
            case "${option}" in
                --model | --arch-class | --prompt-tokens) ;;
                *) other_options+=" ${option}" ;;
            esac
        done
        [[ -z "${other_options}" ]] || usage_error "--preset cache-types takes none of:${other_options}"
        if [[ -n "${MODELS_ROOT}" ]] || ${INCLUDE_26B}; then
            usage_error "--models-root and --include-26b belong to --preset canary"
        fi
        [[ -d "${MODEL_PATH}" ]] || usage_error "model path not found: ${MODEL_PATH}"
        SNAPSHOT_IDENTITY="$(read_snapshot_identity "${MODEL_PATH}")" ||
            usage_error "cannot label the snapshot's weights (above)"
        WEIGHT_QUANT="$(sed -n 's/^weight_quant=//p' <<<"${SNAPSHOT_IDENTITY}")"
        WARMUP_RUNS=2
        KEEP_FAILED_RUNS=true
        ;;
    "")
        [[ -n "${MODEL_PATH}" ]] || usage_error "--model is required without --preset"
        [[ -z "${ARCH_CLASS}" ]] || usage_error "--arch-class belongs to --preset cache-types"
        case "${MEASURE}" in
            "")
                [[ ${#KV_QUANTS[@]} -gt 0 ]] || usage_error "--kv-quant is required without --preset"
                [[ ${#KV_QUANTS[@]} -eq 1 ]] || usage_error "--kv-quant takes one codec without --measure inertness"
                KV_QUANT="${KV_QUANTS[0]}"
                [[ -z "${INERTNESS_OPTIONS}" ]] || usage_error "${INERTNESS_OPTIONS# } belong to --measure inertness"
                ;;
            inertness)
                [[ -z "${RATE_OPTIONS}" ]] || usage_error "--measure inertness takes none of:${RATE_OPTIONS}"
                ;;
            *) usage_error "unknown measure: ${MEASURE}" ;;
        esac
        if [[ -n "${MODELS_ROOT}" ]] || ${INCLUDE_26B}; then
            usage_error "--models-root and --include-26b belong to --preset canary"
        fi
        [[ -z "${RECORD_CLASS}" || "${RECORD_CLASS}" == diagnostic ]] || usage_error "--record takes diagnostic"
        [[ -d "${MODEL_PATH}" ]] || usage_error "model path not found: ${MODEL_PATH}"
        [[ -n "${MEASURE}" || -n "${LABEL}" ]] || LABEL="codec_cell:$(date -u +%Y%m%dT%H%M%SZ):${KV_QUANT}"
        ;;
    *) usage_error "unknown preset: ${PRESET}" ;;
esac
if ! [[ "${PROMPT_TOKENS}" =~ ^[1-9][0-9]*$ && "${MAX_TOKENS}" =~ ^[1-9][0-9]*$ &&
    "${MEASURED_RUNS}" =~ ^[1-9][0-9]*$ && "${WARMUP_RUNS}" =~ ^[0-9]+$ &&
    "${MAX_CTX:-1}" =~ ^[1-9][0-9]*$ ]]; then
    usage_error "--prompt-tokens, --max-tokens, --max-ctx and --runs take a positive integer, --warmup a non-negative one"
fi
if [[ "${MEASURE}" == inertness ]]; then
    MAX_CTX="${MAX_CTX:-$((PROMPT_TOKENS * 5 / 4 + MAX_TOKENS + 2048))}"
    STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
    RECORD_CLASS=diagnostic
    KEEP_FAILED_RUNS=true
fi
if [[ "${PRESET}" == cache-types ]]; then
    (( PROMPT_TOKENS % 1024 == 0 )) || usage_error "--prompt-tokens takes a multiple of 1024 for --preset cache-types"
fi

TARGET_DIR="${REPO_ROOT}/target"
if [[ -z "${BINARY}" ]]; then
    for profile in release-perf release; do
        BINARY="${TARGET_DIR}/${profile}/rmlx"
        # The canary files build_profile=release-perf, so it takes no other build.
        if [[ -x "${BINARY}" || "${PRESET}" == canary ]]; then
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
boundary_args=()
if [[ "${MEASURE}" == inertness ]]; then
    ctx_args=(--max-ctx "${MAX_CTX}")
    if [[ -n "${KV_BOUNDARY}" ]]; then
        boundary_args=(--kv-boundary-layers "${KV_BOUNDARY}")
    fi
    if [[ ${#KV_QUANTS[@]} -eq 0 ]]; then
        KV_QUANTS=(${DEFAULT_CODECS[@]+"${DEFAULT_CODECS[@]}"})
    fi
    BINARY_SHA="$(shasum -a 256 "${BINARY}" | cut -c1-12)"
    CSV="${CSV_PATH:-${DATA_ROOT}/bench/codec_inertness.csv}"
    mkdir -p "$(dirname "${CSV}")"
    if [[ ! -f "${CSV}" ]]; then
        echo "${INERTNESS_HEADER}" > "${CSV}"
    elif [[ "$(head -1 "${CSV}")" != "${INERTNESS_HEADER}" ]]; then
        # Rows appended under another header would land in the wrong columns, silently.
        echo "ERROR: ${CSV} has a header this tool does not write; pass --csv with a new path" >&2
        exit 2
    fi
elif [[ -n "${PRESET}" ]]; then
    if [[ "${PRESET}" == canary ]]; then
        ctx_args=(--max-ctx "${MAX_CTX}")
        mkdir -p "${DATA_ROOT}/bench"
        if [[ ! -f "${CSV}" ]]; then
            echo "ts_utc,git_sha,model,kv_quant,prompt_tokens,decode_tps,stddev,build_profile" > "${CSV}"
        fi
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
    if [[ "${PRESET}" == canary ]]; then
        echo "canary sha=${GIT_SHA}${1:+ $1}"
    elif [[ "${MEASURE}" == inertness ]]; then
        echo "codec_inertness:${STAMP}:${1}"
    else
        echo "${LABEL}"
    fi
}

# One `rmlx baseline`, recorded under <class> when it is not empty; prints
# "<decode_tps>:<prefill_tps>", a missing reading as 0. A run that measured and
# whose record did not land (exit RECORD_PENDING_EXIT) keeps its reading and is
# noted in PENDING, except a gate run, which the canary cannot go on without.
# Any other failure stops the tool, except in the inertness sweep and the cache-types matrix, which
# keep the failed run's row. Leaves the run's readings, exit code, stdout and stderr in RUN_DECODE,
# RUN_PREFILL, RUN_RC, RUN_OUT and RUN_LOG.
# Usage: run_once <phase> <class> <model_path> [baseline args...]
run_once() {
    local phase="$1" class="$2" model_path="$3"
    shift 3
    local record_args=() out rc=0 err ptps
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
    RUN_RC="${rc}"
    RUN_OUT="${out}"
    RUN_LOG="$(cat "${err}")"
    if [[ "${rc}" -eq "${RECORD_PENDING_EXIT}" && "${class}" != gate ]]; then
        record_pending_note "${err}"
        echo "${model_path##*/} ${phase}" >>"${PENDING}"
    elif [[ "${rc}" -ne 0 && "${KEEP_FAILED_RUNS}" == false ]]; then
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
    RUN_DECODE="$(printf '%s\n' "${out}" | grep -o 'decode_tps=[0-9.]*' | cut -d= -f2 || true)"
    ptps="$(printf '%s\n' "${out}" | grep -o 'prefill_tps=[0-9.]*' | cut -d= -f2 || true)"
    RUN_PREFILL="${ptps:-0}"
}

# Warmup runs, then measured runs, filling DECODE and PREFILL with the readings.
# <arm> names the arm in the progress lines and run phases; empty is the default
# arm. A measured run with no reading stops the tool when <strict> is 1, else it
# is warned about and skipped. A run that failed ends the arm with that run's exit
# code (only a tool that keeps failed runs gets here).
# Usage: measure_arm <model_path> <arm> <strict> [baseline args...]
measure_arm() {
    local model_path="$1" arm="$2" strict="$3" i dtps ptps ok
    shift 3
    local tag="${arm:+[${arm}] }"
    DECODE=()
    PREFILL=()
    for ((i = 1; i <= WARMUP_RUNS; i++)); do
        echo "  ${tag}warmup ${i}/${WARMUP_RUNS}..." >&2
        run_once "${arm:+${arm}-}warmup" "${RECORD_CLASS}" "${model_path}" "$@"
        [[ "${RUN_RC}" -eq 0 || "${RUN_RC}" -eq "${RECORD_PENDING_EXIT}" ]] || return "${RUN_RC}"
    done
    for ((i = 1; i <= MEASURED_RUNS; i++)); do
        echo "  ${tag}measured run ${i}/${MEASURED_RUNS}..." >&2
        run_once "${arm:-measured}" "${RECORD_CLASS}" "${model_path}" "$@"
        [[ "${RUN_RC}" -eq 0 || "${RUN_RC}" -eq "${RECORD_PENDING_EXIT}" ]] || return "${RUN_RC}"
        dtps="${RUN_DECODE}"
        ptps="${RUN_PREFILL}"
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
    run_once "" gate "${model_path}"
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

# One codec of the inertness sweep: its run, then its CSV row. exit_code is 0 for a run that
# measured, a pending record included, and the run's own code otherwise.
# Usage: inertness_codec <codec>
inertness_codec() {
    local codec="$1" exit_code recorded="" record_status="" ids_sha="" skipped=0
    local summary kv_bytes measured ttft dtps ptps
    run_once "${codec}" diagnostic "${MODEL_PATH}" --kv-quant "${codec}" \
        ${boundary_args[@]+"${boundary_args[@]}"} --emit-token-ids --log debug
    exit_code="${RUN_RC}"
    if [[ "${exit_code}" -eq 0 ]]; then
        recorded=diagnostic
        record_status=landed
    elif [[ "${exit_code}" -eq "${RECORD_PENDING_EXIT}" ]]; then
        recorded=diagnostic
        record_status=pending
        exit_code=0
    fi

    # `token_ids=` is printed only for a completed generation; the digest of an empty
    # string is not a result.
    if grep -q '^baseline: token_ids=' <<<"${RUN_OUT}"; then
        ids_sha="$(sed -n 's/^baseline: token_ids=//p' <<<"${RUN_OUT}" | sed -n 1p | shasum -a 256 | cut -c1-16)"
    fi
    if grep -q 'packed store skipped' <<<"${RUN_OUT}${RUN_LOG}"; then
        skipped=1
    fi

    summary="$(grep -m1 '^baseline: model=' <<<"${RUN_OUT}" || true)"
    kv_bytes="$(sed -n 's/.*kv_cache_bytes=\([0-9]*\).*/\1/p' <<<"${summary}")"
    measured="$(sed -n 's/.*prompt_tokens=\([0-9]*\).*/\1/p' <<<"${summary}")"
    ttft="$(sed -n 's/.*ttft_ms=\([0-9.]*\).*/\1/p' <<<"${summary}")"
    dtps="$(sed -n 's/.*decode_tps=\([0-9.]*\).*/\1/p' <<<"${summary}")"
    ptps="$(sed -n 's/.*prefill_tps=\([0-9.]*\).*/\1/p' <<<"${summary}")"

    # kv_boundary holds a comma, so it is quoted: unquoted it would be two columns.
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,"%s",%s,%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(basename "${MODEL_PATH}")" "${PROMPT_TOKENS}" "${measured}" \
        "${MAX_CTX}" "${MAX_TOKENS}" "${codec}" "${exit_code}" "${kv_bytes}" "${ids_sha}" "${skipped}" \
        "${ttft}" "${dtps}" "${ptps}" "${BINARY_SHA}" "${KV_BOUNDARY}" "${recorded}" "${record_status}" >>"${CSV}"
    printf '  %-24s rc=%-3s bytes=%-12s ids=%-16s skipped=%s record=%s\n' \
        "${codec}" "${RUN_RC}" "${kv_bytes:-NONE}" "${ids_sha:-NONE}" "${skipped}" "${record_status:-none}"
}

# Appends one row to the matrix summary.
# Usage: types_row <label> <kv_quant> <decode_tps> <decode_stddev> <status>
types_row() {
    SUMMARY_ROWS+="$(printf '%s\t%s\t%s\t%s\t%s' "$@")"$'\n'
}

# Sets the size of the next runs.
# Usage: types_run_size <prompt tokens> <generated tokens> <context>
types_run_size() {
    PROMPT_TOKENS="$1"
    MAX_TOKENS="$2"
    MAX_CTX="$3"
    ctx_args=(--max-ctx "$3")
}

# Why the smoke run that just ended is rejected, or nothing when it passes.
# Usage: smoke_rejection <exit code of the run>
smoke_rejection() {
    local rc="$1" output repeat
    if [[ "${rc}" -eq "${RECORD_PENDING_EXIT}" ]]; then
        rc=0
    fi
    if [[ "${rc}" -eq 78 ]]; then
        echo resolver_skip
        return
    fi
    if [[ "${rc}" -ne 0 ]]; then
        echo "baseline_exit_${rc}"
        return
    fi
    if [[ -n "${RUN_DECODE}" ]] && awk -v t="${RUN_DECODE}" 'BEGIN { exit !(t + 0 < 0.5) }'; then
        echo "tps_zero=${RUN_DECODE}"
        return
    fi
    output="$(LC_ALL=C sed -n 's/^output: //p' <<<"${RUN_OUT}" | tail -1)"
    if [[ -z "${output}" ]]; then
        grep -q '^baseline: ' <<<"${RUN_OUT}" || echo no_output
        return
    fi
    if ! printf '%s' "${output}" | iconv -f utf-8 -t utf-8 >/dev/null 2>&1; then
        echo non_utf8
        return
    fi
    repeat="$(printf '%s\n' "${output}" | python3 "${REPO_ROOT}/scripts/bench/cache_types.py" repeat 2>/dev/null || echo 0)"
    if [[ "${repeat}" -gt 3 ]]; then
        echo "4gram_repeat=${repeat}"
    fi
}

# One combo: its smoke probe, then its measured cell. Adds one row to the summary.
# Usage: cache_types_combo <label> <ctk> <ctv> <kv_quant> <kv_quant shown without a record>
cache_types_combo() {
    local label="$1" ctk="$2" ctv="$3" kv_quant="$4" shown_kv="$5"
    local flags=() rejection status rc=0 med sd summary record_file recorded_kv prompt_name
    echo "" >&2
    echo "==> ${label} ctk='${ctk}' ctv='${ctv}' kv_quant='${kv_quant}'" >&2
    if [[ -n "${ctk}" ]]; then
        flags=(--cache-type-k "${ctk}" --cache-type-v "${ctv}")
    elif [[ "${kv_quant}" != auto ]]; then
        flags=(--kv-quant "${kv_quant}")
    fi

    types_run_size 4096 16 8192
    LABEL="ctype-smoke-${label}"
    run_once smoke diagnostic "${MODEL_PATH}" ${flags[@]+"${flags[@]}"}
    rejection="$(smoke_rejection "${RUN_RC}")"
    if [[ "${rejection}" == resolver_skip ]]; then
        echo "CELL SKIP (resolver) label=${label}" >&2
        types_row "${label}" "${shown_kv}" "" "" skip
        return
    elif [[ -n "${rejection}" ]]; then
        status=smoke_fail
        [[ "${rejection}" != tps_zero=* ]] || status=runtime_fail
        echo "CELL SMOKE_FAIL label=${label} reason=${rejection} status=${status}" >&2
        types_row "${label}" "${shown_kv}" "" "" "${status}"
        return
    fi

    prompt_name="longctx_$((TYPES_PROMPT_TOKENS / 1024))k"
    if [[ ! -f "${REPO_ROOT}/prompts/${prompt_name}.json" ]]; then
        echo "CELL FAIL label=${label}: prompts/${prompt_name}.json is missing" >&2
        types_row "${label}" "${shown_kv}" "" "" fail_rc8
        return
    fi
    types_run_size "${TYPES_PROMPT_TOKENS}" 32 "${TYPES_CTX}"
    LABEL="ctype-${label}"
    measure_arm "${MODEL_PATH}" "" 0 ${flags[@]+"${flags[@]}"} || rc=$?
    if [[ "${rc}" -eq 78 ]]; then
        echo "CELL SKIP (resolver) label=${label}" >&2
        types_row "${label}" "${shown_kv}" "" "" skip
        return
    elif [[ "${rc}" -ne 0 ]]; then
        echo "CELL FAIL label=${label} rc=${rc}" >&2
        types_row "${label}" "${shown_kv}" "" "" "fail_rc${rc}"
        return
    fi

    med="$(median ${DECODE[@]+"${DECODE[@]}"})"
    sd="$(stddev ${DECODE[@]+"${DECODE[@]}"})"
    if awk -v t="${med:-0}" 'BEGIN { exit !(t + 0 < 0.5) }'; then
        echo "CELL RUNTIME_FAIL (median below 0.5) label=${label} median=${med:-0}" >&2
        types_row "${label}" "${shown_kv}" 0.0 "" runtime_fail
        return
    fi

    summary="$(grep -m1 '^baseline: model=' <<<"${RUN_OUT}" || true)"
    record_file="${DATA_ROOT}/metrics/buffer/pending/$(date -u +%Y%m%dT%H%M%SZ)-cache-types-${label}-$$.json"
    mkdir -p "${record_file%/*}"
    if ! recorded_kv="$(python3 "${REPO_ROOT}/scripts/bench/cache_types.py" record \
        --out "${record_file}" --identity-json "${RMLX_IDENTITY_JSON}" \
        ${git_sha_args[@]+"${git_sha_args[@]}"} \
        --model-path "${MODEL_PATH}" --weight-quant "${WEIGHT_QUANT}" \
        --ctk "${ctk}" --ctv "${ctv}" --kv-quant "${kv_quant}" \
        --ctx-max "${TYPES_CTX}" --prompt-file "${REPO_ROOT}/prompts/${prompt_name}.json" \
        --prompt-tokens "${TYPES_PROMPT_TOKENS}" --max-tokens 32 \
        --warmups "${WARMUP_RUNS}" --measured "${MEASURED_RUNS}" \
        --label "ctype-${label}" --tag "${TYPES_TAG}" --median "${med}" --stddev "${sd}" \
        --ttft "$(sed -n 's/.*ttft_ms=\([0-9.]*\).*/\1/p' <<<"${summary}")" \
        --load "$(sed -n 's/.*load=\([0-9.]*\)ms.*/\1/p' <<<"${summary}")" \
        --rss "$(sed -n 's/.*peak_rss=\([0-9.]*\)MB.*/\1/p' <<<"${summary}")")" ||
        ! "${BINARY}" metrics record --file "${record_file}"; then
        echo "CELL FAIL label=${label}: the record did not land (${record_file##*/})" >&2
        types_row "${label}" "${shown_kv}" "" "" fail_rc9
        return
    fi
    echo "CELL OK label=${label} kv=${recorded_kv} median=${med} stddev=${sd}" >&2
    types_row "${label}" "${recorded_kv}" "${med}" "${sd}" ok
}

cache_types_matrix() {
    local entry label ctk ctv kv_quant shown_kv base
    TYPES_PROMPT_TOKENS="${PROMPT_TOKENS}"
    if [[ "${TYPES_PROMPT_TOKENS}" -le 4096 ]]; then
        TYPES_CTX=8192
    elif [[ "${TYPES_PROMPT_TOKENS}" -le 8192 ]]; then
        TYPES_CTX=16384
    elif [[ "${TYPES_PROMPT_TOKENS}" -le 16384 ]]; then
        TYPES_CTX=32768
    else
        TYPES_CTX=$((TYPES_PROMPT_TOKENS * 2))
    fi
    base="${MODEL_PATH##*/}"
    TYPES_TAG="$(printf '%s' "${base#*__}" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g' | cut -c1-20)"
    TYPES_TAG="${TYPES_TAG%-}"

    echo "==> cache types × ${base}: arch=${ARCH_CLASS} weight=${WEIGHT_QUANT} ${#CACHE_TYPE_COMBOS[@]} combos" >&2
    SUMMARY_ROWS=""
    types_row label kv_quant decode_tps decode_stddev status
    for entry in ${CACHE_TYPE_COMBOS[@]+"${CACHE_TYPE_COMBOS[@]}"}; do
        IFS='|' read -r label ctk ctv kv_quant shown_kv <<<"${entry}"
        cache_types_combo "${label}" "${ctk}" "${ctv}" "${kv_quant}" "${shown_kv}"
    done

    echo "" >&2
    echo "MATRIX SUMMARY  model=${base} arch=${ARCH_CLASS} weight=${WEIGHT_QUANT}" >&2
    echo "  prompt_id=longctx_$((TYPES_PROMPT_TOKENS / 1024))k prompt_tokens=${TYPES_PROMPT_TOKENS} ctx_max=${TYPES_CTX}" >&2
    printf '%s' "${SUMMARY_ROWS}" | column -t -s "$(printf '\t')" >&2 || printf '%s' "${SUMMARY_ROWS}" >&2
}

if [[ "${MEASURE}" == inertness ]]; then
    echo "==> inertness × ${MODEL_PATH##*/}: ${#KV_QUANTS[@]} codec(s) -> ${CSV}" >&2
    for codec in ${KV_QUANTS[@]+"${KV_QUANTS[@]}"}; do
        inertness_codec "${codec}"
    done
elif [[ "${PRESET}" == cache-types ]]; then
    cache_types_matrix
elif [[ "${PRESET}" == canary ]]; then
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
        if ! read_snapshot_identity "${MODELS_ROOT}/${dir}" > /dev/null; then
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
if [[ "${PRESET}" == canary ]]; then
    echo "LEGACY CSV: ${CSV}" >&2
    echo "DB: ${DATA_ROOT}/metrics/runs.db (use 'make canary-gate SHA=<sha>' to gate)" >&2
fi
if [[ -s "${PENDING}" ]]; then
    echo "${PRESET:-${MEASURE:-cell}} INCOMPLETE: $(wc -l <"${PENDING}" | tr -d ' ') diagnostic record(s) did not land (named above)" >&2
    exit "${RECORD_PENDING_EXIT}"
fi
